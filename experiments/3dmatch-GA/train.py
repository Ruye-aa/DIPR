import time
import torch
import torch.nn as nn
import torch.optim as optim

import os
import ipdb
import tqdm

from utils import to_cuda, get_log_string, Timer, SummaryBoard

from registration.epoch_based_trainer import EpochBasedTrainer
from config import make_cfg
from dataset import train_valid_data_loader
from model import Registration


class Trainer(EpochBasedTrainer):
    def __init__(self, cfg):
        super().__init__(cfg, max_epoch=cfg.optim.max_epoch)           # 调用父类的__init__方法

        # dataloader
        start_time = time.time()
        train_loader, val_loader = train_valid_data_loader(cfg, self.distributed)
        loading_time = time.time() - start_time
        message = 'Data loader created: {:.3f}s collapsed.'.format(loading_time)
        self.logger.info(message)
        # message = 'Calibrate neighbors: {}.'.format(neighbor_limits)
        # self.logger.info(message)
        self.train_loader = train_loader  # self.register_loader(train_loader, val_loader)
        self.val_loader = val_loader

        # model, optimizer, scheduler
        model = Registration(cfg).cuda()  # create_model(cfg).cuda()
        optimizer = optim.Adam(
                                model.parameters(),
                                lr=cfg.optim.lr, betas=(0.9, 0.99),weight_decay=cfg.optim.weight_decay)
        # optimizer = optim.AdamW(self.parameters(), lr=0.0001, weight_decay=0.0001)
        self.optimizer = optimizer
        scheduler =optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.optim.lr_decay)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, cfg.optim.lr_decay_steps, gamma=cfg.optim.lr_decay)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, 20, 0.5)
        self.scheduler = scheduler
            
        self.model = model
        message = 'Model description:\n' + str(model)
        self.logger.info(message)

        # loss function, evaluator
        # self.loss_func = OverallLoss(cfg).cuda()
        # self.evaluator = Evaluator(cfg).cuda()

    def train_step(self, epoch, iteration, data_dict):
        loss_dict = self.model(data_dict)
        return loss_dict

    def val_step(self, epoch, iteration, data_dict):
        result_dict, output_dict = self.model(data_dict)
        return result_dict

    def check_gradients(self, epoch, iteration, data_dict, output_dict, result_dict):
        # if not self.run_grad_check:
        #     return
        if not self.check_invalid_gradients(epoch, iteration, data_dict, output_dict, result_dict):
            self.logger.error('Epoch: {}, iter: {}, invalid gradients.'.format(epoch, iteration))
            torch.save(data_dict, 'data.pth')
            torch.save(self.model, 'model.pth')
            self.logger.error('Data_dict and model snapshot saved.')
            ipdb.set_trace()

    def train_epoch(self):
        if self.distributed:
            self.train_loader.sampler.set_epoch(self.epoch)
        total_iterations = len(self.train_loader)

        # c_loader_iter = self.train_loader.__iter__()
        # for iteration in tqdm(range(total_iterations)):
        #     data_dict = c_loader_iter.next()

        for iteration, data_dict in enumerate(self.train_loader):
            # continue
            self.inner_iteration = iteration + 1
            self.iteration += 1
            # if self.iteration <= 5021: 
                # continue
            data_dict = to_cuda(data_dict)
            self.timer.add_prepare_time()           # 从__init__的初始时间到这里经过的时间是prepare_time
            # forward
            result_dict = self.train_step(self.epoch, self.inner_iteration, data_dict)  # output_dict, 
            # backward & optimization
            result_dict['total_loss'].backward()
            # self.check_gradients(self.epoch, self.inner_iteration, data_dict, result_dict, result_dict) # output_dict, 
            if iteration % self.grad_acc_steps == 0: # self.optimizer_step(self.inner_iteration)
                self.optimizer.step()
                self.optimizer.zero_grad()

            # after training
            self.timer.add_process_time()           # 从prepare到这一步所花的时间
            result_dict = self.release_tensors(result_dict)         # 分离参数
            self.summary_board.update_from_result_dict(result_dict) # 对result_dict进行更新
            # logging
            if self.inner_iteration % self.log_steps == 0:
                summary_dict = self.summary_board.summary()
                message = get_log_string(
                    result_dict=summary_dict,
                    epoch=self.epoch,
                    max_epoch=self.max_epoch,
                    iteration=self.inner_iteration,
                    max_iteration=total_iterations,
                    lr=self.optimizer.param_groups[0]['lr'],        # self.get_lr(self)
                    timer=self.timer,
                )
                self.logger.info(message)
                self.write_event('train', summary_dict, self.iteration)

            torch.cuda.empty_cache()
        message = get_log_string(self.summary_board.summary(), epoch=self.epoch, timer=self.timer)      # 对参数进行更新
        self.logger.critical(message)
        if self.scheduler is not None:
            self.scheduler.step()
        # snapshot
        self.save_snapshot(f'epoch-{self.epoch}.pth.tar')
        if not self.save_all_snapshots:
            last_snapshot = f'epoch-{self.epoch - 1}.pth.tar'
            if os.path.exists(last_snapshot):
                os.remove(last_snapshot)

    def inference_epoch(self):
        self.training = False
        self.model.eval()
        torch.set_grad_enabled(False)
        summary_board = SummaryBoard(adaptive=True)
        timer = Timer()
        total_iterations = len(self.val_loader)
        pbar = tqdm.tqdm(enumerate(self.val_loader), total=total_iterations)
        for iteration, data_dict in pbar:
            # continue
            self.inner_iteration = iteration + 1
            data_dict = to_cuda(data_dict)
            timer.add_prepare_time()
            result_dict = self.val_step(self.epoch, self.inner_iteration, data_dict)    # output_dict, 
            torch.cuda.synchronize()                    # 等待当前设备上所有流中的所有核心完成
            timer.add_process_time()
            result_dict = self.release_tensors(result_dict)
            summary_board.update_from_result_dict(result_dict)
            message = get_log_string(
                result_dict=summary_board.summary(),
                epoch=self.epoch,
                iteration=self.inner_iteration,
                max_iteration=total_iterations,
                timer=timer,
            )
            pbar.set_description(message)
            torch.cuda.empty_cache()
        summary_dict = summary_board.summary()
        message = '[Val] ' + get_log_string(summary_dict, epoch=self.epoch, timer=timer)
        self.logger.critical(message)
        self.write_event('val', summary_dict, self.epoch)
        self.training = True                           # self.set_train_mode()
        self.model.train()
        torch.set_grad_enabled(True)

    def run(self):
        assert self.train_loader is not None
        assert self.val_loader is not None

        # if not self.args.resume and self.args.snapshot is None:
        #     global_transformer_model = torch.load('/home/anying/code/others/Registration_RoITR/weights/model_3dmatch.pth', map_location=torch.device('cpu'))
        #     # model_dict = torch.load('/home/anying/code/Registration11.14/output/weights/snapshot/geotransformer-3dmatch.pth.tar', map_location=torch.device('cpu'))['model']
        #     self.model.global_transformer.load_state_dict(global_transformer_model, strict=False)
        #     self.model.local_transformer.load_state_dict(global_transformer_model, strict=False)
        #     # self.model.conf_logits_decoder = torch.load('/home/anying/code/Registrations/output/load_overlap.pth').cuda()
        #     # self.model.conf_logits_decoder_local = torch.load('/home/anying/code/Registrations/output/load_overlap.pth').cuda()

        if self.args.resume:                        # 若需要恢复，则从log中的snapshot_dir恢复
            self.load_snapshot(os.path.join(self.snapshot_dir, 'snapshot.pth.tar'))
        elif self.args.snapshot is not None:        # 若存在snapshot，则从自己指定的snapshot中恢复
            self.load_snapshot(self.args.snapshot)
        self.training = True                        # 设置model.train(),并设置求导 self.set_train_mode()
        self.model.train()
        torch.set_grad_enabled(True)
        # self.inference_epoch()
        while self.epoch < self.max_epoch:
            self.epoch += 1
            self.train_epoch()                      # 进行训练以及评价
            self.inference_epoch()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    cfg = make_cfg()
    trainer = Trainer(cfg)
    trainer.run()

if __name__ == '__main__':
    main()