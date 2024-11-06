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
        super().__init__(cfg, max_epoch=cfg.optim.max_epoch)         

        # dataloader
        start_time = time.time()
        train_loader, val_loader, neighbor_limits = train_valid_data_loader(cfg, self.distributed)
        loading_time = time.time() - start_time
        message = 'Data loader created: {:.3f}s collapsed.'.format(loading_time)
        self.logger.info(message)
        message = 'Calibrate neighbors: {}.'.format(neighbor_limits)
        self.logger.info(message)
        self.train_loader = train_loader  
        self.val_loader = val_loader

        # model, optimizer, scheduler
        model = Registration(cfg).cuda() 
        optimizer = optim.Adam(
                                model.parameters(),
                                lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
        self.optimizer = optimizer
        scheduler = optim.lr_scheduler.StepLR(optimizer, cfg.optim.lr_decay_steps, gamma=cfg.optim.lr_decay)
        
        self.scheduler = scheduler
            
        self.model = model
        message = 'Model description:\n' + str(model)
        self.logger.info(message)

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

        for iteration, data_dict in enumerate(self.train_loader):
            self.inner_iteration = iteration + 1
            self.iteration += 1
            # if self.iteration == 21: 
            #     continue
            data_dict = to_cuda(data_dict)
            self.timer.add_prepare_time()          
            # forward
            result_dict = self.train_step(self.epoch, self.inner_iteration, data_dict)
            # backward & optimization
            result_dict['total_loss'].backward()
            # self.check_gradients(self.epoch, self.inner_iteration, data_dict, result_dict, result_dict) 
            if iteration % self.grad_acc_steps == 0: 
                self.optimizer.step()
                self.optimizer.zero_grad()

            # after training
            self.timer.add_process_time()           
            result_dict = self.release_tensors(result_dict)         
            self.summary_board.update_from_result_dict(result_dict) 
            # logging
            if self.inner_iteration % self.log_steps == 0:
                summary_dict = self.summary_board.summary()
                message = get_log_string(
                    result_dict=summary_dict,
                    epoch=self.epoch,
                    max_epoch=self.max_epoch,
                    iteration=self.inner_iteration,
                    max_iteration=total_iterations,
                    lr=self.optimizer.param_groups[0]['lr'],       
                    timer=self.timer,
                )
                self.logger.info(message)
                self.write_event('train', summary_dict, self.iteration)

            torch.cuda.empty_cache()
        message = get_log_string(self.summary_board.summary(), epoch=self.epoch, timer=self.timer)      
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
            self.inner_iteration = iteration + 1
            data_dict = to_cuda(data_dict)
            timer.add_prepare_time()
            result_dict = self.val_step(self.epoch, self.inner_iteration, data_dict) 
            torch.cuda.synchronize()                  
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
        self.training = True                           
        self.model.train()
        torch.set_grad_enabled(True)

    def run(self):
        assert self.train_loader is not None
        assert self.val_loader is not None

        if self.args.resume:                       
            self.load_snapshot(os.path.join(self.snapshot_dir, 'snapshot.pth.tar'))
        elif self.args.snapshot is not None:      
            self.load_snapshot(self.args.snapshot)
        self.training = True                        
        self.model.train()
        torch.set_grad_enabled(True)

        while self.epoch < self.max_epoch:
            self.epoch += 1
            self.train_epoch()                      
            self.inference_epoch()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    cfg = make_cfg()
    trainer = Trainer(cfg)
    trainer.run()

if __name__ == '__main__':
    main()