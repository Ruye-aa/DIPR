import argparse
import os.path as osp
import time
import torch
from tqdm import tqdm

import numpy as np

from registration.based_tester import BaseTester
from dataset import test_data_loader
from config import make_cfg
from loss import Evaluator
from model import Registration
from utils.common import ensure_dir, get_log_string
from utils.torch import release_cuda, to_cuda
from utils.timer import Timer
from utils.summary_board import SummaryBoard
import GPUtil

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=str, default='', help='load from snapshot')
    parser.add_argument('--test_epoch', type=int, default=None, help='test epoch')
    parser.add_argument('--test_iter', type=int, default=None, help='test iteration')
    parser.add_argument('--stage_nums', type=int, default=3, help='iter nums')
    parser.add_argument('--classifier_threshold', type=float, default=30.0, help='classifier threshold nums')

    parser.add_argument('--epsilon', type=float, default=5, help='dbscan cluster parameter')
    parser.add_argument('--min_samples', type=int, default=5, help='dbscan cluster parameter')
    return parser

class Tester(BaseTester):
    def __init__(self, cfg):
        super().__init__(cfg, parser=make_parser())
        # self.init_memory = GPUtil.getGPUs()[0].memoryUsed
        # print('init-GPU-Mem:', self.init_memory)
        
        # dataloader
        start_time = time.time()
        data_loader, neighbor_limits = test_data_loader(cfg)
        loading_time = time.time() - start_time
        message = f'Data loader created: {loading_time:.3f}s collapsed.'
        self.logger.info(message)
        message = f'Calibrate neighbors: {neighbor_limits}.'
        self.logger.info(message)
        self.register_loader(data_loader)

        # model
        cfg.model.stage_num = self.args.stage_nums
        cfg.model.classifier_threshold = self.args.classifier_threshold
        cfg.model.epsilon = self.args.epsilon
        cfg.model.min_samples = self.args.min_samples
        model = Registration(cfg).cuda()  # create_model(cfg).cuda()
        r"""Register model. DDP is automatically used."""
        self.model = model
        message = 'Model description:\n' # + str(model)
        self.logger.info(message)

        # evaluator
        self.evaluator = Evaluator(cfg).cuda()
    
        # preparation
        self.output_dir = osp.join(cfg.feature_dir)
        ensure_dir(self.output_dir)

    def test_step(self, iteration, data_dict):
        output_dict = self.model(data_dict)
        return output_dict

    def eval_step(self, iteration, data_dict, output_dict):
        result_dict = self.evaluator(output_dict, data_dict)
        return result_dict
    
    def summary_string(self, iteration, data_dict, output_dict, result_dict):
        seq_id = data_dict['seq_id']
        ref_frame = data_dict['ref_frame']
        src_frame = data_dict['src_frame']
        message = f'seq_id: {seq_id}, id0: {ref_frame}, id1: {src_frame}'
        message += ', ' + get_log_string(result_dict=result_dict)
        message += ', nCorr: {}'.format(output_dict['corr_scores'].shape[0])
        return message

    def after_test_step(self, iteration, data_dict, output_dict, result_dict):
        seq_id = data_dict['seq_id']
        ref_frame = data_dict['ref_frame']
        src_frame = data_dict['src_frame']

        file_name = osp.join(self.output_dir, f'{seq_id}_{src_frame}_{ref_frame}.npz')
        np.savez_compressed(
            file_name,
            ref_points=release_cuda(output_dict['ref_points']),
            src_points=release_cuda(output_dict['src_points']),
            ref_points_f=release_cuda(output_dict['ref_points_f']),
            src_points_f=release_cuda(output_dict['src_points_f']),
            ref_points_c=release_cuda(output_dict['ref_points_c']),
            src_points_c=release_cuda(output_dict['src_points_c']),
            ref_feats_c=release_cuda(output_dict['ref_feats_c']),
            src_feats_c=release_cuda(output_dict['src_feats_c']),
            ref_node_corr_indices=release_cuda(output_dict['ref_node_corr_indices']),
            src_node_corr_indices=release_cuda(output_dict['src_node_corr_indices']),
            ref_corr_points=release_cuda(output_dict['ref_corr_points']),
            src_corr_points=release_cuda(output_dict['src_corr_points']),
            corr_scores=release_cuda(output_dict['corr_scores']),
            gt_node_corr_indices=release_cuda(output_dict['gt_node_corr_indices']),
            gt_node_corr_overlaps=release_cuda(output_dict['gt_node_corr_overlaps']),
            estimated_transform=release_cuda(output_dict['estimated_transform']),
            batch_indices=release_cuda(output_dict['batch_indices']),
            transform=release_cuda(data_dict['transform']),
        )

    def run(self):
        assert self.test_loader is not None
        self.load_snapshot(self.args.snapshot)
        self.model.eval()
        torch.set_grad_enabled(False)
        summary_board = SummaryBoard(adaptive=True)
        timer = Timer()
        
        memory_usage = []
        total_corr_num = 0
        total_iterations = len(self.test_loader)
        pbar = tqdm(enumerate(self.test_loader), total=total_iterations)
        for iteration, data_dict in pbar:
            # on start
            self.iteration = iteration + 1
            # if self.iteration < 600: continue
            data_dict['iterarion'] = self.iteration
            data_dict = to_cuda(data_dict)
            # test step
            torch.cuda.synchronize()
            timer.add_prepare_time()
            

            # # test step
            result_dict, output_dict = self.model(data_dict)
            torch.cuda.synchronize()
            timer.add_process_time()

            # memory_diff = GPUtil.getGPUs()[0].memoryUsed  - self.init_memory
            # print('GPU Memory Used:', memory_diff / (1024))
            # memory_usage.append(memory_diff / (1024))
            total_corr_num += output_dict['corr_scores'].shape[0]

            self.after_test_step(self.iteration, data_dict, output_dict, result_dict)

            # logging
            result_dict = release_cuda(result_dict)
            summary_board.update_from_result_dict(result_dict)
            message = self.summary_string(self.iteration, data_dict, output_dict, result_dict) 
            message += f', {timer.tostring()}'
            # print(message)
            # pbar.set_description(message)       # debug + #
            torch.cuda.empty_cache()
        
        # print(f', {timer.tostring()}')
        # print("Mean correspondence numbers: {}".format(total_corr_num/total_iterations))
        # average_memory_usage = sum(memory_usage) / total_iterations
        # print('average memory usage:', average_memory_usage, "GB")

        summary_dict = summary_board.summary()
        message = get_log_string(result_dict=summary_dict, timer=timer)
        self.logger.critical(message)

def main():
    cfg = make_cfg()
    tester = Tester(cfg)
    tester.run()


if __name__ == '__main__':
    main()


