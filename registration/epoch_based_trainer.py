import torch
import torch.distributed
import abc
import argparse
import os, time, sys
import json

from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter

from utils.timer import Timer
from utils.summary_board import SummaryBoard
from utils.torch import initialize, release_cuda, all_reduce_tensors
from engine import Logger


def inject_default_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('--resume', default=False, help='resume training')
    parser.add_argument('--snapshot', default=None, help='load from snapshot')  # None
    parser.add_argument('--epoch', type=int, default=None, help='load epoch')
    parser.add_argument('--log_steps', type=int, default=10, help='logging steps') 
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank for ddp')

    return parser

class EpochBasedTrainer(abc.ABC):
    def __init__(self,
        cfg,
        max_epoch,
        parser=None,
        cudnn_deterministic=True,
        autograd_anomaly_detection=False,
        save_all_snapshots=True,
        run_grad_check=False,
        grad_acc_steps=1,
    ):
        self.max_epoch = max_epoch

        # Additional parser
        parser = inject_default_parser(parser)
        self.args = parser.parse_args()

        # logger
        log_file = os.path.join(cfg.log_dir, 'train-{}.log'.format(time.strftime('%Y%m%d-%H%M%S')))
        self.logger = Logger(log_file=log_file, local_rank=self.args.local_rank)

        # command executed
        message = 'Command executed: ' + ' '.join(sys.argv)
        self.logger.info(message)

        # print config
        message = 'Configs:\n' + json.dumps(cfg, indent=4)
        self.logger.info(message)

        # tensorboard
        self.writer = SummaryWriter(log_dir=cfg.event_dir)
        self.logger.info(f'Tensorboard is enabled. Write events to {cfg.event_dir}.')

        # cuda and distributed
        if not torch.cuda.is_available():
            raise RuntimeError('No CUDA devices available.')
        self.distributed = self.args.local_rank != -1
        if self.distributed:
            torch.cuda.set_device(self.args.local_rank)
            torch.distributed.init_process_group(backend='nccl')
            self.world_size = torch.distributed.get_world_size()
            self.local_rank = self.args.local_rank
            self.logger.info(f'Using DistributedDataParallel mode (world_size: {self.world_size})')
        else:
            if torch.cuda.device_count() > 1:
                self.logger.warning('DataParallel is deprecated. Use DistributedDataParallel instead.')
            self.world_size = 1
            self.local_rank = 0
            self.logger.info('Using Single-GPU mode.')
        self.cudnn_deterministic = cudnn_deterministic  
        self.autograd_anomaly_detection = autograd_anomaly_detection
        self.seed = cfg.seed + self.local_rank
        initialize(                                    
            seed=self.seed,
            cudnn_deterministic=self.cudnn_deterministic,
            autograd_anomaly_detection=self.autograd_anomaly_detection,
        )

        # basic config
        self.snapshot_dir = cfg.snapshot_dir
        self.log_steps = self.args.log_steps
        self.run_grad_check = run_grad_check
        self.save_all_snapshots = save_all_snapshots

        # state                                        
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.epoch = 0
        self.iteration = 0                             
        self.inner_iteration = 0                        

        self.train_loader = None
        self.val_loader = None
        self.summary_board = SummaryBoard(last_n=self.log_steps, adaptive=True)  
        self.timer = Timer()
        self.saved_states = {}

        # training config
        self.training = True
        self.grad_acc_steps = grad_acc_steps            

    def save_snapshot(self, filename):
        if self.local_rank != 0:
            return
        model_state_dict = self.model.state_dict()
        # Remove '.module' prefix in DistributedDataParallel mode.
        if self.distributed:
            model_state_dict = OrderedDict([(key[7:], value) for key, value in model_state_dict.items()])

        # save model                                    
        filename = os.path.join(self.snapshot_dir, filename)
        state_dict = {
            'epoch': self.epoch,
            'iteration': self.iteration,
            'model': model_state_dict,
        }
        torch.save(state_dict, filename)
        self.logger.info('Model saved to "{}"'.format(filename))

        # save snapshot                                
        snapshot_filename = os.path.join(self.snapshot_dir, 'snapshot.pth.tar')
        if self.scheduler is not None:
            state_dict['optimizer'] = self.optimizer.state_dict()
            state_dict['scheduler'] = self.scheduler.state_dict()
        torch.save(state_dict, snapshot_filename)
        self.logger.info('Snapshot saved to "{}"'.format(snapshot_filename))
 

    def load_snapshot(self, snapshot, fix_prefix=True):             
        self.logger.info('Loading from "{}".'.format(snapshot))
        state_dict = torch.load(snapshot, map_location=torch.device('cpu'))

        # Load model
        model_dict = state_dict['model']
        if fix_prefix and self.distributed:
            model_dict = OrderedDict([('module.' + key, value) for key, value in model_dict.items()])
        self.model.load_state_dict(model_dict, strict=False)          # ***********************

        if 'ppo' in model_dict:
            self.model.ppo.policy.load_state_dict(model_dict['ppo']['policy'])
            self.model.ppo.policy_old.load_state_dict(model_dict['ppo']['policy'])

        # log missing keys and unexpected keys                        
        snapshot_keys = set(model_dict.keys())
        model_keys = set(self.model.state_dict().keys())
        missing_keys = model_keys - snapshot_keys
        unexpected_keys = snapshot_keys - model_keys
        if self.distributed:
            missing_keys = set([missing_key[7:] for missing_key in missing_keys])
            unexpected_keys = set([unexpected_key[7:] for unexpected_key in unexpected_keys])
        if len(missing_keys) > 0:
            message = f'Missing keys: {missing_keys}'
            self.logger.warning(message)
        if len(unexpected_keys) > 0:
            message = f'Unexpected keys: {unexpected_keys}'
            self.logger.warning(message)
        self.logger.info('Model has been loaded.')

        # Load other attributes                                      
        if self.args.resume or self.args.snapshot is not None:
            if 'epoch' in state_dict:
                self.epoch = state_dict['epoch']
                self.logger.info('Epoch has been loaded: {}.'.format(self.epoch))
            if 'iteration' in state_dict:
                self.iteration = state_dict['iteration']
                self.logger.info('Iteration has been loaded: {}.'.format(self.iteration))
            if 'optimizer' in state_dict and self.optimizer is not None:
                self.optimizer.load_state_dict(state_dict['optimizer'])
                self.logger.info('Optimizer has been loaded.')
            if 'scheduler' in state_dict and self.scheduler is not None:    
                self.scheduler.load_state_dict(state_dict['scheduler'])
                self.logger.info('Scheduler has been loaded.')


    def check_invalid_gradients(self, epoch, iteration, data_dict, output_dict, result_dict):
        for name, param in self.model.parameters():
            if torch.isnan(param.grad).any():
                self.logger.error('NaN in gradients.')
                return False
            if torch.isinf(param.grad).any():
                self.logger.error('Inf in gradients.')
                return False
        return True

    def release_tensors(self, result_dict):
        r"""All reduce and release tensors."""
        if self.distributed:
            result_dict = all_reduce_tensors(result_dict, world_size=self.world_size)
        result_dict = release_cuda(result_dict)                        
        return result_dict

    def write_event(self, phase, event_dict, index):
        r"""Write TensorBoard event."""
        if self.local_rank != 0:
            return
        for key, value in event_dict.items():
            self.writer.add_scalar(f'{phase}/{key}', value, index)

    @abc.abstractmethod
    def run(self):
        raise NotImplemented




















