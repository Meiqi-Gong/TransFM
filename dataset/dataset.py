from copy import deepcopy
import math
import torch
from torch.utils.data.sampler import Sampler
from functools import partial
from os import path as osp
from options.options import parse_options
from utils import *
import time
from dataset.video_dataset_base import VideoFusionDataset, VideoFusionTestDataset


class CPUPrefetcher():
    """CPU prefetcher.

    Args:
        loader: Dataloader.
    """

    def __init__(self, loader):
        self.ori_loader = loader
        self.loader = iter(loader)

    def next(self):
        try:
            return next(self.loader)
        except StopIteration:
            return None

    def reset(self):
        self.loader = iter(self.ori_loader)

def build_dataset(dataset_opt):
    """Build dataset from options.

    Args:
        dataset_opt (dict): Configuration for dataset. It must constain:
            name (str): Dataset name.
            type (str): Dataset type.
    """
    if dataset_opt['type']=='VideoFusionDataset':
        dataset = VideoFusionDataset(dataset_opt)
    elif dataset_opt['type']=='VideoFusionTestDataset':
        dataset = VideoFusionTestDataset(dataset_opt)
    # DATASET_REGISTRY = Registry('dataset')
    # dataset_opt = deepcopy(dataset_opt)
    # dataset = DATASET_REGISTRY.get(dataset_opt['type'])(dataset_opt)
    return dataset

def build_dataloader(dataset, dataset_opt, num_gpu=1, dist=False, sampler=None, seed=None):
    """Build dataloader.

    Args:
        dataset (torch.utils.data.Dataset): Dataset.
        dataset_opt (dict): Dataset options. It contains the following keys:
            phase (str): 'train' or 'val'.
            num_worker_per_gpu (int): Number of workers for each GPU.
            batch_size_per_gpu (int): Training batch size for each GPU.
        num_gpu (int): Number of GPUs. Used only in the train phase.
            Default: 1.
        dist (bool): Whether in distributed training. Used only in the train
            phase. Default: False.
        sampler (torch.utils.data.sampler): Data sampler. Default: None.
        seed (int | None): Seed. Default: None
    """
    phase = dataset_opt['phase']
    rank, _ = get_dist_info()
    if phase == 'train':
        if dist:  # distributed training
            batch_size = dataset_opt['batch_size_per_gpu']
            num_workers = dataset_opt['num_worker_per_gpu']
        else:  # non-distributed training
            multiplier = 1 if num_gpu == 0 else num_gpu
            batch_size = dataset_opt['batch_size_per_gpu'] * multiplier
            num_workers = dataset_opt['num_worker_per_gpu'] * multiplier
        dataloader_args = dict(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            sampler=sampler,
            drop_last=True)
        if sampler is None:
            dataloader_args['shuffle'] = True
        dataloader_args['worker_init_fn'] = partial(
            worker_init_fn, num_workers=num_workers, rank=rank, seed=seed) if seed is not None else None
    elif phase in ['val', 'test']:  # validation
        dataloader_args = dict(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)
    else:
        raise ValueError(f'Wrong dataset phase: {phase}. ' "Supported ones are 'train', 'val' and 'test'.")

    dataloader_args['pin_memory'] = dataset_opt.get('pin_memory', False)

    prefetch_mode = dataset_opt.get('prefetch_mode')
    # if prefetch_mode == 'cpu':  # CPUPrefetcher
    #     num_prefetch_queue = dataset_opt.get('num_prefetch_queue', 1)
    #     logger = get_root_logger()
    #     logger.info(f'Use {prefetch_mode} prefetch dataloader: ' f'num_prefetch_queue = {num_prefetch_queue}')
    #     return PrefetchDataLoader(num_prefetch_queue=num_prefetch_queue, **dataloader_args)
    # else:
        # prefetch_mode=None: Normal dataloader
        # prefetch_mode='cuda': dataloader for CUDAPrefetcher
    return torch.utils.data.DataLoader(**dataloader_args)


def create_train_val_dataloader(opt):
    # create train and val dataloaders
    train_loader, val_loader = None, None
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = build_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)
            # print(opt)
            # import sys
            # sys.exit(0)
            train_loader = build_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])

            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio / (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
            # logger.info('Training statistics:'
            #             f'\n\tNumber of train images: {len(train_set)}'
            #             f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
            #             f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
            #             f'\n\tWorld size (gpu number): {opt["world_size"]}'
            #             f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
            #             f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')

        elif phase == 'val':
            val_set = build_dataset(dataset_opt)
            val_loader = build_dataloader(
                val_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
            # logger.info(f'Number of val images/folders in {dataset_opt["name"]}: ' f'{len(val_set)}')
        # else:
        #     raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, train_sampler, val_loader, total_epochs, total_iters

def create_test_dataloader(opt):
    # create train and val dataloaders
    test_loader = None, None
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'test':
            test_set = build_dataset(dataset_opt)
            test_loader = build_dataloader(
                test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
            print(f'Number of val images/folders in {dataset_opt["name"]}: ' f'{len(test_set)}')
        # else:
        #     raise ValueError(f'Dataset phase {phase} is not recognized.')

    return test_loader

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    opt = parse_options(root_path, is_train=True)
    resume_state = load_resume_state(opt)
    opt['root_path'] = root_path
    # mkdir for experiments and logger
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name'] and opt['rank'] == 0:
            mkdir_and_rename(osp.join(opt['root_path'], 'tb_logger', opt['name']))

    # WARNING: should not use get_root_logger in the above codes, including the called functions
    # Otherwise the logger will not be properly initialized
    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{time.strftime('%Y%m%d_%H%M%S', time.localtime())}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(dict2str(opt))
    # initialize wandb and tb loggers
    tb_logger = init_tb_loggers(opt)
    result = create_train_val_dataloader(opt, logger)