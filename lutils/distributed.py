import os
from argparse import Namespace as ArgsNamespace

import torch
_initialized = False

def setup_torch_distributed(rank: int, args: ArgsNamespace, temp_dir: str):
    global _initialized
    if _initialized:
        print(f"[Rank {rank}] setup already called, skipping")
        return
    _initialized = True
    # Init torch.distributed
    if args.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank,
                                                 world_size=args.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank,
                                                 world_size=args.num_gpus)

        # torch.cuda.set_device(rank)
        torch.cuda.set_device(torch.device(f"cuda:{rank}"))
        print(f"[Rank {rank}] using device: {torch.cuda.current_device()} ({torch.cuda.get_device_name(torch.cuda.current_device())})")
