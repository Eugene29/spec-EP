"""Shared helpers for the benchmarks."""
import os

import torch
import torch.distributed as dist


def init_dist() -> tuple[int, int, torch.device]:
    """Init NCCL from SLURM env.

    Returns:
        rank: this rank.
        world: number of ranks.
        device: this rank's GPU.
    """
    rank = int(os.environ["SLURM_PROCID"])
    world = int(os.environ["SLURM_NTASKS"])
    local_rank = int(os.environ["SLURM_LOCALID"])

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", rank=rank, world_size=world, device_id=device)
    return rank, world, device
