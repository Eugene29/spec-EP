"""Shared helpers for the benchmarks."""
import os
import socket

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


def check_placement(gpus_per_node: int) -> list[str]:
    """Check ranks fill at least 2 nodes in blocks of gpus_per_node.

    Block order (ranks 0-3 on node 0, 4-7 on node 1, ...) is what makes
    rank + gpus_per_node the same local rank on a different node.

    Returns:
        hosts: hostname of every rank.
    """
    world = dist.get_world_size()
    if world % gpus_per_node != 0:
        raise RuntimeError(f"{world} ranks is not a multiple of {gpus_per_node} per node.")

    num_nodes = world // gpus_per_node
    if num_nodes < 2:
        raise RuntimeError(f"Expected at least 2 nodes, got {num_nodes}. Use srun -N 2 or more.")

    hosts = [None] * world
    dist.all_gather_object(hosts, socket.gethostname())

    # Every node should be a different host.
    node_hosts = hosts[::gpus_per_node]
    if len(set(node_hosts)) != num_nodes:
        raise RuntimeError(f"Some nodes share a host: {node_hosts}.")
    return hosts
