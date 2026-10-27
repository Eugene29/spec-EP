"""Benchmark of all2all in uneven (real workload) and even workload.

Loads the recording of [token, experts] and mimics all2all.
"""
import os
import sys
import numpy as np
import torch
import torch.distributed as dist

E, H = 256, 2048
WARMUP = 3
ITERATIONS = 5
SCALE = 1


def init_dist():
    """Init NCCL from SLURM env. Returns (rank, world, device)."""
    rank = int(os.environ["SLURM_PROCID"])
    world = int(os.environ["SLURM_NTASKS"])
    local_rank = int(os.environ["SLURM_LOCALID"])

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", rank=rank, world_size=world, device_id=device)
    return rank, world, device


# Count rows sent between every pair of ranks.
def get_send_recv_splits(ids, E, rank, world, device, scale=SCALE):
    send = torch.bincount(
        ids.long().flatten().to(device) // (E // world),
        minlength=world,
    )
    counts = [torch.empty_like(send) for _ in range(world)]
    dist.all_gather(counts, send)
    counts = torch.stack(counts).cpu() * scale

    send_splits = counts[rank].tolist()
    recv_splits = counts[:, rank].tolist()
    return send_splits, recv_splits


def benchmark(name, send_splits, recv_splits):
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", torch.cuda.current_device())

    x = torch.randn(sum(send_splits), H, device=device, dtype=torch.bfloat16)
    y = torch.empty(sum(recv_splits), H, device=device, dtype=x.dtype)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(WARMUP):
        dist.all_to_all_single(y, x, recv_splits, send_splits)

    dist.barrier()
    torch.cuda.synchronize()
    start.record()
    for _ in range(ITERATIONS):
        dist.all_to_all_single(y, x, recv_splits, send_splits)
    end.record()
    end.synchronize()

    # Bytes this rank sends to other ranks (the self-copy never hits the network).
    row_bytes = H * x.element_size()
    sent = (sum(send_splits) - send_splits[rank]) * row_bytes
    # Assume every rank sends the same amount as rank 0.
    total = sent * world
    ms = torch.tensor(start.elapsed_time(end) / ITERATIONS, device=device)
    dist.all_reduce(ms, op=dist.ReduceOp.MAX)

    if rank == 0:
        seconds = ms.item() / 1000
        agg_gbs = total / seconds / 1e9
        print(
            f"{name}: {ms.item():.3f} ms | "
            f"{total / 2**20:.1f} MiB off-rank | "
            f"{agg_gbs:.2f} GB/s aggregate | "
            f"{agg_gbs / world:.2f} GB/s per GPU"
        )


if __name__ == "__main__":
    run_dir = sys.argv[1]
    layer = int(sys.argv[2])

    rank, world, device = init_dist()

    # Load this rank's expert assignments: [tokens, top_k].
    ids = torch.load(f"{run_dir}/experts_rank{rank}.pt", map_location="cpu")[layer]
    tokens, top_k = ids.shape

    send_splits, recv_splits = get_send_recv_splits(ids, E, rank, world, device)
    benchmark("uneven", send_splits, recv_splits)

    balanced = [tokens * top_k * SCALE // world] * world
    benchmark("even", balanced, balanced)

    dist.destroy_process_group()
