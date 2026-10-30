"""Benchmark of all-to-all of MoE expert weights across nodes.

Each GPU splits a buffer of N experts evenly across the other nodes and sends
one chunk to the GPU with the same local rank on each of them, receiving a
chunk from each in return. Every rank sends and receives the same N experts'
worth of bytes as the ring in expert_send_recv_parallel.py, so the two are
directly comparable. With 2 nodes the all-to-all is the same bidirectional
swap as the ring.

This measures whether internode bisection bandwidth is faster than ring.

Also, with --balance uneven, we can test whether uneven all2all can be slower 
than even all2all in internode setting. 

The expert architecture assumes a gated FFNN (e.g. SwiGLU), as in
Qwen3.6-35B-A3B: each expert holds gate and up projections [I, H] and a
down projection [H, I], so 3 * H * I parameters. Defaults are taken from
its config (hidden_size=2048, moe_intermediate_size=512), giving 6 MiB per
expert in bf16. Experts are sent as one flat buffer, split by element, so a
chunk may hold part of an expert.

Usage:
    # 2 nodes: each GPU swaps with its counterpart on the other node
    srun -N 2 --ntasks-per-node=4 python -u benchmark/internode/expert_all2all.py

    # 4 nodes: each GPU sends a third of its buffer to each of the other 3 nodes
    srun -N 4 --ntasks-per-node=4 python -u benchmark/internode/expert_all2all.py

    # 4 nodes, uneven: nodes 0-1 send 1.5N experts, nodes 2-3 send 0.5N
    srun -N 4 --ntasks-per-node=4 python -u benchmark/internode/expert_all2all.py --balance uneven
"""
import argparse
import os

import torch
import torch.distributed as dist

from benchmark.utils import check_placement, init_dist


def get_peer_group(rank: int, world: int, gpus_per_node: int) -> dist.ProcessGroup:
    """Group this rank with the same local rank on every node.

    Returns:
        group: ranks local_rank, local_rank + gpus_per_node, ...; this rank's
            index in it is its node index.
    """
    group = None
    for local_rank in range(gpus_per_node):
        ranks = list(range(local_rank, world, gpus_per_node))
        g = dist.new_group(ranks)
        if rank in ranks:
            group = g
    return group


def get_split_matrix(numel: int, num_nodes: int, balance: str) -> list[list[int]]:
    """Elements each node sends to each other node, split evenly across the other nodes.

    With even balance every node sends numel. With uneven balance the first half
    of nodes send 1.5 * numel and the second half 0.5 * numel, so the average
    stays numel. Nothing goes to a node's own node, so every byte crosses the network.

    Returns:
        splits: splits[src][dst] is elements src sends to dst.
    """
    if balance == "even":
        sends = [numel] * num_nodes
    else:
        if num_nodes % 2 != 0 or numel % 2 != 0:
            raise RuntimeError(f"Uneven balance needs an even number of nodes and elements, "
                               f"got {num_nodes} nodes and {numel} elements.")
        half = num_nodes // 2
        sends = [numel * 3 // 2] * half + [numel // 2] * half

    splits = []
    for src, send in enumerate(sends):
        if send % (num_nodes - 1) != 0:
            raise RuntimeError(f"{send} elements don't split evenly across {num_nodes - 1} nodes.")
        chunk = send // (num_nodes - 1)
        splits.append([0 if dst == src else chunk for dst in range(num_nodes)])
    return splits


def benchmark(
    n: int,
    args: argparse.Namespace,
    group: dist.ProcessGroup,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[int, list[float]]:
    """Every rank sends N experts split across the other nodes at once.

    Returns:
        nbytes: average bytes each rank sends in total.
        times: ms of each timed all-to-all, the slowest of all ranks.
    """
    # Gated FFNN: gate [I, H] + up [I, H] + down [H, I].
    numel = n * 3 * args.hidden * args.intermediate
    splits = get_split_matrix(numel, group.size(), args.balance)

    # Send this node's row, receive this node's column.
    my_node = group.rank()
    send_splits = splits[my_node]
    recv_splits = [row[my_node] for row in splits]

    send_buf = torch.randn(sum(send_splits), device=device, dtype=dtype)
    recv_buf = torch.empty(sum(recv_splits), device=device, dtype=dtype)

    for _ in range(args.warmup):
        dist.all_to_all_single(recv_buf, send_buf, recv_splits, send_splits, group=group)

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.iterations)]

    for start, end in zip(starts, ends):
        # Line all ranks up so every group's all-to-all starts together.
        dist.barrier()
        start.record()
        dist.all_to_all_single(recv_buf, send_buf, recv_splits, send_splits, group=group)
        end.record()
    torch.cuda.synchronize()

    # The all-to-all is done only once every rank finishes; take the slowest rank.
    ms = torch.tensor([s.elapsed_time(e) for s, e in zip(starts, ends)], device=device)
    dist.all_reduce(ms, op=dist.ReduceOp.MAX)
    return numel * send_buf.element_size(), ms.tolist()


def main(args: argparse.Namespace) -> None:
    """Run the all-to-all benchmark for each expert count and print results on rank 0."""
    rank, world, device = init_dist()
    gpus_per_node = world // int(os.environ["SLURM_NNODES"])
    hosts = check_placement(gpus_per_node)
    group = get_peer_group(rank, world, gpus_per_node)
    dtype = getattr(torch, args.dtype)

    if rank == 0:
        node_hosts = hosts[::gpus_per_node]
        print(
            f"all-to-all of {len(node_hosts)} nodes x {gpus_per_node} GPUs: "
            f"{', '.join(node_hosts)} | {args.dtype} | balance={args.balance}",
            flush=True,
        )

    for n in args.experts:
        nbytes, times = benchmark(n, args, group, device, dtype)
        if rank == 0:
            avg = sum(times) / len(times)
            per_rank_gbs = nbytes / (avg / 1000) / 1e9
            print(
                f"{n} experts: {nbytes / 2**20:.1f} MiB per rank | "
                f"min {min(times):.3f} / avg {avg:.3f} / max {max(times):.3f} ms | "
                f"{per_rank_gbs:.2f} GB/s per rank | "
                f"{per_rank_gbs * world:.2f} GB/s aggregate",
                flush=True,
            )
            print(f"  all ms: {', '.join(f'{t:.3f}' for t in times)}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experts", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32],
                        help="Number of experts per message")
    parser.add_argument("--hidden", type=int, default=2048, help="hidden_size")
    parser.add_argument("--intermediate", type=int, default=512,
                        help="moe_intermediate_size")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"],
                        default="bfloat16")
    parser.add_argument("--balance", choices=["even", "uneven"], default="even",
                        help="even: every node sends n experts; uneven: first half "
                             "of nodes send 1.5n, second half 0.5n")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=5)
    main(parser.parse_args())
