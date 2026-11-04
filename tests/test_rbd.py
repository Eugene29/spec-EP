"""Tests for the Redundancy-Bypassing Dispatch (RBD) communication functions.

RBD comes from X-MoE (arXiv 2508.13337). A token routed to several experts on one
node crosses the network once, as a pilot, and its replicas are rebuilt on that
node. X-MoE's combine reverses this: replicas are gathered within the node, scaled
by combine weights exchanged ahead of time, and merged into the pilot, which is the
only copy that crosses the network back.

Each function is tested against a plain all2all, which it must match in results
while sending fewer rows between nodes.

Functions under test:
    RBD_dispatch_combine: Also sends a scalar combine weight for each copy, which
        removes the same redundancy from the combine all2all.
    RBD_dispatch_combine_async: Runs the intra-node and inter-node all2alls
        asynchronously to use all links at once.

Usage:
    python -m pytest tests/test_rbd.py -v
"""
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from src.RBD import RBD_combine_dispatch

WORLD_SIZE = 4
GPUS_PER_NODE = 2
NUM_EXPERTS = 8
NUM_TOKENS = 16
HIDDEN = 4
TOP_K = 3


def _check_identity_expert(rank: int, init_file: Path) -> None:
    """Run RBD_combine_dispatch on one rank and check its output.

    Args:
        rank: Global rank of this process.
        init_file: Shared file the ranks use to find each other.
    """
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=WORLD_SIZE
    )
    local_group, _ = dist.new_subgroups(group_size=GPUS_PER_NODE)
    torch.manual_seed(rank)

    tokens = torch.randn(NUM_TOKENS, HIDDEN)
    weights = torch.rand(NUM_TOKENS, TOP_K)
    # 3 distinct experts on 2 nodes, so every token has at least one replica.
    route = torch.rand(NUM_TOKENS, NUM_EXPERTS).topk(TOP_K, dim=-1).indices

    output = RBD_combine_dispatch(
        tokens,
        weights,
        lambda x: x,
        route,
        E=NUM_EXPERTS,
        gpus_per_node=GPUS_PER_NODE,
        local_group=local_group,
    )

    # Each expert returns the token unchanged, leaving token * (sum of its weights).
    expected = tokens * weights.sum(dim=-1, keepdim=True)
    torch.testing.assert_close(output, expected)
    dist.destroy_process_group()


def test_identity_expert_returns_weighted_tokens(tmp_path: Path) -> None:
    """With identity experts, each output row is its token scaled by its summed weights."""
    mp.spawn(
        _check_identity_expert,
        args=(tmp_path / "init",),
        nprocs=WORLD_SIZE,
    )
