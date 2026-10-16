# function read torch files and gather them. 


# load the tokens from 1 of the layer (load *.pt)
# call all2all uneven varying
# call all2all even varying (flatten it)
# call all2all even regualr

import os
import sys
import torch
import torch.distributed as dist

run_dir = sys.argv[1]
layer = int(sys.argv[2])

E, H = 256, 2048
rank = int(os.environ["SLURM_PROCID"])
world = int(os.environ["SLURM_NTASKS"])
local_rank = int(os.environ["SLURM_LOCALID"])

torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
dist.init_process_group("nccl", rank=rank, world_size=world, device_id=device)

# Load this rank's expert assignments: [tokens, top_k].
ids = torch.load(f"{run_dir}/experts_rank{rank}.pt", map_location="cpu")[layer]
tokens, top_k = ids.shape

# Count rows sent between every pair of ranks.
send = torch.bincount(
    ids.long().flatten().to(device) // (E // world),
    minlength=world,
)
counts = [torch.empty_like(send) for _ in range(world)]
dist.all_gather(counts, send)
counts = torch.stack(counts).cpu()

send_splits = counts[rank].tolist()
recv_splits = counts[:, rank].tolist()


def benchmark(name, send_splits, recv_splits):
    x = torch.randn(sum(send_splits), H, device=device, dtype=torch.bfloat16)
    y = torch.empty(sum(recv_splits), H, device=device, dtype=x.dtype)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # warm up
    dist.all_to_all_single(y, x, recv_splits, send_splits)
    
    dist.barrier()
    torch.cuda.synchronize()
    start.record()
    dist.all_to_all_single(y, x, recv_splits, send_splits)
    end.record()
    end.synchronize()

    ms = torch.tensor(start.elapsed_time(end), device=device)
    dist.all_reduce(ms, op=dist.ReduceOp.MAX)

    if rank == 0:
        print(f"{name}: {ms.item():.3f} ms")



benchmark("uneven", send_splits, recv_splits)

balanced = [tokens * top_k // world] * world
benchmark("even", balanced, balanced)

dist.destroy_process_group()