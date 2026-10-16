import json
import os
import time
import torch
import deepspeed
import torch.distributed as dist

from deepspeed.profiling.flops_profiler import FlopsProfiler
from transformers import AutoTokenizer, Qwen3_5MoeForCausalLM
from datetime import datetime
from pathlib import Path

from torch.profiler import profile, ProfilerActivity, record_function

# Export env for srun launcher
if "RANK" not in os.environ:
    os.environ["RANK"] = os.environ["SLURM_PROCID"]
    os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
    os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
    
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
logs = Path(os.environ["LOGS"])
# local_rank = int(os.environ["LOCAL_RANK"])


def print0(msg):
    if rank == 0:
        print(msg, flush=True)

def count_params(m):
    total = sum(p.numel() for p in m.parameters())
    expert = sum(p.numel() for n, p in m.named_parameters() if "experts" in n)
    return total, expert

def load_model(model_id):
    print0("Loading model and tokenizer...")
    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    model, loading_info = Qwen3_5MoeForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map=None,             # Load on CPU; DeepSpeed handles GPU placement
        attn_implementation="sdpa",
        output_loading_info=True,
    )

    total, expert = count_params(model)
    print0(f"Full model: {total/1e9:.2f}B params ({expert/1e9:.2f}B in experts, "
           f"{(total-expert)/1e9:.2f}B dense)")

    assert model.config.model_type == "qwen3_5_moe_text"
    model.config.use_cache = False  # Q. What is this? 

    # Do not benchmark with accidentally uninitialized text weights.
    if loading_info["missing_keys"]:
        raise RuntimeError(
            f"Missing model weights: {loading_info['missing_keys']}"
        )
    return model
        
def init_deepspeed(model, ds_config_pth):
    with open(ds_config_pth, 'r') as file:
        ds_config = json.load(file)

    # Then apply the AutoEP config from above.
    print0("Applying autoEP...")
    engine, _, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config,
    )
    engine.eval()
    return engine
    
    
def get_batch(model_id, dev, seq_len):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    text = Path("data/short_text.txt").read_text(encoding="utf-8")
    text = " ".join(text.split())
    
    tokens = tokenizer(
        text,
        add_special_tokens=False,
        return_tensors="pt",
    )

    strt, end = rank*seq_len, (rank+1)*seq_len
    
    if tokens.input_ids[:, strt:end].shape[1] < seq_len:
        raise ValueError(f"Provide more text: at least {seq_len} tokens are required. "
                         f"[{rank}] has {tokens.input_ids[:, strt:end].shape[1]} seq.")

    batch = {
        "input_ids": tokens.input_ids[:, strt:end].to(dev),
        "attention_mask": tokens.attention_mask[:, strt:end].to(dev),
    }
    
    return batch
    
def shutdown_comm():
    # Tear down NCCL cleanly so ranks don't race on rank 0's TCPStore at exit.
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

def report_peak_mem(dev):
    peak_alloc = torch.cuda.max_memory_allocated(dev) / 2**30
    peak_resv = torch.cuda.max_memory_reserved(dev) / 2**30

    for i in range(world_size):
        dist.barrier()
        if i == rank:
            print(f"rank {rank}: peak allocated {peak_alloc:.2f} GiB, "
                  f"peak reserved {peak_resv:.2f} GiB", flush=True)

def start_flops_profile(engine):
    prof = FlopsProfiler(engine.module)
    prof.start_profile()
    return prof
    
def end_flops_profile(prof, outdir):
    prof.stop_profile()
    if rank == 0:
        prof.print_model_profile(profile_step=1, module_depth=-1, top_modules=3,
                                 detailed=True, 
                                 output_file=str(outdir / "flops_profile.txt"))
    flops = prof.get_total_flops()
    params = prof.get_total_params()
    print(f"rank {rank}: {flops/1e12:.2f} TFLOPs (fwd), "
          f"{params/1e9:.2f}B local params", flush=True)
    prof.end_profile()
    
def torch_profile_ctx():
    return profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,        # True gives source lines but makes traces much larger
        with_flops=True,
    )

def save_torch_profile(prof, outdir, verbose=False):
    trace = outdir / "profile" / f"trace_rank{rank}.json"
    prof.export_chrome_trace(str(trace))

    events = prof.key_averages()
    total_flops = sum(e.flops for e in events)   # per-rank, from the profiled step
    print(f"rank {rank}: torch-profiler counted {total_flops/1e12:.2f} TFLOPs "
          f"(aten ops only)", flush=True)
          
    if verbose and (rank == 0):
        table = events.table(sort_by="cuda_time_total", row_limit=30)
        (outdir / "profile" / "key_averages.txt").write_text(table)
        
# def get_avg_flops_per_gpu(engine, batch):
#     """Not supported after compiled"""
#     # TODO: Fix the flops looking terrible. Too bad to be true.
#     prof = FlopsProfiler(engine.module)      # run BEFORE engine.compile()
#     prof.start_profile()
#     with torch.inference_mode():
#         engine(**batch, use_cache=False)
#     prof.stop_profile()
#     f = torch.tensor([float(prof.get_total_flops())], device=engine.device)
#     prof.end_profile()
#     tot = f.clone()
#     dist.all_reduce(tot)
#     print0(f"flops/GPU avg {tot.item()/world_size/1e12:.2f} T")
#     return tot.item()
        
def main():
    model_id = "/pscratch/sd/e/eku/spec_ep/models/Qwen3.6-35B-A3B"
    seq_len = 4096
    # ds_config_pth = "configs/qwen_ep4.json"
    ds_config_pth = "configs/qwen_ep8.json"
    
    model = load_model(model_id)
    engine = init_deepspeed(model, ds_config_pth)
    batch = get_batch(model_id, engine.device, seq_len)
    engine.compile()
    

    with torch.inference_mode():
        # warmup
        engine(**batch, use_cache=False)
        engine(**batch, use_cache=False)

        # Calculate flops without profiler
        # strt = time.time()
        # torch.cuda.synchronize(engine.device)
        # engine(**batch, use_cache=False)
        # torch.cuda.synchronize(engine.device)
        # dur = time.time() - strt
        # TFlops_per_gpu = total_flop_count / dur / 1e12 / world_size
        # print(f"[{rank}] real TFlops: {TFlops_per_gpu}", flush=True)
        
        # Torch profiler pass. The FLOPs profiler hooks are left out, since they add overhead.
        with torch_profile_ctx() as tprof:
            with record_function("forward"):
                outputs = engine(**batch, use_cache=False)
            torch.cuda.synchronize(engine.device)
        save_torch_profile(tprof, outdir=logs)

        # DeepSpeed Flops profiler.
        # prof = start_flops_profile(engine)
        # outputs = engine(**batch, use_cache=False)
        # torch.cuda.synchronize(engine.device)
        # end_flops_profile(prof, outdir=logs)
        

    # bookeeping
    print(f"[{rank}] outputs: {outputs.logits.shape}", flush=True)
    report_peak_mem(engine.device)
    shutdown_comm()


if __name__ == "__main__":
    main()