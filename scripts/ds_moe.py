import json
import os
import torch
import deepspeed
import torch.distributed as dist

from deepspeed.profiling.flops_profiler import FlopsProfiler
from transformers import AutoTokenizer, Qwen3_5MoeForCausalLM
from datetime import datetime
from pathlib import Path

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
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
        
def init_deepspeed(model):
    with open("configs/qwen_ep.json", 'r') as file:
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
    
    text = Path("data/long_text.txt").read_text(encoding="utf-8")
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
    
def make_outdir(base="logs"):
    # NOTE: broadcast the log if you need to print on more than rank 0
    if rank == 0:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        outdir = Path(base) / stamp
        (outdir / "profile").mkdir(parents=True, exist_ok=True)
        return outdir
    else:
        return None
    
def main():
    model_id = "/pscratch/sd/e/eku/spec_ep/models/Qwen3.6-35B-A3B"
    seq_len = 4096
    
    model = load_model(model_id)
    engine = init_deepspeed(model)
    engine.compile()
    batch = get_batch(model_id, engine.device, seq_len)
    outdir = make_outdir()

    with torch.inference_mode():
        # warmup
        engine(**batch, use_cache=False)
        
        prof = start_flops_profile(engine)
        outputs = engine(**batch, use_cache=False)
        torch.cuda.synchronize(engine.device)
        end_flops_profile(prof, outdir)

    # bookeeping
    print(f"[{rank}] outputs: {outputs}", flush=True)
    report_peak_mem(engine.device)
    shutdown_comm()


if __name__ == "__main__":
    main()