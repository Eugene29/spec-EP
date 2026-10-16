import json
import os
import time
import torch
import deepspeed

from transformers import AutoTokenizer, Qwen3_5MoeForCausalLM
from pathlib import Path

from utils import print0, shutdown_comm, get_rank_and_world_size
from utils_profile import count_params, end_flops_profile, add_expert_statistics_hook, record_function, report_peak_mem, save_torch_profile, start_flops_profile, torch_profile_ctx


rank, world_size = get_rank_and_world_size()
logs = Path(os.environ["LOGS"])


def load_model(model_id):
    print0("Loading model and tokenizer...")
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

def record_expert_routing(engine, batch):
    """Save expert routing statistics (not a2a)"""
    sent = {}
    add_expert_statistics_hook(engine.module, sent)
    engine(**batch, use_cache=False)
    torch.save(torch.stack(list(sent.values())), logs / f"experts_rank{rank}.pt")

def record_latency(engine, batch):
    """Calculate latency and flops without profiler"""
    strt = time.time()
    torch.cuda.synchronize(engine.device)
    engine(**batch, use_cache=False)
    torch.cuda.synchronize(engine.device)
    dur = time.time() - strt
    print(f"dur: {dur}", flush=True)
    # TFlops_per_gpu = total_flop_count / dur / 1e12 / world_size
    # print(f"[{rank}] real TFlops: {TFlops_per_gpu}", flush=True)
    
def run_pytorch_profile(engine, batch):
    """Torch profiler pass. The FLOPs profiler hooks are left out, since they add overhead."""
    with torch_profile_ctx() as tprof:
        with record_function("forward"):
            outputs = engine(**batch, use_cache=False)
        torch.cuda.synchronize(engine.device)
    save_torch_profile(tprof, outdir=logs)

def run_ds_flops_profile(engine, batch):
    """DeepSpeed Flops profiler"""
    prof = start_flops_profile(engine)
    outputs = engine(**batch, use_cache=False)
    torch.cuda.synchronize(engine.device)
    end_flops_profile(prof, outdir=logs)

def main():
    model_id = "/pscratch/sd/e/eku/spec_ep/models/Qwen3.6-35B-A3B"
    seq_len = 4096
    # ds_config_pth = "configs/qwen_ep4.json"
    ds_config_pth = "configs/qwen_ep8.json"
    
    model = load_model(model_id)
    engine = init_deepspeed(model, ds_config_pth)
    batch = get_batch(model_id, engine.device, seq_len)

    with torch.inference_mode():
        # compile + warmup
        engine.compile()
        engine(**batch, use_cache=False)
    
        # record_expert_routing(engine, batch)
        record_latency(engine, batch)
        # run_pytorch_profile(engine, batch)
        # run_ds_flops_profile(engine, batch)

    # bookeeping
    report_peak_mem(engine.device)
    shutdown_comm()

if __name__ == "__main__":
    main()