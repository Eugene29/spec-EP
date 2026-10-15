import json
import os
import torch
import deepspeed
from transformers import AutoTokenizer, Qwen3_5MoeForCausalLM
from pathlib import Path
import torch.distributed as dist

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])


def count_params(m):
    total = sum(p.numel() for p in m.parameters())
    expert = sum(p.numel() for n, p in m.named_parameters() if "experts" in n)
    return total, expert

def load_model(model_id):
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model, loading_info = Qwen3_5MoeForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map=None,             # Load on CPU; DeepSpeed handles GPU placement
        attn_implementation="sdpa",
        output_loading_info=True,
    )

    total, expert = count_params(model)
    print(f"Full model: {total/1e9:.2f}B params ({expert/1e9:.2f}B in experts, "
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
    print("Applying autoEP...")
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
                         f"But recevied {tokens.input_ids[:, strt:end].shape[1]}")

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

def main():
    model_id = "/pscratch/sd/e/eku/spec_ep/models/Qwen3.6-35B-A3B"
    seq_len = 512
    
    model = load_model(model_id)
    engine = init_deepspeed(model)
    batch = get_batch(model_id, engine.device, seq_len)

    with torch.inference_mode():
        outputs = engine(**batch, use_cache=False)
    print(f"outputs: {outputs}", flush=True)
    torch.cuda.synchronize(engine.device)

    shutdown_comm()


if __name__ == "__main__":
    main()