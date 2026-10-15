cd /global/cfs/cdirs/m5421/eku/speculative_expert_parallelism


# deepspeed --num_gpus 4 scripts/ds_moe.py
/dev/shm/spec-EP/bin/python -m deepspeed.launcher.runner --num_gpus 4 scripts/ds_moe.py