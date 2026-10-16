#!/bin/bash
cd /global/cfs/cdirs/m5421/eku/speculative_expert_parallelism

# NCCL + RDMA libraries.
module load nccl/2.29.2-cu13 # provides libnccl-net.so (aws-ofi-nccl)
export NCCL_NET="AWS Libfabric" # fail loudly instead of silently falling back to Socket
export NCCL_SOCKET_IFNAME=hsn
export NCCL_CROSS_NIC=1
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_MR_CACHE_MONITOR=userfaultfd
export NCCL_DEBUG=NET
# export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT

export LOGS="logs/$(date +%Y-%m-%d_%H-%M-%S)"
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500
python="/dev/shm/spec-EP/bin/python"

mkdir -p "$LOGS/profile"

# Broadcast the env tarball to every node, extract once per node.
sbcast -f /global/cfs/cdirs/m5421/eku/envs/spec-EP.tar /dev/shm/spec-EP.tar
srun --ntasks-per-node=1 tar -xf /dev/shm/spec-EP.tar -C /dev/shm

# Only activates conda env, already bcasted.
# Env actiavted on main and CONDA_PREFIX bcasted
source ~/conda_trick.sh

srun --ntasks-per-node=4 \
     /dev/shm/spec-EP/bin/python -u benchmark/all2all/all2all.py logs/2026-10-16_14-07-50 0