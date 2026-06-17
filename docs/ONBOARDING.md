# AlpaGym Onboarding

This guide walks through running AlpaGym locally on one or two GPUs: installing
host dependencies, authenticating with Hugging Face and Weights & Biases,
getting a model bundle, and launching a run. For the system overview, see the
[README](../README.md).

## Table of Contents

- [Hardware and Disk Budget](#hardware-and-disk-budget)
- [AlpaSim Setup](#alpasim-setup)
- [Prerequisites](#prerequisites)
- [Hugging Face](#hugging-face)
- [Weights and Biases](#weights-and-biases)
- [Check the Workspace](#check-the-workspace)
- [Get the Alpamayo 1.5 Model](#get-the-alpamayo-15-model)
- [Check GPU Peer-to-Peer](#check-gpu-peer-to-peer)
- [Run Locally](#run-locally)
- [Evaluate a Checkpoint in AlpaSim (optional)](#evaluate-a-checkpoint-in-alpasim-optional)
- [Cleaning Up After a Crash](#cleaning-up-after-a-crash)

## Hardware and Disk Budget

To comfortably run AlpaGym locally with the AR1.5 policy, we recommend two (with smaller
models, colocated on one GPU is also feasible)CUDA GPU with at least 40 GBof VRAM
(e.g. A6000) and ~100–150 GB of free diskfor the `uv` environment,
container images, and model weights (Alpamayo-R1-10B ~21 GB), plus ~1.5 GB for
each NuRec scene you download (the full `public_2601` NuRec suite is ~1.5 TB).

## AlpaSim Setup

First follow the
[AlpaSim Onboarding Guide](https://github.com/NVlabs/alpasim/blob/main/docs/ONBOARDING.md).
To verify the simulator setup, clone the
[AlpaSim](https://github.com/NVlabs/alpasim) repo and run "Level 1" from the
[tutorial](https://github.com/NVlabs/alpasim/blob/main/docs/TUTORIAL.md). Once
that works, AlpaGym can manage its own AlpaSim checkout.

## Prerequisites

Install the host-side tools and system dependencies:

```bash
# UV for Python environment and package management.
uv self update

# Make sure docker and docker compose are installed and working.
docker compose version

# CUDA apt repository for cuDNN packages.
wget -O /tmp/cuda-keyring_1.1-1_all.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i /tmp/cuda-keyring_1.1-1_all.deb
sudo apt-get update

# CUDA/cuDNN headers.
sudo apt-get install -y libcudnn9-dev-cuda-12

# NCCL headers.
sudo apt-get install -y libnccl-dev=2.26.2-1+cuda12.8 libnccl2=2.26.2-1+cuda12.8

# Redis executable used by Cosmos-RL.
sudo apt-get install -y redis-server
redis-server --version

# Git LFS files for AlpaSim checkouts.
sudo apt-get install -y git-lfs
git lfs install
git lfs pull
```

The runtime expects a CUDA-capable host with CUDA, cuDNN, and NCCL headers.
AlpaSim Wizard expects Docker Compose. Cosmos-RL starts its own Redis process,
so the `redis-server` executable must be installed even though Redis is not a
Python package.

Before syncing, run the pre-flight check to confirm the host tools and headers
above are installed correctly. It prints a fix command for anything missing:

```bash
install/check_env.sh
```

Then sync the Python workspace:

```bash
uv sync --all-packages
```

Note: For Slurm deployments, we split `runtime` and `host` dependencies to avoid
installing heavy ML packages on login nodes.

## Hugging Face

AlpaSim downloads scenes from the gated
`nvidia/PhysicalAI-Autonomous-Vehicles-NuRec` dataset. Access needs two steps:

1. Request access on the
   [dataset page](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec).
2. Authenticate with a token from the approved account:

```bash
hf auth login
# Or export HF_TOKEN=...
```

## Weights and Biases

If you want Weights and Biases logging, set `WANDB_API_KEY` in your shell
profile so child processes started by Cosmos-RL can read it:

```bash
export WANDB_API_KEY=...
```

Additionally, add `cosmos.logging.logger='[console,wandb]'` to your command and
optionally set `cosmos.logging.experiment_name`.

## Check the Workspace

Run the default test suite:

```bash
uv run pytest
```

Real AlpaSim and NCCL end-to-end tests require local infrastructure and are
opt-in.

## Get the Alpamayo 1.5 Model

Download the Alpamayo-1.5 10b model and convert it to the AlpaGym
checkpoint format:

```bash
uv run --no-sync python -c "\
from huggingface_hub import snapshot_download; \
print(snapshot_download('nvidia/Alpamayo-1.5-10B', \
local_dir='./tmp/checkpoints/Alpamayo-1.5-10B'))\
"

uv run --no-sync --package alpagym-runtime python \
  packages/policies/alpamayo_r1/scripts/convert_release_to_alpagym_checkpoint.py \
  --input ./tmp/checkpoints/Alpamayo-1.5-10B \
  --output ./tmp/checkpoints/alpamayo-1.5-10B_alpagym_ckpt \
  --overwrite
```

## Check GPU Peer-to-Peer

The policy and rollout run on separate GPUs and exchange weights over NCCL
point-to-point send/recv. On some hosts the direct GPU-to-GPU transport (NVLink
or PCIe P2P) is advertised by NCCL but then stalls, which appears as a hang
during the first weight sync (`[Rollout] Starting to execute ... weight sync receives ...`) with both GPUs stuck at 100%.

Confirm your topology works before launching a full run:

```bash
uv run --no-sync torchrun --nproc-per-node=2 install/check_nccl_p2p.py
```

If that hangs for ~10s and then errors, your direct P2P transport is broken.
Re-run with shared-memory transport to confirm:

```bash
NCCL_P2P_DISABLE=1 uv run --no-sync torchrun --nproc-per-node=2 install/check_nccl_p2p.py
```

If the check passes only with `NCCL_P2P_DISABLE=1`, prefix the run command below
with the same variable (`NCCL_P2P_DISABLE=1 uv run ...`). Weight sync then
stages through host memory, which is slower but reliable. This is usually caused
by PCIe ACS / IOMMU settings; `nvidia-smi topo -m` shows how your GPUs are
connected.

## Run Locally

Start a local run on two GPUs. The 10B model requires two GPUs; this command was
tested on 2x50 GB GPUs (RTX 6000 Ada):

```bash
uv run --no-sync --all-packages python -m alpagym_host.cli \
  experiment=alpamayo_1_5_local_2gpu_smoke \
  policy.model.path="$(pwd)/tmp/checkpoints/alpamayo-1.5-10B_alpagym_ckpt" \
  reward=progress_safety
```

Outputs are written under `outputs/` and run artifacts under `tmp/alpagym-runs/`.

In the future, we are planning to release a distillation script that can
convert the 10B Alpamyo model to a smaller checkpoint (e.g. 2B) that can run on a single
GPU.

## Evaluate a Checkpoint in AlpaSim (optional)

Optionally, evaluate a post-trained checkpoint in AlpaSim.

**1. Convert the export to inference format.** Each run writes a safetensors export
at its final step under `tmp/alpagym-runs/<run>/cosmos/<timestamp>/safetensors/step_<N>/`.
The AlpaSim driver can't load it directly, so convert it (the inverse of the
model-conversion step above):

```bash
uv run --no-sync --package alpagym-alpamayo-r1 python \
  packages/policies/alpamayo_r1/scripts/convert_alpagym_checkpoint_to_inference.py \
  --input tmp/alpagym-runs/<run>/cosmos/<timestamp>/safetensors/step_<N> \
  --output ./tmp/checkpoints/alpamayo-1.5-10B_clrl_inference --overwrite
```

**2. Run AlpaSim eval** from the cached AlpaSim checkout:

```bash
cd "$(ls -d ~/.cache/alpagym/alpasim/*/ | head -n1)"
uv run alpasim_wizard \
  deploy=local topology=1gpu \
  driver=alpamayo1_5 \
  defines.drivers=/ABS/PATH/TO/alpagym_public/tmp/checkpoints \
  driver.model.checkpoint_path=/mnt/drivers/alpamayo-1.5-10B_clrl_inference \
  scenes.scene_ids=[clipgt-9ea70552-6dcb-4ee8-a368-9a906a333f6e]
```

For more about AlpaSim, see the [AlpaSim repo](https://github.com/NVlabs/alpasim).

## Cleaning Up After a Crash

A local run that crashes can leave AlpaSim Wizard containers and GPU-holding
processes behind. Reclaim the GPU and remove the stale containers with:

```bash
pkill -9 -f 'cosmos.entrypoint|cosmos_rl.launcher|alpagym_host.cli|alpasim_wizard|torchrun'
docker rm -f $(docker ps -aq -f name=wizard) 2>/dev/null
```
