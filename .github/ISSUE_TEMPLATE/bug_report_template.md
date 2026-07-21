---
name: Bug report
about: Create a bug report to help us improve Alpamayo
title: "[BUG]"
labels: "? - Needs Triage, bug"
assignees: 'yesfandiari'

---

**Describe the bug**
A clear and concise description of what the bug is.

**Steps/Code to reproduce bug**
Follow this guide http://matthewrocklin.com/blog/work/2018/02/28/minimal-bug-reports to craft a minimal bug report. This helps us reproduce the issue and resolve it more quickly.

**Expected behavior**
A clear and concise description of what you expected to happen.

**Environment overview (please complete the following information)**
 - Deployment: [local (`deploy=local`) or Slurm (`deploy=slurm`, `topology=slurm_*`)]
 - Install method: `uv sync --all-packages` — paste `uv --version` and Python version
 - Experiment / config: Hydra `experiment=` preset (e.g. alpamayo_1_5_local_2gpu_smoke) and `reward=` used
 - Policy / model: [e.g. Alpamayo 1.5 10B]; converted checkpoint present under `tmp/checkpoints/`? (yes/no)
 - AlpaSim + Cosmos-RL versions / commits in use

**Environment details**
 - Hardware: GPU type(s), VRAM, number of GPUs (the 10B model needs ≥2 GPUs), nodes
 - Operating System
 - CUDA / NVIDIA driver version (from `nvidia-smi`)
 - Relevant artifacts: Hydra output dir (`outputs/`) and run artifacts (`tmp/alpagym-runs/`)

**Additional context**
Add any other context about the problem here.
