# AlpaGym

AlpaGym is a reinforcement-learning framework for end-to-end autonomous-driving
policies. It runs a policy in closed loop inside a simulator, scores the
resulting drives, and trains on them — so the policy learns from the
consequences of its own steering rather than from logged ground truth alone.

It stands on two systems: [AlpaSim](https://github.com/NVlabs/alpasim) provides
the closed-loop simulator (the environment), and
[Cosmos-RL](https://github.com/nvidia-cosmos/cosmos-rl) provides distributed
rollout and training orchestration (the trainer). AlpaGym is the harness that
wires them to a driving policy and keeps the interfaces small enough to swap any
one piece.

AlpaGym is in early but active development. It currently supports the
[Alpamayo 1.5](https://github.com/NVlabs/alpamayo1.5) model with 10b parameters.
Current work focuses on throughput and scaling, and on supporting more models and
training algorithms.

## Table of Contents

- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [Documentation](#documentation)

## Quick Start

Full setup — host dependencies, Hugging Face and W&B auth, and downloading and
converting a model bundle — is in the [Onboarding Guide](docs/ONBOARDING.md).
Once the host tools are installed and a converted checkpoint is under
`./tmp/checkpoints/`:

```bash
# from the repository root
uv sync --all-packages

uv run --no-sync --all-packages python -m alpagym_host.cli \
  experiment=alpamayo_1_5_local_2gpu_smoke \
  policy.model.path="$(pwd)/tmp/checkpoints/alpamayo-1.5-10B_alpagym_ckpt" \
  reward=progress_safety
```

This runs one episode against AlpaSim, computes a reward, takes one training
step, and writes artifacts under `tmp/alpagym-runs/`. Hydra output is written to
`outputs/`.

> **Note:** The default 10B Alpamayo model requires two GPUs; see the [Onboarding Guide](docs/ONBOARDING.md) for getting the checkpoint.

## Configuration

See `packages/host/src/alpagym_host/conf/default.yaml` for the full set of configuration options.
For example:

- `expected_valid_steps` controls the simulation length
- `dataset` controls which scenes to simulate. This argument is passed to AlpaSim.
- `alpasim` allows you to control the simulation parameters (see [AlpaSim](https://github.com/NVlabs/alpasim) for details)
- `cosmos` controls the topology and training parameters.
- `deploy=slurm` and the `topology=slurm_*` presets are starting points for
  cluster deployments. They are not expected to run out of the box; set the
  partition, account, container image, cache paths, mounts, and AlpaSim Wizard
  deploy preset for your Slurm environment.

## Architecture

AlpaGym is split into small workspace packages — a `host` control plane and a
`runtime` harness. During a run,
Cosmos-RL asks the runtime for rollouts; the runtime drives AlpaSim episodes
over gRPC, accumulates observations, scores the completed episodes, and hands
the artifacts back through Cosmos-RL to the trainer.

### Packages

AlpaGym is split into small workspace packages, each with one job. The
workspace is designed to make adding new packages straightforward.

- **`packages/host`** — the control plane. Owns the CLI, the typed Hydra config,
  AlpaSim checkout and bring-up, Slurm submission, and the per-run artifact
  directory. On a cluster it runs on the login node.
- **`packages/runtime`** — the harness that runs inside the GPU container. Owns
  the Cosmos-RL adapters (`cosmos/`), the rollout loop (`episode_runner/`), the
  gRPC egodriver (`alpasim/`), batched GPU inference (`inference/`), the
  policies and model adapters (`policies/`), the reward terms (`rewards/`), and
  the episode transport (`transport/`).
- **`packages/policies`** — the driving policies. Currently the Alpamayo 1.5
  policy bundle, configs, tokenizer, and checkpoint conversion script.
- **`packages/alpasim_configs`** — AlpaSim topology configs used by AlpaGym.
- **`packages/plugins`** — utilities for discovering optioninal plugin packages.

### End-to-End Run Flow

A run has two halves: the host prepares and launches, the runtime drives and
trains.

On the host (control plane):

1. Compose the config with Hydra and freeze it to a per-run directory.
2. Resolve and cache the AlpaSim checkout.
3. Start AlpaSim's Wizard launcher and publish its gRPC endpoint.
4. Launch the Cosmos-RL runtime.

In the runtime (GPU):

5. The rollout backend pulls scenes and runs episodes against AlpaSim. Each
   tick, AlpaSim sends camera, ego, and route observations to the egodriver
   gRPC server; the policy steps and returns a trajectory; AlpaSim advances the
   ego car and asks again.
6. Completed episodes are scored and written as artifacts.
7. The trainer consumes the artifacts and takes a GRPO step; updated weights
   sync back to the rollout workers.

Local runs do all of this on one machine. Slurm runs prepare steps 1–4 on the
login node, then re-load the frozen config and run the same lifecycle on the
cluster.

## Documentation

- [Onboarding Guide](docs/ONBOARDING.md) — host setup, auth, getting a model, and running locally.
- [Contributing](CONTRIBUTING.md) — code style and review process.

## Project Contributors:

Contributors in each topic in alphabetical order

**Project Lead:** Yuxiao Chen, Maximilian Igl

**Core contributors:** Bertrand Douillard, Peter Karkus, Alex Nauman, Jef Packer, Thomas Tian

**Contributors:** Jesse Archer​, William Lew​, Jack Shi, Michael Watson

**Senior Mgmt:** Nat Duca, Boris Ivanovic, Marco Pavone

**Production support:** Matt Cragun, Aditya Mahajan, Selena Williams

**Acknowledgements for additional contributions:** Liang Feng, Aaron Smith
