# Asynchronous Humanoid PPO on IAD

This document records the integration work required to run the G1 humanoid
AlpaGym pipeline as asynchronous PPO on one IAD node. It compares the result
with Thomas's `thomast/humanoid-alpagym-integration` branch and records the
verified behavior of the `codex/thomas-r6-iad-e2e` branch.

## Table of Contents

- [Scope](#scope)
- [Runtime Architecture](#runtime-architecture)
- [Integration Issues and Fixes](#integration-issues-and-fixes)
  - [Cluster Topology and Process Placement](#cluster-topology-and-process-placement)
  - [Container Runtime Environment](#container-runtime-environment)
  - [Asynchronous Rollout Lifecycle](#asynchronous-rollout-lifecycle)
  - [Dataset and Transport Correctness](#dataset-and-transport-correctness)
  - [Model and Replay Parity](#model-and-replay-parity)
  - [PPO Diagnostics](#ppo-diagnostics)
- [Thomas Branch Updates](#thomas-branch-updates)
- [End-to-End Verification](#end-to-end-verification)
- [Known Limitations](#known-limitations)
- [Code Provenance](#code-provenance)

## Scope

Thomas's branch provided the pinned AlpaSim/AlpaGym integration, G1 VLA model
loading, humanoid runtime configuration, and PPO trainer path. Its original
workflow was designed around local or staged execution. The work described
here extends that path to a disaggregated asynchronous topology on IAD:

- dedicated learner GPUs;
- one rollout-policy process per rollout GPU;
- one AlpaSim runtime colocated with each rollout-policy process;
- NCCL rollout transfer to the learners;
- exact learner replay of the rollout policy density.

The issues below are therefore a mixture of missing cluster/async capabilities
and numerical bugs exposed only after rollout and learner execution were moved
into separate processes. They do not imply that Thomas's original pinned local
workflow was broken.

## Runtime Architecture

The verified one-node topology uses all eight GPUs:

| GPUs | Role | Processes |
| --- | --- | --- |
| 0-1 | Learning | Two Cosmos policy/learner replicas |
| 2-7 | Rollout cells | Six rollout-policy workers, each colocated with one AlpaSim runtime |

Each rollout worker has an explicit GPU assignment and preferred AlpaSim
runtime ID. AlpaSim and its rollout policy share a GPU, while optimization is
isolated on the learner GPUs. Cosmos schedules rollouts asynchronously and
transfers the resulting PPO payloads to the learner.

The branch also includes a seven-rollout/one-learner configuration for cases
where learner memory and throughput permit a single training GPU.

## Integration Issues and Fixes

### Cluster Topology and Process Placement

The existing topology model could describe whole Cosmos and AlpaSim hosts, but
not multiple GPU-scoped workers colocated on one Slurm node. This made a strict
learner-plus-rollout-cell layout ambiguous.

The branch adds:

- explicit `6+2` and `7+1` single-node topology configurations;
- a GPU assignment and global index for every Cosmos worker;
- one preferred AlpaSim runtime ID for every rollout worker;
- validation that worker indices are consecutive and every GPU belongs to the
  declared host GPU pool;
- validation that learner and rollout replica counts exactly consume the
  intended GPUs;
- Slurm launch scripts that set `CUDA_VISIBLE_DEVICES` separately for every
  worker process.

AlpaSim endpoint acquisition now accepts the preferred runtime ID published for
the rollout cell. A rollout worker therefore cannot silently attach to another
cell's simulator.

### Container Runtime Environment

Pyxis and login-shell initialization can replace `HOME` and `PATH` after Slurm
exports them. This caused runtime state to be written to unintended locations
and could hide the Slurm client binaries inside the container.

The Slurm launcher now reapplies the configured `HOME` and `PATH` inside the
container shell. IAD runs use Lustre-backed runtime homes and caches. The launch
path reuses the pinned SquashFS and does not compile or install dependencies at
startup.

### Asynchronous Rollout Lifecycle

Cosmos did not recognize `alpagym_rollout` as an async-capable backend, and the
blocking AlpaSim episode call could block its asynchronous scheduler.

The runtime now:

- registers `alpagym_rollout` with the Cosmos async scheduler;
- requires one payload per async generation call;
- runs the blocking simulator operation through `asyncio.to_thread`;
- preserves the payload's weight version for staleness accounting;
- treats repeated prompt-drained notifications as idempotent so a worker can
  remain alive until the controller sends its stop command;
- installs the fresh colocated model bridge when training does not resume from
  a checkpoint.

### Dataset and Transport Correctness

Scene repetitions were not consistently represented in both the Cosmos
dataset and the trainer's expected rollout count. The runtime now expands every
scene repetition into an independent prompt work item and applies the same
factor to trainer accounting.

The NCCL serializer also rejected all zero-element tensors because PyNCCL
cannot transmit an empty buffer. Empty tensors are valid in nested runtime
payloads, so the serializer now records an empty-tensor manifest containing
shape, dtype, and device. The receiver reconstructs the tensor without an NCCL
buffer transfer.

### Model and Replay Parity

PPO requires the learner to recover the rollout policy density for the exact
sampled action. Four cross-process differences had to be removed:

1. **Critic initialization:** the value head depended on ambient process RNG.
   Its initialization now uses an isolated deterministic seed.
2. **Master parameter dtype:** the rollout loader materialized parameters in
   BF16 while the learner retained FP32 master parameters. The rollout model is
   now materialized in FP32 and uses BF16 autocast only for forward execution.
3. **RoPE buffers:** generic meta-device construction could turn non-persistent
   rotary buffers into meta tensors. The loader now uses
   `init_on_device("meta", include_buffers=False)` and restores the Qwen rotary
   buffers before inference.
4. **BF16 GEMM reduction:** rollout and learner processes used different BF16
   reduced-precision accumulation settings. Both paths now disable BF16
   reduced-precision reduction.

An offline replay of a captured production payload isolated the fourth item:
changing only `allow_bf16_reduced_precision_reduction` reproduced the learner
log probabilities exactly. The real async run subsequently reported unit
pre-update ratios with zero replay error.

### PPO Diagnostics

Behavior-KL rejection messages previously omitted the information needed to
distinguish payload corruption from model or numerical drift. Diagnostics now
report:

- ratio p01, p50, and p99;
- maximum absolute log-ratio;
- maximum absolute ratio error;
- valid actor and value rows;
- pre-update and post-update approximate KL;
- whether an optimizer step and actor backtracking were applied.

Diagnostic payload dumping was used to find the reduction mismatch, then
removed from the production branch.

## Thomas Branch Updates

The branch includes Thomas's current commit `2ac1c83`, which makes Phase-A
behavior KL telemetry-only. That change merged without conflict. The strict
post-update PPO acceptance checks and async replay diagnostics remain active.

## End-to-End Verification

The strict IAD run used Slurm allocation `6592663` with two learner replicas
and six rollout cells. All six rollouts reached the learner and one optimizer
step completed.

Key results:

| Metric | Result |
| --- | ---: |
| Accepted rollouts | 6 |
| Optimizer steps applied | 1 |
| Pre-update ratio minimum | 1.0 |
| Pre-update ratio maximum | 1.0 |
| Pre-update maximum ratio error | 0.0 |
| Pre-update approximate KL | 0.0 |
| Post-update approximate KL | 0.0001500 |
| Post-update maximum ratio error | 0.02430 |
| Post-update value maximum delta | 0.3941 |
| Maximum rollout weight staleness | 0 |

The combined post-merge regression selection passed 307 tests. Earlier focused
selections passed 193 host/topology tests and 26 policy/runtime tests. Python
compilation and `git diff --check` also passed. The cached runtime image does
not include the `pre-commit` executable, so the repository-level pre-commit
command was not run.

## Known Limitations

- The strict E2E stopped after one optimizer step. It verifies async collection,
  NCCL transfer, exact pre-update replay, and parameter updates, but not repeated
  post-update weight synchronization over a sustained run.
- Cosmos left the two policy wrapper processes alive after the one-step
  controller had shut down. The completed training result was unaffected, but
  the launcher shutdown lifecycle still requires a fix.
- Weights & Biases was disabled for the strict diagnostic run.

A multi-update smoke should be the next validation. Its acceptance criteria are
continued unit pre-update ratios for fresh-version rollouts, bounded configured
staleness for in-flight rollouts, repeated optimizer steps, and clean automatic
shutdown.

## Code Provenance

- Thomas branch: `thomast/humanoid-alpagym-integration`
- Thomas commit included by this document: `2ac1c83`
- Development branch: `codex/thomas-r6-iad-e2e`
- Async implementation commits: `3b670ec`, `2738639`
- Merge of Thomas's latest branch: `8c8c769`
- Verified run directory:
  `/lustre/fsw/portfolios/av/users/yuxiaoc/alpagym_rl_assets/r6/runs/g1-r6-async-6p2-e2e-6592663/20260825T201622Z-031e0f7764ca4f61a81c104ac6ec1123`
