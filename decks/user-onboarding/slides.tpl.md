---
marp: true
theme: nvidia
paginate: true
html: true
---

<a class="slide-id" id="6799ff88-b964-441c-bf98-a7e1bb4a393e"></a>

<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><rect width='32' height='32' rx='6' fill='%230a0a0a'/><rect x='4' y='6' width='24' height='16' rx='2' fill='none' stroke='%2376b900' stroke-width='2'/><rect x='8' y='10' width='8' height='2' rx='1' fill='%2376b900'/><rect x='8' y='14' width='12' height='2' rx='1' fill='%2376b900' opacity='.5'/><rect x='8' y='18' width='6' height='2' rx='1' fill='%2376b900' opacity='.3'/><rect x='12' y='22' width='8' height='2' rx='1' fill='%2376b900'/></svg>">

<!-- _class: lead -->

# **AlpaGym**

## Train Alpamayo driving policies in closed-loop simulation

A researcher's quick start — set up, run, tune, and plug in your own policy

---

<a class="slide-id" id="e606169d-b62e-4d3b-8ee8-5fffc729d10a"></a>

## What is AlpaGym?

A **closed-loop RL gym** for Alpamayo driving policies: roll your policy out in **AlpaSim**, score it, and train it with **Cosmos-RL** (GRPO). You bring a policy + checkpoint — the host CLI runs the loop.

<!--FIGSM:alpagym-loop-->

<span class="tag tag-green">Tap any box</span> to open the file that implements it.

---

<a class="slide-id" id="210e16c9-3c21-4a4b-8f90-7282d3142677"></a>

## Setup

Three things, once, from the repo root:

<div class="columns">
<div class="card">

### 1 · Workspace

```bash
uv sync --all-packages
```

Local runs also need Docker Compose, Redis, Git LFS, and CUDA/cuDNN/NCCL headers.

</div>
<div class="card">

### 2 · Scenes

Request **NuRec** dataset access (browser), then:

```bash
hf auth login
```

</div>
<div class="card">

### 3 · Weights

Get an Alpamayo **checkpoint** bundle — you'll pass it as `policy.model.path`.

</div>
</div>

> The launcher checks all three up front and **fails fast with a link** if one is missing.

---

<a class="slide-id" id="e1210d46-eea7-4b76-a30e-91a219d19c6c"></a>

## Run it

One local training step with the `alpamayo_r1` policy:

```bash
uv run --no-sync --all-packages python -m alpagym_host.cli \
  experiment=alpamayo_1_5_local_2gpu_smoke \
  policy.model.path=$(pwd)/tmp/checkpoints/<your-ckpt> \
  reward=progress_safety
```

Each run writes a self-contained directory under `./tmp/alpagym-runs/<timestamp>/` (the CLI prints the path). A good smoke run shows `Step: 1/1` in `logs/…/controller.log`.

---

<a class="slide-id" id="b757ad28-dcbf-4324-bda1-10234d3ecdf3"></a>

## Common config changes

Everything is Hydra — override any field as `key=value` on the CLI, or pick a preset by name (the filename in [`conf/`](https://github.com/NVlabs/alpagym/tree/main/packages/host/src/alpagym_host/conf)).

| Override                              | Changes                              | Presets in        |
| ------------------------------------- | ------------------------------------ | ----------------- |
| `deploy=` `local` / `slurm`           | run inline vs. submit a Slurm job    | `conf/deploy/`    |
| `topology=` `local_*` / `slurm_*`     | how GPUs are split across train/sim  | `conf/topology/`  |
| `policy=` (+ `policy.model.kind=`)    | which policy + checkpoint            | `conf/policy/`    |
| `cosmos.rollout.*` / `cosmos.train.*` | rollouts per scene, epochs, LR, …    | — (Hydra override) |

e.g. `cosmos.rollout.n_generation=8 cosmos.train.num_epochs=2`

---

<a class="slide-id" id="8e91b902-c2ad-45f7-aa1f-d79428c32fcc"></a>

<!-- _class: invert -->

# Add your own policy

The part you came for

---

<a class="slide-id" id="40a0da29-ee0e-48df-8004-dfd6a0c941a4"></a>

## 1 · The contract

The simulator calls one small protocol every drive tick — implement [`Policy`](https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/types.py):

```python
class Policy(Protocol):
    def step(self, policy_input: PolicyInput) -> PolicyOutput: ...
    def close(self) -> None: ...
```

For the training side, provide an [`InferenceModel`](https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/inference/types.py) (forward + log-probs for GRPO).

> Copy [`AlpamayoPolicy`](https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/policies/alpamayo/policy.py) as the reference implementation.

---

<a class="slide-id" id="c585c5b9-a247-49e7-8fdd-b81fefc5560f"></a>

## 2 · Wire it in

<div class="columns">
<div>

1. Pick a `kind` string for your policy — e.g. `my_policy` (it's opaque to [`config.py`](https://github.com/NVlabs/alpagym/blob/main/packages/host/src/alpagym_host/config.py), no enum to edit).
2. Ship a package whose `get_bundle()` returns a [`PolicyBundle`](https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/policies/registry.py) and declares an `alpagym.policy_bundles` entry point named `my_policy`. Copy [`alpamayo_r1`](https://github.com/NVlabs/alpagym/blob/main/packages/policies/alpamayo_r1/src/alpagym_alpamayo_r1/bundle.py).
3. Ship its Hydra configs via the `alpagym.configs` entry point — no public code edits.

</div>
<div>

4. Install the package — `check-policy-bundles` confirms the wiring (no GPU needed).
5. Run it:

```bash
… policy=my_policy \
  policy.model.kind=my_policy \
  policy.model.path=/path/to/ckpt
```

</div>
</div>

<span class="tag tag-green">Why this shape</span> the cosmos trainer stays policy-agnostic — your bundle is imported lazily, only for the kind you run.

---

<a class="slide-id" id="33dd0cfa-5ca1-4b13-bc2f-64c7dce065ce"></a>

<!-- _class: lead -->

## That's the loop

Bring a policy → point at a checkpoint → run → read the run directory.

Go deeper: the **[dev-onboarding deck](dev-onboarding.html)** (full architecture) and the [`README.md`](https://github.com/NVlabs/alpagym/blob/main/README.md). Links point at the public `main` branch.
