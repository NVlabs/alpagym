---
marp: true
theme: nvidia
paginate: true
html: true
---

<a class="slide-id" id="0d65c412-6686-4f70-a6d7-a42db356fced"></a>

<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><rect width='32' height='32' rx='6' fill='%230a0a0a'/><rect x='4' y='6' width='24' height='16' rx='2' fill='none' stroke='%2376b900' stroke-width='2'/><rect x='8' y='10' width='8' height='2' rx='1' fill='%2376b900'/><rect x='8' y='14' width='12' height='2' rx='1' fill='%2376b900' opacity='.5'/><rect x='8' y='18' width='6' height='2' rx='1' fill='%2376b900' opacity='.3'/><rect x='12' y='22' width='8' height='2' rx='1' fill='%2376b900'/></svg>">

<!-- _class: lead -->

# **AlpaGym**

## The rollout & training harness for Alpamayo policies

A newcomer's guided tour — broad concepts first, then the code

---

<a class="slide-id" id="81d5ee9a-f8e0-433b-9a62-777cb2ecccc0"></a>

<!-- _class: lead -->

## What you'll leave with

A mental model of how a driving **policy**, the **AlpaSim** simulator and the **Cosmos-RL** trainer fit together — and a map of exactly *where in the code* each piece lives.

Every diagram box and file name in this deck is a **clickable link** into the repo at [`github.com/NVlabs/alpagym`](https://github.com/NVlabs/alpagym) · tracks the public `main` branch.

---

<a class="slide-id" id="89d6301d-4b86-46c2-be50-4c7f3cd85b87"></a>

## Agenda

<div class="columns">
<div>

1. [**The Big Picture**](#the-big-picture) — what it is & how it fits
2. [**Run It**](#run-it) — your first rollout in minutes
3. [**The Control Plane**](#the-control-plane) — the `host` package
4. [**The RL Runtime**](#the-rl-runtime) — the Cosmos-RL engine
5. [**The Episode Loop**](#the-episode-loop) — sim ⇄ policy
6. [**Policy & Inference**](#policy-and-inference) — how the car decides

</div>
<div>

7. [**Rewards**](#rewards) — scoring an episode
8. [**Episode Transport**](#episode-transport) — disk → NCCL
9. [**The Plugin Boundary**](#the-plugin-boundary) — bring your own policy
10. [**Engineering Culture**](#engineering-culture) — how we write code
11. [**Where To Look**](#where-to-look) — the reference map

</div>
</div>

---

<a class="slide-id" id="bc60802c-3cf3-417b-b240-6788b80ac851"></a>

<!-- _class: invert -->

# The Big **Picture**

---

<a class="slide-id" id="be5ce82e-f88b-4b30-b50f-d42b29b04d04"></a>

## What is AlpaGym?

> **AlpaGym** is the **rollout and training harness** for Alpamayo autonomous-driving policies. It runs a policy inside a closed-loop simulator, scores the resulting drives, and feeds them to a reinforcement-learning trainer.

<div class="columns">
<div>

### Three moving parts

- **Policy** — the neural net that drives
- **AlpaSim** — the simulated world
- **Cosmos-RL** — the RL trainer

A fourth piece, the **host**, orchestrates them.

</div>
<div>

### Why it exists

- A *lightweight*, researcher-friendly closed-loop RL harness
- Lives at [`github.com/NVlabs/alpagym`][readme]
- Clean interfaces so researchers can swap policies, rewards and models

</div>
</div>

---

<a class="slide-id" id="2e686df7-ca2a-482f-9ab7-38490f191c95"></a>

## The mental model: closed-loop RL for driving

It's the classic **agent ⇄ environment** loop — specialised for autonomous driving. The policy proposes a trajectory; the simulator executes it and returns the next observation; episodes and rewards flow to the trainer.

<!--FIG:arch-hero-->

<span class="tag tag-blue">Tap any box</span> Host → [`cli.py`][cli] · Policy → [`policy.py`][policy] · AlpaSim/egodriver → [`driver_server.py`][driver] · Trainer → [`trainer.py`][trainer]

---

<a class="slide-id" id="da515d94-fa93-4d8f-835b-120976c4b39a"></a>

## Key terms (glossary)

<div class="columns">
<div class="card">

### Policy
The driver. [`AlpamayoPolicy`][policy] maps an observation (cameras, ego history, route) to a future **trajectory**.

### AlpaSim
The closed-loop **simulator** (a separate repo). AlpaGym drives it over gRPC via the **egodriver**.

### Cosmos-RL
NVIDIA's RL training framework (GRPO). Supplies the launcher, trainer and rollout abstractions AlpaGym plugs into.

</div>
<div class="card">

### Episode / rollout
One simulated drive of a **scene** → an executed trajectory, metrics and a scalar **reward**.

### Egodriver
The gRPC **server** AlpaGym runs so the simulator can ask *"what should the car do next?"*.

### Wizard
AlpaSim's own launcher. The host starts it to bring the simulator up.

</div>
</div>

---

<a class="slide-id" id="e7500430-c254-41e7-b199-344a04d37c1e"></a>

## Repository layout

Five packages under [`packages/`](https://github.com/NVlabs/alpagym/tree/main/packages). **host** (control plane) and **runtime** (the harness) are the core; **plugins** + **policies** form the extensibility layer — policies plug in via entry points — and **alpasim_configs** ships simulator presets.

<!--FIG:packages-->

---

<a class="slide-id" id="004c6c25-d432-4652-8bc8-bc997f81fd97"></a>

## The layered architecture

Inside `runtime`, the code is organized as **layers with one job each** — and only `cosmos/` is Cosmos-RL-aware, so everything below it is reusable. *Full write-up: the [runtime README][runtimereadme].*

<!--FIG:layers-crosscutting-->

---

<a class="slide-id" id="80fc7a2e-af70-4e70-83cd-6485e1b16925"></a>

## How a run flows, end to end

<div class="columns">
<div>

### On the host (control plane)
1. Compose config with Hydra → freeze it
2. Resolve / cache the **AlpaSim** checkout
3. Start **AlpaSim's Wizard** launcher (brings the simulator up) & publish its **gRPC endpoint** — the channel AlpaGym opens to drive AlpaSim (the `simulate()` RPC)
4. Launch the **Cosmos-RL** runtime

</div>
<div>

### In the runtime (GPU)
5. Rollout backend pulls scenes, runs **episodes** against AlpaSim
6. Each tick: AlpaSim pushes camera/ego/route observations to AlpaGym's **egodriver** gRPC server → it steps the **policy** → returns the chosen **trajectory** (rig→world); AlpaSim advances the ego car and asks again
7. Score episodes → write artifacts
8. Trainer consumes them (RL)

</div>
</div>

> Local runs do steps 1–8 on one machine; Slurm runs prepare 1–4 on the login node and re-run on the cluster. See [**Run It**](#run-it).

---

<a class="slide-id" id="a265f34d-e0f6-4cdb-865d-ed24636fa9b7"></a>

<!-- _class: invert -->

# Run **It**

---

<a class="slide-id" id="1a9800ce-8e4d-4f81-af3c-aff3f8950955"></a>

## Your first rollout: a local run

Run the whole machine on one box. The simplest entry is an **`experiment`** preset — it bundles a `deploy` target, a `topology` and a policy. The public `alpamayo_1_5` smoke preset brings up AlpaSim via Wizard + Docker Compose and runs the Cosmos-RL runtime inline:

```bash
# from the repo root, after: uv sync --all-packages
uv run --no-sync --all-packages python -m alpagym_host.cli \
  experiment=alpamayo_1_5_local_2gpu_smoke \
  policy.model.path=$(pwd)/tmp/checkpoints/alpamayo-1.5-10B_alpagym_ckpt \
  reward=progress_safety
```

This runs one episode against AlpaSim, computes a [reward](#rewards), takes one **GRPO** step and writes artifacts.

> **Why start here?** It exercises every interface in the [episode loop](#the-episode-loop) and [RL runtime](#the-rl-runtime). `deploy=local` clones the public AlpaSim repo; config defaults live in [`conf/default.yaml`][defaults], and the `alpamayo_1_5` presets ship with the [`alpamayo_r1`](#policy-and-inference) policy package.

---

<a class="slide-id" id="2f11a74a-bf36-454e-8f44-21496e67d891"></a>

## Two ways to run: local vs Slurm

<div class="columns">
<div class="card">

### `deploy=local`
- Runs the host **and** the Cosmos-RL runtime inline on your machine
- AlpaSim comes up via **Docker Compose**
- Great for development & debugging
- Pair with a local topology — `topology=local_colocated_1gpu`

</div>
<div class="card">

### `deploy=slurm`
- Host **prepares** the run on the login node, then submits an `sbatch` job
- Runtime executes inside the pinned **container** on a GPU node
- The cluster node re-loads `resolved_config.yaml` and runs the same lifecycle
- Pair with a Slurm topology — `topology=slurm_full_node_1_3_4`

</div>
</div>

The **`topology`** preset selects the backend (`local_process` vs `slurm`); the **`deploy`** preset fills the site settings (partition, account, image, caches) and, for `deploy=slurm`, switches the command to **submit** — see [`config.py`][config]. Slurm submission & image caching live in [`slurm.py`][slurm].

---

<a class="slide-id" id="6c2cffd0-1316-4ea8-8430-874449159e61"></a>

## Running on the cluster

Override the experiment's `deploy`/`topology` to submit a Slurm job. The container mounts shared storage, so the checkpoint path can be passed directly:

```bash
uv run --no-sync --all-packages python -m alpagym_host.cli \
  experiment=alpamayo_1_5_local_2gpu_smoke \
  deploy=slurm \
  topology=slurm_full_node_1_3_4 \
  policy.model.path=/path/to/alpamayo-1.5-10B_alpagym_ckpt
```

> AlpaGym passes only `policy.model.path`; the [`alpamayo_r1` policy package](#the-plugin-boundary) owns its model code and resolves the trajectory tokenizer through its bundle. Full recipes (local install, HF auth, image cache) are in the [README][readme].

---

<a class="slide-id" id="610bb8a2-3209-4700-8e83-fff0b8d4bfcc"></a>

<!-- _class: invert -->

# The Control **Plane**

<p><code>packages/host</code> — orchestrates everything, runs on the login node</p>

---

<a class="slide-id" id="70d49554-4933-4c7f-8219-f6c93ea8943e"></a>

## The CLI & run lifecycle

One Hydra entrypoint, [`cli.py`][cli]: `deploy=local` executes inline, `deploy=slurm` submits an `sbatch` job. The core lifecycle lives in [`execute_run()`][execute_run] (in `run_lifecycle.py`):

<div class="columns">
<div>

1. Resolve & cache the AlpaSim checkout — [`resolve_alpasim_checkout()`][dep]
2. Start the Wizard subprocess — [`start_wizard()`][wizard]
3. Wait for its gRPC endpoint — [`wait_for_runtime_ready()`][wizard]

</div>
<div>

4. Publish the endpoint to the [topology registry][topo] so the runtime can find it
5. **Launch the [Cosmos-RL runtime](#the-rl-runtime)** — [`_build_cosmos_launcher_command()`](https://github.com/NVlabs/alpagym/blob/main/packages/host/src/alpagym_host/run_lifecycle.py) runs `cosmos_rl.launcher.launch_all` (controller + policy + rollout replicas); on Slurm it's wrapped in `srun`
6. On exit, tear the Wizard down

</div>
</div>

> Everything downstream is driven by one **frozen config** the host writes before launching — never by ad-hoc flags. See next slide.

---

<a class="slide-id" id="e1a89ae3-3856-4751-82bd-acfbba595de0"></a>

## Configuration: one source of truth

The host composes YAML defaults and CLI overrides into a typed [`RunConfig`][config], then writes per-run artifacts via [`write_run_artifacts()`][writeart].

<!--FIG:config-compose-->

---

<a class="slide-id" id="34dc403b-5e99-4f9b-bfca-8fffd40d0324"></a>

## Bringing up the simulator

<div class="columns">
<div>

### Wizard lifecycle — [`alpasim_wizard.py`][wizard]
- `start_wizard()` spawns AlpaSim in the resolved checkout
- `wait_for_runtime_ready()` polls for the generated `runtime-server.yaml` **and** TCP connectivity
- A `finally` block guarantees the process is terminated

</div>
<div>

### Service discovery — [`endpoint_registry.py`][topo]
- A tiny **file-backed registry** (`FileTopologyRegistry`): endpoints written as YAML
- `publish_alpasim_runtime()` advertises `host:port`
- The runtime later `acquire_alpasim_runtime()`s it to open its gRPC channel
- No service mesh — just files in the run dir

</div>
</div>

> The AlpaSim checkout itself is resolved & cached (keyed by `repo_url@ref`) by [`alpasim_dependency.py`][dep], which also compiles protos and `uv sync`s the simulator.

---

<a class="slide-id" id="f8b44b57-e1e1-4b28-b2d6-2aeb307addd1"></a>

## Submitting to Slurm

[`slurm.py`][slurm] turns a prepared run into a cluster job:

<div class="columns">
<div>

### Submit — [`submit_slurm_job()`][submitjob]
- Writes an `sbatch` file and submits it
- The job body re-invokes `alpagym-host command=run` with `execution.resolved_config_path=...`
- The node reruns the **same** frozen config

</div>
<div>

### Containers — [`prepare_container_image()`][prepimg]
- Docker image URIs are converted to **`.sqsh`** via `enroot import`, cached on shared storage
- `build_cosmos_srun_command()` / `build_wizard_srun_command()` add `--container-image`, mounts & env
- Credentials validated before submit

</div>
</div>

---

<a class="slide-id" id="31fea2a9-493b-4d6c-ac98-934fd5836ca2"></a>

## What a run leaves behind

[`run_artifacts.py`][writeart] builds a timestamped run directory and writes the hand-off files:

| Artifact | Purpose |
|----------|---------|
| `resolved_config.yaml` | The frozen run — re-loaded verbatim by a Slurm node |
| `cosmos_config.toml` | Launcher config for Cosmos-RL (mode, replicas, GRPO) |
| `topology/`, `logs/`, `alpasim/` | Endpoint registry, logs, simulator output |

> One run = one self-contained directory. That's what makes a Slurm job reproducible from the login node.

---

<a class="slide-id" id="6feed53a-0d68-4358-b48a-230fc9d49dc9"></a>

<!-- _class: invert -->

# The RL **Runtime**

<p><code>packages/runtime</code> — the harness, inside the Cosmos-RL container</p>

---

<a class="slide-id" id="afd42948-1605-40ab-a79a-8f5814f242e2"></a>

## What is Cosmos-RL?

[**Cosmos-RL**](https://github.com/nvidia-cosmos/cosmos-rl) is NVIDIA's open-source, distributed **RL & post-training framework for Physical AI** — LLMs, world-foundation models and **vision-language-action** policies. AlpaGym is built *on top of* it.

<div class="columns">
<div>

### Decoupled, single-controller design
- A **controller** coordinates everything
- **Policy replicas** train; **rollout replicas** generate experience — scaled independently
- Replicas are wired by dynamic **NCCL** process groups; **Redis** is the control bus
- PyTorch-native · TP / FSDP / CP / PP parallelism · FP8

</div>
<div>

### Algorithms & extensibility
- SFT + RL: **GRPO**, DAPO (LLM); FlowGRPO, DiffusionNFT (world models); VLA GRPO/SFT
- A **registry / plugin** model: swap in your own model, rollout backend, trainer, reward fns
- [github.com/nvidia-cosmos/cosmos-rl](https://github.com/nvidia-cosmos/cosmos-rl) · [docs][cosmosdocs]

</div>
</div>

> AlpaGym registers its driving-specific pieces and hands control to the launcher — the host starts it with `cosmos_rl.launcher.launch_all` (see [the control plane](#the-control-plane)).

---

<a class="slide-id" id="d799f86b-4944-42f7-a2f1-4ff24673c35b"></a>

## Cosmos-RL architecture

The [controller][cosmosctl] owns the dataset and orchestrates **policy** (trainer) and **rollout** (actor) replicas over a **Redis** bus; trained weights sync policy → rollout over **NCCL**.

<!--FIG:cosmos-arch-->

---

<a class="slide-id" id="0356e794-710f-41c5-9472-a24f677823b1"></a>

## How the Cosmos-RL loop runs

<div class="columns">
<div>

1. **Launch** — [`launch_all`][cosmoslaunch] reads `cosmos_config.toml`, places replicas, starts the controller (which spawns **Redis**)
2. **Register** — each replica runs the entrypoint, firing the `@…Registry.register` decorators, then calls `worker_entry.main(...)`
3. **Generate** — the controller dispatches prompts; rollout replicas produce trajectories and **rewards**

</div>
<div>

4. **Optimize** — batches go to policy replicas; the **GRPO** trainer takes a gradient step
5. **Weight sync** — updated weights broadcast policy → rollout over NCCL, so the next round is on-policy
6. **Repeat** until `max_num_steps`

</div>
</div>

> **GRPO** (Group Relative Policy Optimization) is a critic-free PPO variant: sample a *group* of trajectories per prompt and normalize their rewards into advantages — no value network. AlpaGym runs **colocated** (policy + rollout share GPUs).

---

<a class="slide-id" id="4cfeec69-1339-4877-a901-ab4f1622bf70"></a>

## Plugging into Cosmos-RL

Two ways to plug in: **decorator registries** (keyed by a string referenced in `cosmos_config.toml`) and **callbacks** handed to `worker_entry.main(...)`. AlpaGym wires all of it in [`cosmos/entrypoint.py`][entrypoint] — importing its modules fires the decorators, then it calls the launcher.

<div class="columns">
<div>

### Registered by string key
- `@RolloutRegistry.register("alpagym_rollout")` — [`rollout_backend.py`][rollout]
- `@TrainerRegistry.register("alpagym_trainer")` — [`trainer.py`][trainer]

</div>
<div>

### Passed to `worker_entry.main`
- **Dataset** of scenes — [`dataset.py`][dataset]
- **Data packer** — [`packer.py`][packer]
- **Reward fns** — [`reward_fn.py`][rewardfn]

</div>
</div>

> The actual driving model does **not** live in Cosmos's model slot — it runs inside the rollout backend's [inference engine](#policy-and-inference), loaded by the selected [policy bundle](#the-plugin-boundary).

---

<a class="slide-id" id="f4ac192b-fc5d-4a99-9371-9f14e67bb824"></a>

## The RL data flow

Scenes become prompts, the **actor** drives them into episodes, episodes are scored and persisted, and the learner consumes them.

<!--FIG:rl-dataflow-->

---

<a class="slide-id" id="55ab5d8c-230d-463e-900d-40bc190ec001"></a>

## The actor: rollout backend

[`AlpagymRollout`][rollout] is Cosmos's "actor". Its key method, [`rollout_generation()`](https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/cosmos/rollout_backend.py):

<div class="columns">
<div>

### Setup (once per worker)
- `init_engine()` builds the batched **inference engine**, the **policy factory**, and the **egodriver** gRPC server
- Reads the AlpaSim endpoint from the topology registry

</div>
<div>

### Per `rollout_generation`
- Maps Cosmos payloads → **scene IDs**
- `submit_payload(...)` each to the [streaming worker][runbatch], then awaits their `Future`s
- Wraps the returned `.pt` paths into Cosmos `RolloutResult`s

</div>
</div>

> This is where the simulator, the policy and the GPU all meet — unpacked in [The Episode Loop](#the-episode-loop).

---

<a class="slide-id" id="b88cb15a-2cce-4449-b2ae-5bdbdb0a7f96"></a>

## Inside one node: compute & rollout flow

<!--FIG:rollout-topology-->

---

<a class="slide-id" id="7b8921e8-50eb-4eeb-934b-4facd76ba423"></a>

## The trainer: GRPO <span class="tag tag-green">on main</span>

[`AlpagymGRPOTrainer`][trainer] subclasses Cosmos-RL's `GRPOTrainer` and runs a **real** optimization step — [`step_training()`][trainer] filters rollouts, builds minibatches and takes a gradient step. *(The earlier `AlpagymFakeTrainer` no-op stub is gone.)*

<div class="columns">
<div>

### Replay training, not re-sampling
- Rollout workers record a durable **replay contract** ([`replay.py`][replay]) — the *selected action* plus a model-family payload — into each artifact
- The trainer unpacks it via the [data packer][packer] and **recomputes logprobs** for that exact recorded action
- Applies the **GRPO / PPO surrogate** (`_compute_ppo_surrogate`) over those logprobs + group-relative advantages, with ratio clipping; optional **KL penalty** vs a refreshed reference model (`kl_beta`)

</div>
<div>

### Wired into Cosmos
- Updated policy weights **sync** policy → rollout (NCCL), so the next round is on-policy
- GRPO knobs live under `cosmos.train.train_policy` in [`conf/default.yaml`][defaults] — `grpo_ratio_clip_*`, `kl_beta`, `grpo_optimization_iterations`

</div>
</div>

---

<a class="slide-id" id="568902ab-9616-4821-bb01-2ba44300fd59"></a>

<!-- _class: invert -->

# The Episode **Loop**

<p>Where the simulator and the policy talk, step by step</p>

---

<a class="slide-id" id="491c8939-e5e0-40de-b62a-76184209b281"></a>

## What is AlpaSim?

**AlpaSim** is the **closed-loop driving simulator** AlpaGym drives — its own repo, [`NVlabs/alpasim`](https://github.com/NVlabs/alpasim). AlpaGym treats it as the *environment*: it never imports it, it talks to it over **gRPC**.

<div class="columns">
<div>

### What it provides
- Re-simulates a recorded **scene**: renders the **camera** observations and the ego state each step
- **Advances the ego car** along the trajectory the policy returns, then asks again
- Computes episode **metrics**
- Exposes a gRPC `RuntimeService` (the `simulate()` RPC) and calls back into our **egodriver**

</div>
<div>

### How AlpaGym uses it
- The host launches it through **Wizard** (AlpaSim's own launcher) — see [the control plane](#the-control-plane)
- The runtime opens the published gRPC endpoint and runs one **session** per scene × generation
- Conversion lives in [`proto_conversion.py`][proto]; the server is [`driver_server.py`][driver]

</div>
</div>

> **Closed-loop vs open-loop:** open-loop just replays ground-truth and scores predictions; **closed-loop** lets the policy *actually steer*, so mistakes compound over the episode — the realistic setting for evaluation and RL.

---

<a class="slide-id" id="3b4ecbbf-8f57-4159-8900-5320ae2538b6"></a>

## Inside AlpaSim — the *communication* view

A central **Runtime** (the *client*) calls each **gRPC microservice** in turn (①–⑤) per tick, logs the world state, and repeats. <span class="tag tag-orange">AlpaGym</span> plugs in as the **Driver ③**.

<!--FIGSM:alpasim-workflow-->

<div style="position:absolute;left:4.5rem;right:9rem;bottom:1.05rem;font-size:0.72rem;color:#666;line-height:1.4">Source: public repo <a href="https://github.com/NVlabs/alpasim">NVlabs/alpasim</a> · <a href="https://github.com/NVlabs/alpasim/blob/main/docs/DESIGN.md">DESIGN.md</a> — tap any box to open it on GitHub.</div>

---

<a class="slide-id" id="2b9d330c-5905-451a-91b8-6a876204a704"></a>

## Inside AlpaSim — the *dependency* view

The same services as **per-tick data flow** — *who feeds whom*. A clockwise cycle: world state → sensors & traffic → **Driver** → controller → physics → **Runtime**; **Eval** taps the logs, off-loop.

<!--FIGSM:alpasim-dataflow-->

<div style="position:absolute;left:4.5rem;right:9rem;bottom:1.05rem;font-size:0.72rem;color:#666;line-height:1.4">Source: public repo <a href="https://github.com/NVlabs/alpasim">NVlabs/alpasim</a> — mirrors the "Dependencies" panel of <a href="https://github.com/NVlabs/alpasim/blob/main/docs/DESIGN.md">DESIGN.md</a>.</div>

---

<a class="slide-id" id="6fe26d12-94a9-4548-ba4e-bf98a35e5c49"></a>

## Dispatching rollouts — the streaming worker

On `main`, [`StreamingRolloutWorker`][runbatch] runs **one `simulate(1)` per rollout**, not a batch:

1. `submit_payload(payload)` — **idempotent** by `prompt_idx`; creates a per-payload `Future` and enqueues `rollouts_per_payload` sibling jobs
2. A `PriorityQueue` feeds **N `alpagym-sim-{i}` threads** (N = the AlpaSim runtime's capacity)
3. Each thread fires `simulate(1)` (pinning a `session_uuid`), drains the `SessionRecord`, computes [reward](#rewards), writes a `.pt` artifact
4. **Retries preempt** fresh work (priority 0); after `n_target` siblings land, the `Future` resolves

> Cosmos **prefetches** (`enqueue_prefetch_payloads`) so the next batch's sims start while the current one is still in flight. A batch only exists at the `rollout_generation` boundary — under the hood, every rollout is an independent queue-driven job.

---

<a class="slide-id" id="f809ad36-586f-4123-9378-3a06947c5759"></a>

## The simulator ⇄ egodriver cycle

AlpaSim pushes observations and asks `drive()`; the egodriver snapshots them, steps the policy, and returns the chosen trajectory.

<!--FIG:episode-sequence-->

---

<a class="slide-id" id="ddf4c0db-3dbe-499e-abba-6ea04b40114a"></a>

## Per-session state & the tick buffer

> A **tick** is one simulation step — the simulator gathers the latest sensors, asks the policy for a control, applies it, and advances time.

<div class="columns">
<div>

### Staging a tick — [`tick_buffer.py`][tick]
AlpaSim sends cameras, egomotion and route in **separate** RPCs. The `TickBuffer` accumulates them for the current tick until `drive()` fires, then the servicer snapshots it.

### Cameras — [`cameras.py`][cameras]
A fixed name→index map puts the 7 cameras in the order the model expects, regardless of arrival order.

</div>
<div>

### Sessions & records — [`driver_server.py`][driver]
- One `_Session` per scenario: its own policy, tick buffer and lock
- `drive()` → snapshot → `policy.step()` → record I/O → `DriveResponse`
- On `close_session()` the data is frozen into an immutable **`SessionRecord`** the runner drains

</div>
</div>

---

<a class="slide-id" id="ee076ffe-3b71-4ed3-bfca-7b1049eb9db4"></a>

## The protobuf boundary

All gRPC ⇄ Python translation lives in one file, [`proto_conversion.py`][proto] — the single place where wire formats meet AlpaGym types.

<div class="columns">
<div>

### Inbound
- `policy_input_from_tick_buffer()` → [`PolicyInput`][rtypes]
- `calibration_from_proto()` (sent once at session start)
- `ground_truth_from_proto()`

</div>
<div>

### Outbound
- `drive_response_from_policy_output()` → `DriveResponse`
- `build_simulation_request_proto()` → the `simulate()` call

</div>
</div>

> Keeping conversion in one module means the rest of the runtime works with clean dataclasses, never raw protobufs.

---

<a class="slide-id" id="204fa82e-aea1-455d-852b-fd29852a066a"></a>

<!-- _class: invert -->

# Policy and **Inference**

<p>How the car turns pixels into a trajectory</p>

---

<a class="slide-id" id="8dbbc1ad-74b3-42e5-9f55-8e6fe3c12458"></a>

## One policy step

[`AlpamayoPolicy.step()`](https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/policies/alpamayo/policy.py) is four phases:

<div class="columns">
<div>

1. **Preprocess** — fold the new observation into [buffers][buffers]; build a `ModelInput` (camera frames, ego-history in **rig frame**, route)
2. **Infer** — `inference_engine.infer(...)` returns a `Future`; block on it

</div>
<div>

3. **Select** — a [selector][selectors] picks one trajectory from the sampled candidates
4. **Postprocess** — convert the chosen trajectory from **rig frame → world frame** ([`output_trajectory.py`][outtraj]) and emit a `PolicyOutput`

</div>
</div>

> The policy never touches the GPU directly — it hands work to the shared inference engine and waits.

---

<a class="slide-id" id="69be8255-d2c1-4e4d-9aef-d5497cca9212"></a>

## Inference: many sessions, one GPU batch

The [`InferenceEngine`][infeng] decouples *who asks* from *how it runs*: per-session threads enqueue requests and block on futures; a single worker thread batches them through the model.

<!--FIG:inference-batching-->

---

<a class="slide-id" id="a681b382-66cd-4070-a879-d2d879968823"></a>

## Models plug in as policies

A policy package ships an `InferenceModel` and registers a bundle under a `policy.model.kind` string, so the rollout machinery stays **model-agnostic** ([`factory.py`][factory] → [`registry.py`][registry]).

| Kind | Where | What it is |
|------|------|------------|
| `alpamayo_r1` | [`policies/alpamayo_r1`][r1] | VLM that **reasons** (chain-of-thought) before predicting a trajectory — the reference policy |
| *your policy* | external package | any policy package plugs in the same way (see [The Plugin Boundary](#the-plugin-boundary)) |
| *test double* | tests | a deterministic stand-in exercises the plumbing with no GPU or weights |

> A bundle's model code is loaded from an HF directory or tarball via [`model_bundle.py`][modelbundle]; how a kind resolves to a package is [The Plugin Boundary](#the-plugin-boundary).

---

<a class="slide-id" id="428786bc-f3ab-450c-b680-0380e5da12f3"></a>

## Buffers, selectors & coordinate frames

<div class="columns">
<div>

### Observation buffers — [`buffers.py`][buffers]
Per-session, fixed-capacity **GPU-resident** ring buffers for decoded camera frames and a monotonic ego-pose history. Frames are GPU-decoded into on-device storage (`frame_device`) and served as chronologically-ordered views.

### Selectors — [`selectors.py`][selectors]
- `IdentitySelector` — always the first candidate
- `ClosestToPreviousSelector` — the candidate closest to last step's plan, for **smooth, continuous** driving

</div>
<div>

### Rig ⇄ world frames
- Models think in the **rig frame** (ego-centric, origin at the latest pose)
- The simulator wants **world frame** (absolute)
- [`geometry_utils.py`][geom] handles quaternions, rotation matrices and interpolation; [`output_trajectory.py`][outtraj] does the final rig→world transform and prepends the current pose

</div>
</div>

---

<a class="slide-id" id="7b6d2fec-202a-4187-ab7f-cc3114e229d8"></a>

<!-- _class: invert -->

# **Rewards**

---

<a class="slide-id" id="0d4be1f8-518f-4cfa-9928-fcf712a19c9f"></a>

## Scoring an episode

[`compute_reward()`][compute] is a simple, transparent **weighted sum** of independent terms — each term reads from the episode and contributes a scaled value plus a report.

<div class="columns">
<div>

### Term kinds
- **`metric`** — pull a value straight from `episode.metrics.aggregated`
- **`distance_to_gt`** — compare the executed trajectory to the recorded ground truth

Configured in [`conf/reward/*.yaml`](https://github.com/NVlabs/alpagym/tree/main/packages/host/src/alpagym_host/conf/reward); validated by `RewardConfig`.

</div>
<div>

### distance-to-GT — [`distance_to_gt.py`][dist]
1. Align executed & GT trajectories at a common anchor time
2. Interpolate GT onto executed timestamps
3. **XY RMSE** over the overlap window → scaled reward

Useful for imitation / trajectory-tracking objectives.

</div>
</div>

> Reward is computed during the rollout and stored on the episode; Cosmos later reads `episode.reward.total` back via [`reward_fn.py`][rewardfn].

---

<a class="slide-id" id="b1935183-9365-4de9-99be-45062fe44a0f"></a>

<!-- _class: invert -->

# Episode **Transport**

<p>Getting a finished episode from the rollout worker to the trainer</p>

---

<a class="slide-id" id="eb62b7d4-d1af-4a76-953a-6b53d32627da"></a>

## The transport seam

The rollout → trainer hand-off goes through one egress protocol — `EpisodeWriter` ([`base.py`][txbase]) — with **two implementations** chosen by `transport=`: **disk** (the default, a JSON artifact) or GPU-direct **NCCL** (sender on the rollout worker, receiver on the policy worker).

<!--FIG:transport-seam-->

A **handle** is the string the writer returns and the reader resolves — for disk, just the JSON file path; for NCCL, the rendezvous manifest key.

---

<a class="slide-id" id="75fce661-f3df-4850-b05c-ffd086f10fab"></a>

## Disk — the default transport <span class="tag tag-green">default</span>

`DiskEpisodeWriter.write` ([`disk.py`][disk]) serializes the whole `EpisodeOutput` (reward + replay payload) to one **`.json`** artifact; the trainer reads it back with `read_episode_json`. This is the default hand-off (`transport=disk`) unless you opt into NCCL.

<!--FIG:disk-flow-->

---

<a class="slide-id" id="48fb5ff0-7bf8-44ab-acf9-9d3ae81ff103"></a>

## NCCL — the opt-in fast path

Set `transport=nccl` to ship episode **tensors GPU → GPU** instead of through disk, with only a tiny **manifest** on the side. Three separate channels keep the heavy data path off the slow one:

<div class="columns">
<div class="card">

### Control — `TCPStore`
The reconstruction manifest + rendezvous keys ([`comm_init.py`][txcomm], [`rendezvous.py`][txrdv]). A manifest-driven rendezvous gates every blocking transfer, so there is **one commit point and no deadlock**.

</div>
<div class="card">

### Data — `NCCL`
Tensor payloads **only**, one send per tensor on a dedicated CUDA stream ([`sender.py`][txsender] → [`receiver.py`][txreceiver]).

### Discard — `Redis`
Cosmos publishes discards; the sender frees the GPU buffer.

</div>
</div>

> Opt-in, and `cosmos.mode=disaggregated`-only (colocated always uses disk); single-host today. Full design, env handling and constraints: [NCCL&nbsp;README][nccldesign] · [`transport_env.py`][transportenv].

---

<a class="slide-id" id="c4ac9bff-1ccf-4372-8cd0-b3c8d35fe93d"></a>

<!-- _class: invert -->

# The Plugin **Boundary**

<p>How policies attach to the core — and how to add your own</p>

---

<a class="slide-id" id="9e7ce598-0a1f-4fae-b46f-62ec1cfbd941"></a>

## Policies are plugins

The core never imports a specific driving model. A policy is discovered as an **installed plugin** and selected by a `policy.model.kind` string — so you add or swap a policy without touching core code.

<div class="columns">
<div>

### The mechanism
- `policy.model.kind` is an **opaque string** — [`config.py`][config]
- Policies are discovered as **installed plugins** — [`plugins.py`][plugins]
- Each model lives in its **own package**

</div>
<div>

### Why it's shaped this way
- The core builds and runs with **no policy package** installed
- Adding a policy = ship a package + declare an entry point; no edits to `packages/*`
- Cosmos-RL registries stay at the Cosmos boundary — not the policy-selection mechanism

</div>
</div>

> One string in, one installed plugin out — the rest of the harness stays policy-agnostic.

---

<a class="slide-id" id="1a644019-99d6-46e2-bde4-31aadd190a6e"></a>

## How it works — entry-point discovery

Two **entry-point groups** keep policy names out of the core. The core discovers whatever plugins are *installed*; each policy package advertises itself from its own `pyproject.toml`.

<div class="columns">
<div>

### The two groups
- `alpagym.policy_bundles` → a [`PolicyBundle`][registry] of runtime hooks
- `alpagym.configs` → Hydra config search paths

A `PluginRegistry` ([`plugins.py`][plugins]) reads the installed table; `get_policy_bundle(kind)` loads only the one you selected.

</div>
<div>

### `alpamayo_r1` as the worked example
```toml
[project.entry-points."alpagym.policy_bundles"]
alpamayo_r1 = "alpagym_alpamayo_r1.bundle:get_bundle"
[project.entry-points."alpagym.configs"]
alpamayo_r1 = "alpagym_alpamayo_r1.configs"
```

Install the package → `policy.model.kind=alpamayo_r1` resolves. Nothing in `packages/*` imports it directly.

</div>
</div>

---

<a class="slide-id" id="b7113de2-8c82-48ae-b52b-7da813c7b558"></a>

## The PolicyBundle contract

The selected plugin returns a frozen [`PolicyBundle`][registry] of five hooks; the factory and trainer call **only the bundle** — never `import` a model package directly.

<div class="columns">
<div>

### The five hooks
- `install_runtime_bridge()` — register the model's code with the runtime
- `load_inference_model(…)` — build the GPU model
- `build_model_inputs(…)` — replay payload → model kwargs
- `build_data_packer(…)` · `setup_tokenizer(…)`

</div>
<div>

### The payoff
- **Fail fast** if the selected bundle isn't installed — [`plugins.py`][plugins]
- `check-policy-bundles` validates the wiring — no GPU, no checkpoint, no AlpaSim — [`check_policy_bundles.py`][checkbundles]
- The core builds and runs with **no policy package** installed
- Cosmos-RL registries stay at the Cosmos boundary — not the policy-selection mechanism

</div>
</div>

> Each model ships as a package-owned plugin (e.g. [`alpamayo_r1`][r1bundle]) — no per-model branch in the core.

---

<a class="slide-id" id="098750a7-78fe-4dfb-b5ef-de379052b1de"></a>

<!-- _class: invert -->

# Engineering **Culture**

---

<a class="slide-id" id="f9d33ea9-1c40-4afd-9f9c-9af0f1bb40cd"></a>

## How AlpaGym is written

The [`CONTRIBUTING.md`][contributing] guide captures the public contribution contract. Target reader: *a researcher who will modify the code, not just run it.*

<div class="columns">
<div>

### Code principles
- **Readability over flexibility** — abstractions must reduce net complexity (YAGNI)
- **Locality of behavior** — few hops; inline over gratuitous wrappers
- **One way to do it** — opinionated, no legacy paths
- **Fail fast** — raise on bad input, no silent fallbacks
- **Config in one place** — Hydra, no shadow constants

</div>
<div>

### Tests & review
- Tests prove **behavior**, not config defaults or implementation details
- Authors read every changed line before requesting review
- Contributions are signed off with the Developer Certificate of Origin
- The guide is a **living document** — update it when a rule is missing or unclear

</div>
</div>

---

<a class="slide-id" id="a2a55c82-e5a4-480b-a8cf-d509bfc3a1a1"></a>

<!-- _class: invert -->

# Where To **Look**

---

<a class="slide-id" id="465fed22-98f1-4822-95a3-5875f776fe04"></a>

## The reference map

<div class="columns">
<div>

### Control plane (`host`)
- Entry & lifecycle — [`cli.py`][cli]
- Config schema — [`config.py`][config]
- Simulator bring-up — [`alpasim_wizard.py`][wizard]
- Endpoint registry — [`endpoint_registry.py`][topo]
- Slurm & containers — [`slurm.py`][slurm]
- Run artifacts — [`run_artifacts.py`][writeart]

### RL runtime (`cosmos`)
- Registry & entry — [`entrypoint.py`][entrypoint]
- Actor — [`rollout_backend.py`][rollout]
- Trainer — [`trainer.py`][trainer]
- Replay contract — [`replay.py`][replay]
- Framework — [Cosmos-RL][cosmosdocs] · [controller][cosmosctl]
- Transport — [`base.py`][txbase] · [NCCL README][nccldesign]

</div>
<div>

### Episode & sim
- Rollout worker — [`streaming_worker.py`][runbatch]
- Egodriver server — [`driver_server.py`][driver]
- Proto boundary — [`proto_conversion.py`][proto]

### Policy & inference
- Policy step — [`policy.py`][policy]
- Inference engine — [`inference_engine.py`][infeng]
- Model bundle loader — [`model_bundle.py`][modelbundle]

### Plugins & rewards
- Plugin registry — [`plugins.py`][plugins]
- Policy bundle — [`registry.py`][registry] · [`alpamayo_r1`][r1bundle]
- Reward sum — [`compute.py`][compute]

</div>
</div>

---

<a class="slide-id" id="95cf011a-fdd0-403f-803f-fb811d639021"></a>

<!-- _class: lead -->

# Thank you

Start with the [smoke run](#run-it), then follow the [reference map](#where-to-look) into the code.

[Repo (`NVlabs/alpagym`)][readme] · [README][readme] · [Contributing][contributing] · links follow `main`

<!-- LINK REFERENCES (github.com/NVlabs/alpagym public main) -->
[readme]: https://github.com/NVlabs/alpagym/blob/main/README.md
[contributing]: https://github.com/NVlabs/alpagym/blob/main/CONTRIBUTING.md
[cli]: https://github.com/NVlabs/alpagym/blob/main/packages/host/src/alpagym_host/cli.py
[execute_run]: https://github.com/NVlabs/alpagym/blob/main/packages/host/src/alpagym_host/run_lifecycle.py
[config]: https://github.com/NVlabs/alpagym/blob/main/packages/host/src/alpagym_host/config.py
[writeart]: https://github.com/NVlabs/alpagym/blob/main/packages/host/src/alpagym_host/run_artifacts.py
[wizard]: https://github.com/NVlabs/alpagym/blob/main/packages/host/src/alpagym_host/alpasim_wizard.py
[topo]: https://github.com/NVlabs/alpagym/blob/main/packages/host/src/alpagym_host/endpoint_registry.py
[dep]: https://github.com/NVlabs/alpagym/blob/main/packages/host/src/alpagym_host/alpasim_dependency.py
[slurm]: https://github.com/NVlabs/alpagym/blob/main/packages/host/src/alpagym_host/slurm.py
[submitjob]: https://github.com/NVlabs/alpagym/blob/main/packages/host/src/alpagym_host/slurm.py
[prepimg]: https://github.com/NVlabs/alpagym/blob/main/packages/host/src/alpagym_host/slurm.py
[defaults]: https://github.com/NVlabs/alpagym/blob/main/packages/host/src/alpagym_host/conf/default.yaml
[entrypoint]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/cosmos/entrypoint.py
[trainer]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/cosmos/trainer.py
[rollout]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/cosmos/rollout_backend.py
[dataset]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/cosmos/dataset.py
[packer]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/cosmos/packer.py
[rewardfn]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/cosmos/reward_fn.py
[runbatch]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/episode_runner/streaming_worker.py
[driver]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/alpasim/driver_server.py
[proto]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/alpasim/proto_conversion.py
[tick]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/alpasim/tick_buffer.py
[cameras]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/alpasim/cameras.py
[infeng]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/inference/inference_engine.py
[inftypes]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/inference/types.py
[factory]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/policies/factory.py
[policy]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/policies/alpamayo/policy.py
[buffers]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/policies/alpamayo/buffers.py
[selectors]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/policies/alpamayo/selectors.py
[outtraj]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/policies/alpamayo/output_trajectory.py
[geom]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/policies/alpamayo/geometry_utils.py
[modelbundle]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/policies/model_bundle.py
[registry]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/policies/registry.py
[checkbundles]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/policies/check_policy_bundles.py
[plugins]: https://github.com/NVlabs/alpagym/blob/main/packages/plugins/src/alpagym_plugins/plugins.py
[r1]: https://github.com/NVlabs/alpagym/blob/main/packages/policies/alpamayo_r1/src/alpagym_alpamayo_r1/inference_model.py
[r1bundle]: https://github.com/NVlabs/alpagym/blob/main/packages/policies/alpamayo_r1/src/alpagym_alpamayo_r1/bundle.py
[compute]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/rewards/compute.py
[dist]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/rewards/distance_to_gt.py
[rtypes]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/types.py
[replay]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/replay.py
[disk]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/transport/disk.py
[txbase]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/transport/base.py
[txsender]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/transport/nccl/sender.py
[txreceiver]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/transport/nccl/receiver.py
[txrdv]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/transport/nccl/rendezvous.py
[txcomm]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/transport/nccl/comm_init.py
[transportenv]: https://github.com/NVlabs/alpagym/blob/main/packages/host/src/alpagym_host/transport_env.py
[nccldesign]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/src/alpagym_runtime/transport/nccl/README.md
[runtimereadme]: https://github.com/NVlabs/alpagym/blob/main/packages/runtime/README.md
[cosmosdocs]: https://nvidia-cosmos.github.io/cosmos-rl/
[cosmosctl]: https://github.com/nvidia-cosmos/cosmos-rl/blob/main/cosmos_rl/dispatcher/controller.py
[cosmoslaunch]: https://github.com/nvidia-cosmos/cosmos-rl/blob/main/cosmos_rl/launcher/launch_all.py
