# G1 SceneStore RL on standalone AlpaGym, AlpaSim, and Cosmos-RL

Handoff for Yuxiao  
Status date: 2026-08-18

## Scope and status

This is the standalone-repository version of the humanoid integration. It keeps
Yuxiao's ownership split instead of copying AlpaSim into AlpaGym:

```text
AlpaGym host and Cosmos-RL
  -> AlpaSim Wizard and RuntimeService
  -> managed MJLab/MuJoCo-Warp humanoid dynamics
  -> AlpaGym HumanoidPolicyService
  -> strict actor-critic replay
  -> alpagym_ppo update and checkpoint
```

The corresponding stack was qualified end to end in single-GPU colocated mode
in the integration workspace. This standalone port has passed its targeted
host, runtime, replay, PPO, protobuf, and G1 package tests. A fresh standalone
checkout is **not yet externally closed**, because the public AlpaSim revision
pinned by AlpaGym predates the required humanoid protobuf ABI. Until the matching
AlpaSim work is published, use the local sibling checkout and set
`ALPASIM_GRPC_ROOT` explicitly.

Distributed asynchronous mode has not been validated. In particular, do not
infer multi-node actor/learner overlap, stale-rollout behavior under load, NCCL
transport, or recovery guarantees from the colocated qualification. Host
validation deliberately rejects `cosmos.mode=disaggregated` for humanoid runs
until start-version reporting and session-boundary weight synchronization are
ported and qualified.

## Branches

Use these branches together:

| Repository | Branch | Baseline / known SHA | Responsibility |
|---|---|---|---|
| `humanoid-rl-joint-sim` | `thomast/humanoid-alpagym-integration` | `d2e49245986f0797d254e0ba758ec01bee9ec368` | SceneStore-backed G1 task, VideoMimic V9 ABI, policy exporter, reward and metrics |
| `alpasim` | `thomast/humanoid-alpagym-integration` | `fc9b0d53679209847eff28f7df3635baa8d7b81f` | humanoid protobuf/runtime domain and managed dynamics service |
| standalone `alpagym` | `thomast/humanoid-alpagym-integration` | based on `ede34f1bcd2b2eed29af35c51ce1e5297737ff16` | host integration, policy callback, replay/PPO, G1 policy package and experiments |

The standalone AlpaGym checkout is a direct `NVlabs/alpagym` repository. It is
not the `projects/alpagym` subdirectory of another repository.

AlpaGym pins Cosmos-RL exactly at:

```text
a367b4cc814ff153e846f386fde59919ab7247e3
```

The public AlpaSim dependency remains pinned at
`10cae1c2b943da72eabafad65df80f4e3c16d90d`. This is intentional: the matching
`fc9b0d5...` branch is not currently publishable to the public repository from
this workstation. Do not replace the public source with a private URL. The
standalone external-closure blocker is resolved only when equivalent protobuf
and runtime changes are available at a public immutable revision and the pin is
updated.

## Ownership boundaries

AlpaGym owns launch topology, policy and rollout replicas, behavior-policy
versions, the policy callback server, replay packing, GAE/PPO, optimizer state,
and checkpoints.

AlpaSim owns RuntimeService scheduling, humanoid policy/dynamics session
lifecycle, action/state tick joins, terminal facts, and transport of route and
episode metrics.

`humanoid-rl-joint-sim` owns SceneStore resolution and fingerprint validation,
the MJLab/MuJoCo-Warp task, physics/observation/action timing, reward profile,
and termination logic.

Scene generation and NuRec rendering remain outside the current direct-V9 RL
episode. The training profile consumes ground-truth MJLab state and its
analytical/raycast heightmap; it does not train from NuRec RGB yet.

## Contracts that fail closed

The standalone runtime validates the generated protobuf descriptors before a
humanoid run. The selected AlpaSim ABI must carry:

- scene, scenario, attempt, reset, observation/action schema, and joint-order
  identity;
- bootstrap-only final-state value queries;
- reward, terminated, truncated, final-state, and episode-step facts;
- the exact behavior-policy version in policy responses and rollout returns.

An explicitly configured but missing `ALPASIM_GRPC_ROOT` is an error. An unset
variable still permits the existing AV path to use the installed public
package.

For humanoid PPO, the weight version is frozen when the payload is submitted,
reserved by session UUID, checked against the AlpaSim return, copied into every
transition and padding row, and compared against the Cosmos rollout version by
the trainer. Missing or mixed versions are rejected; they are never replaced
with zero.

Replay joins are lane-local and require reset ID, env ID, step index, timestamp,
value, aligned dense transition metrics, the terminal boundary, and a final
value for truncation. The final value returned by the policy RPC must match both
the dense and aggregate AlpaSim final-bootstrap metrics. There are no
zero/default reward or bootstrap fallbacks.

The current standalone delivery is deliberately `num_envs=1`. It does not yet
claim the multi-lane `TrajectoryStepId`/reset-identity design needed to prove
correct vectorized replay across asynchronous lane resets. Keep `num_envs: 1`
until that contract is implemented and qualified; config construction rejects
any other value.

Each G1 policy step resolves the model from the shared inference engine instead
of capturing the initialization-time model object. This is required because
Cosmos can replace the rollout model during colocated policy-to-rollout weight
publication.

## Direct VideoMimic V9 policy contract

| Field | Contract |
|---|---|
| Observation | 499 float32 values: 375 torso/action history, 2 target XY, 1 target yaw, 121 terrain heights |
| Heightmap | 11 x 11 independent downward rays, 0.1 m spacing, torso-yaw frame, transposed flatten order |
| Action | 23 Gaussian G1 joint offsets in VideoMimic V9 wire order |
| Actuation | clip raw action to +/-8, then `target = default_q + 0.25 * action` |
| Physics | MuJoCo-Warp `implicitfast`, 0.00125 s step (800 Hz) |
| Policy | decimation 16, 0.02 s step (50 Hz) |
| Replay | observation, executed action, old log probability/value, transition facts, bootstrap value and behavior version |

The direct policy is for stack qualification. A future VLA should emit a
versioned motion reference consumed by the SONIC/GRAIL tracker rather than
inheriting the private 23-D joint-offset interface.

## Experiments

Three standalone presets are included:

- `g1_mjlab_hq_stairs_local_1gpu_smoke`: 250 policy ticks;
- `g1_mjlab_hq_stairs_full_episode`: 750 ticks / 15 seconds;
- `g1_mjlab_hq_stairs_route_center`: full episode with the route-center reward
  and a 64-transition PPO minibatch.

The centerline reward reads the selected SceneStore scenario route; it is not
hard-coded to this staircase. The route-center thresholds are 0.10 m dead band,
0.30 m maximum distance for progress credit, and 0.45 m corridor half-width.

Keep Cosmos rollout-group geometry separate from PPO transition geometry:

```yaml
cosmos:
  train:
    train_batch_per_replica: 1
    train_policy:
      mini_batch: 1
      step_mini_batch: 64
```

A 750-row episode then produces 12 optimizer minibatches (`11 x 64 + 46`).
Padding keeps the same behavior version and remains present in fixed distributed
row geometry.

## Local colocated run

Use explicit sibling checkouts and assets:

```bash
export HUMANOID_REPO=/absolute/path/to/humanoid-rl-joint-sim
export ALPASIM_REPO=/absolute/path/to/alpasim
export ALPAGYM_REPO=/absolute/path/to/alpagym
export TWIN_SCENE_STORE_ROOT=/absolute/path/to/scene_store
export ALPASIM_GRPC_ROOT="$ALPASIM_REPO/src/grpc"
```

Export the raw actor-critic checkpoint, not the actor-only deployment file:

```bash
cd "$HUMANOID_REPO"
python -m integrations.alpasim.export_policy_bundle \
  --checkpoint checkpoints/ppo_ft_best_v9_raw.pt \
  --out /tmp/alpagym_g1_bundle
```

Build the managed dynamics image:

```bash
cd "$ALPASIM_REPO"
docker build -f Dockerfile.humanoid -t alpasim-humanoid:local .
```

Run from a tmux window:

```bash
cd "$ALPAGYM_REPO"
uv sync --frozen --all-packages

CUDA_VISIBLE_DEVICES=0 UV_NO_SYNC=1 .venv/bin/python -m alpagym_host.cli \
  experiment=g1_mjlab_hq_stairs_route_center \
  policy.model.path=/tmp/alpagym_g1_bundle \
  alpasim.repo_path="$ALPASIM_REPO" \
  alpasim.humanoid.repo_path="$HUMANOID_REPO" \
  alpasim.humanoid.scene_store_path="$TWIN_SCENE_STORE_ROOT" \
  cosmos.train.max_num_steps=2 \
  cosmos.train.num_epochs=2 \
  run_root=/tmp/alpagym-humanoid-route-center-runs
```

This is the supported local verification shape. The only available GPU is
shared by colocated Cosmos and the managed dynamics process. Distributed mode
needs separate learner and rollout/dynamics capacity and remains unverified.

## What the qualification does and does not prove

The previous colocated qualification established that real Cosmos launch,
AlpaSim rollout, replay, gradients, weight publication, and checkpoint export
can complete. It did not establish convergence: returns did not show an upward
trend after one update. Treat the route reward as correctly wired, not as a
finished learning recipe.

Before a long run:

1. add a fixed-seed evaluation lane that never updates weights;
2. collect multiple independent episodes per behavior version;
3. add an initial-policy KL or action-deviation anchor;
4. compare success, fall rate, route progress, center error, and return over
   multiple seeds;
5. publish an immutable humanoid AlpaSim revision/image and then qualify the
   distributed async topology.

Assets remain external inputs: the raw checkpoint, exported policy bundle, and
SceneStore are not embedded in standalone AlpaGym.
