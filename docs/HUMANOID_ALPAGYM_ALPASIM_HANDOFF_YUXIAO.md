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
| `humanoid-rl-joint-sim` | `thomast/humanoid-alpagym-integration` | `c50a0089a55a4810f2b5a8acd10996f86a39a833` | SceneStore-backed G1 tasks, direct-V9 and motion-reference runtime ABIs, reward and metrics |
| `alpasim` | `thomast/humanoid-alpagym-integration` | `f094010bc7491d9cd65d5bec90439ed642c8914c` | humanoid protobuf/runtime domain and managed dynamics service |
| standalone `alpagym` | `thomast/humanoid-alpagym-integration` | based on `ede34f1bcd2b2eed29af35c51ce1e5297737ff16` | host integration, policy callback, replay/PPO, G1 policy package and experiments |

The standalone AlpaGym checkout is a direct `NVlabs/alpagym` repository. It is
not the `projects/alpagym` subdirectory of another repository.

AlpaGym pins Cosmos-RL exactly at:

```text
a367b4cc814ff153e846f386fde59919ab7247e3
```

The public AlpaSim dependency remains pinned at
`10cae1c2b943da72eabafad65df80f4e3c16d90d`. This is intentional: the matching
`f094010...` branch is not currently publishable to the public repository from
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
the MJLab/MuJoCo-Warp direct task, the MuJoCo reference-tracking backend,
physics/observation/action timing, reward profiles, and termination logic.

Scene generation and NuRec rendering remain outside the current humanoid RL
episodes. Direct V9 consumes ground-truth MJLab state and its raycast heightmap;
the motion-reference profile uses realized MuJoCo state and GRAIL's privileged
terrain rays. Neither profile trains from NuRec RGB yet.

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

In motion-reference mode, raw Wizard `extra_overrides` cannot replace
`runtime_domain` or any `runtime.humanoid.controller` setting. Reward identity,
thresholds, controller backend, and the derived 1,500-tick horizon remain owned
by the typed AlpaGym config and are auditable in the resolved manifest.

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

The direct policy is for stack qualification. The implemented
motion-reference profile is the intended controller boundary for a future VLA;
the VLA should not inherit the private 23-D joint-offset interface.

## VideoMimic planner with frozen GRAIL tracking

The planner profile uses the action schema
`g1_motion_reference_29d_50hz_h50.v1`. At each 10 Hz outer decision,
VideoMimic produces a complete 50-frame, 29-joint reference. The frozen GRAIL
heightmap controller executes the first five 20 ms ticks in MuJoCo, then the
planner receives the realized state and replans. H50 is a receding one-second
reference, not the episode horizon.

The canonical episode is 300 macro decisions, 1,500 controller ticks, or
30 seconds. AlpaGym owns `expected_valid_steps=300`; Wizard derives and AlpaSim
validates the matching controller limit. Reward is produced from each realized
50 Hz post-control state and the five primitive rewards are retained for
semi-Markov PPO/GAE. Cosmos' scalar episode callback is reporting/bookkeeping;
changing it to binary success does not replace the primitive PPO rewards.

The training objective therefore remains dense credited route progress plus
center/heading/tilt/safety costs and a terminal success bonus. Model selection
uses held-out fixed-seed success rate first, followed by fall/off-route rate,
final and credited progress, center error, and return. Pure success reward is
too sparse for the current stair task.

The canonical preset selects `reference_route_centered.v3`: progress and
success each have a total budget of +10, fall/off-route adds -5, and the small
clearance cost is enabled. The terminal penalty is not a safety certificate;
the held-out fall/off-route gate is what prevents an unsafe checkpoint from
being promoted.

Semi-Markov GAE interprets `gamma` and `lambda` per 50 Hz controller tick: it
discounts the five primitive rewards in order and raises both factors to the
committed tick duration at the macro boundary. The preset therefore uses the
fifth roots `gamma=0.9979919516614258` and
`lambda=0.9897937816869885`, which preserve `0.99` and `0.95` per 10 Hz planner
decision. Supplying `0.99/0.95` directly would discount both five times per
planner decision.

The canonical optimizer uses `1e-6` learning rate and `kl_beta=0.1` against a
frozen copy of the initial V9 policy. `reference_reset_interval=0` keeps that
anchor fixed for the whole run, including Cosmos checkpoint resume: the trainer
reconstructs the teacher from Cosmos's pre-resume initial-policy state instead
of cloning the restored live policy. AlpaGym rejects nonzero
`reference_reset_interval` values because a moving anchor is not part of the
standard Cosmos resume contract. The first update has zero reference KL by
construction, so the small learning rate is still required; later updates use
the KL term to prevent cumulative action-distribution drift.

## Experiments

Four standalone presets are included:

- `g1_mjlab_hq_stairs_local_1gpu_smoke`: 250 policy ticks;
- `g1_mjlab_hq_stairs_full_episode`: 750 ticks / 15 seconds;
- `g1_mjlab_hq_stairs_route_center`: full episode with the route-center reward
  and a 64-transition PPO minibatch.
- `g1_videomimic_planner_hq_stairs_local_1gpu`: 300 planner decisions /
  30 seconds with online motion references and frozen GRAIL tracking.

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

The motion-reference preset instead uses `step_mini_batch=300` for one episode.
For eight episodes per update, set `n_generation=8`,
`train_batch_per_replica=8`, and `step_mini_batch=2400`; padding rows remain
masked and do not become valid transitions.

## Local colocated run

The host must provide Docker Compose (AlpaSim Wizard), `redis-server`
(Cosmos-RL's local data plane), and `redis-cli` (clean shutdown of that local
server). They are host tools, not Python dependencies. Verify them before
launching:

```bash
docker compose version
redis-server --version
redis-cli --version
```

`alpagym_host` now fails during local-process preflight if any executable is
missing, before it starts Wizard or reserves a GPU. See `docs/ONBOARDING.md` for
the supported installation prerequisites; the launcher does not install system
packages.

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

For planner/tracker PPO, export the VideoMimic planner bundle and select the
motion-reference experiment. In addition to the paths above, provide the
hash-locked GRAIL checkout:

```bash
CUDA_VISIBLE_DEVICES=0 UV_NO_SYNC=1 .venv/bin/python -m alpagym_host.cli \
  experiment=g1_videomimic_planner_hq_stairs_local_1gpu \
  policy.model.path=/absolute/path/to/g1_videomimic_planner_bundle \
  alpasim.repo_path="$ALPASIM_REPO" \
  alpasim.humanoid.repo_path="$HUMANOID_REPO" \
  alpasim.humanoid.scene_store_path="$TWIN_SCENE_STORE_ROOT" \
  alpasim.humanoid.grail_root_path=/absolute/path/to/GRAIL \
  cosmos.rollout.n_generation=8 \
  cosmos.train.train_batch_per_replica=8 \
  cosmos.train.max_num_steps=2 \
  cosmos.train.num_epochs=2 \
  cosmos.train.train_policy.step_mini_batch=2400 \
  run_root=/tmp/alpagym-humanoid-reference-runs
```

## What the qualification does and does not prove

On 2026-08-18 the motion-reference path completed a real two-update, batch-8
Cosmos PPO run and a non-updating paired evaluation on seeds 424200--424207.
All 16 training artifacts, both step-2 checkpoint formats, and all eight
evaluation artifacts passed closure checks. The canonical v3 candidate changed
the fixed panel from initial V9's `3S/2F/3O` to `4S/2F/2O`, while mean
cross-track error changed from 0.1033 m to 0.0912 m. The candidate was not
promoted: N=8 is small, mean credited progress declined from 0.8045 to 0.7872,
and one initial success became an early fall.

The same panel established that 20 seconds is not a valid task horizon. For an
earlier checkpoint, four 20-second timeouts all became successes between 20.78
and 22.74 seconds under the 30-second horizon. Initial V9 also exposed one late
success and one late off-route result. Formal training and model selection must
therefore use 30 seconds; 20 seconds may only be labeled as a smoke test.

This proves real colocated launch, rollout, primitive-reward replay, SMDP GAE,
gradients, KL anchoring, weight publication, checkpoint export, and fixed-seed
evaluation. It does not prove convergence or a production model improvement.
Before a long run, expand the held-out seed panel and independent training runs,
then publish immutable humanoid and AlpaSim revisions/images. Distributed async
placement remains unqualified.

Assets remain external inputs: the raw checkpoint, exported policy bundle, and
SceneStore are not embedded in standalone AlpaGym.
