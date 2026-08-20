# G1 Wenhao VLA Flow-PPO

This package integrates Wenhao's pinned Qwen3-VL 2B + Psi0 VLA with the
AlpaGym humanoid rollout path. The VLA is the policy: from same-shot D455 RGB,
language, 29-joint proprioception, and deterministic BATS history it samples a
native `30 x 38` float32 action chunk at 30 Hz. A pure adapter converts that
one-second chunk into the one-second H50 buffer consumed by the existing
GRAIL/SONIC controller and MuJoCo dynamics.

There is no fake planner, shadow simulator, policy-specific controller, or
H70 execution format in the formal path.

## Fixed contract

| Concern | Contract |
| --- | --- |
| Trainer | Cosmos `trainer_type: alpagym_flow_ppo` (RLinf Flow-PPO, not GRPO) |
| Model | Content-addressed `qwen3vl-wenhao2b-s1-step3203` Psi0 checkpoint |
| Trainable parameters | Psi action head and float32 critic |
| Frozen parameters | Qwen3-VL 2B `vlm_model` |
| Optimizers | Separate AdamW groups: action `5e-6`, critic `1e-4`; eps `1e-8`, weight decay `0.01`, betas `(0.9,0.999)`, grad clip `1.0` |
| Observation | Same-shot D455 RGB (`224 x 140`), 29 policy joints, deterministic BATS history, language instruction |
| Policy action | Raw `30 x 38` float32 chunk at 30 Hz (one second) |
| Controller buffer | `g1_motion_reference_29d_50hz_h50.v1`: 50 targets at 50 Hz (one second) |
| Replan trigger | After 15 source rows / 25 SONIC ticks / 0.5 seconds |
| Low-level controller | Existing GRAIL/SONIC path; MuJoCo physics at 500 Hz |
| Replacement | Atomic, `crossfade_frames=0` |
| Underflow | Repeat the terminal pose with zero joint velocity until a new buffer arrives |
| Flow sampler | Ten-step Flow-SDE with `noise_level=0.4`, `ignore_last=true` |

The raw `30 x 38` tensor is the stochastic action stored and rescored by PPO.
H50 is only its deterministic controller representation. The 25-tick value is
a replan/transport interval; it is not the lifetime of the active buffer.

## Exact 30 Hz to H50 mapping

The converter decodes all 30 source rows and samples 50 SONIC targets on a
50 Hz grid. For H50 target `n`, the source phase is `3*n/5`, clamped to source
row 29. This is equivalent to the terminal-clamped NumPy interpolation in the
Wenhao evaluator.

- H50 frame 0 is the first policy target. It is not replaced by the measured
  MuJoCo joint pose or pelvis orientation.
- H50 frames 0..49 are the converted policy targets.
- Joint velocity at a plan boundary uses the target immediately preceding the
  actual H30 source start row. At nominal H50 cursor 25 this is row 14; a late
  cursor uses a later predecessor, and an exhausted chunk uses terminal row 29.
  Later velocities use adjacent H50 targets at 20 ms.
- Wenhao has no root-Z action. The adapter preserves the realized root position
  rather than inventing a root trajectory.
- If no replacement has arrived after frame 49, dynamics continues to append
  frame 49 with zero joint velocity. A later H50 replaces the active buffer
  atomically.

## Runtime dataflow

1. AlpaGym freezes scene, checkpoint, statistics, camera, instruction, and
   behavior-version identities.
2. `HumanoidPolicyService` supplies one same-shot D455 image and its render
   qpos.
3. The policy applies checkpoint-compatible D455/BATS preprocessing. The first
   sample is unconditioned; later samples hard-inpaint a conservative prefix
   from the preceding chunk before sampling the leased Flow policy.
4. The adapter converts the native chunk to H50 and returns it through the
   standard motion-reference gRPC response.
5. AlpaSim installs that buffer atomically and SONIC consumes it at 50 Hz. A
   dynamics request may contain no plan update, in which case the existing
   buffer keeps running.
6. AlpaGym joins the realized tick rewards and controller receipts to the
   original Flow replay, then the RLinf-derived Flow-PPO trainer performs its
   clipped policy/value update.

The hard-prefix scheduler matches the native Wenhao policy clock. Nominal
inference starts at H50 cursor 25 / H30 row 15. For every later launch, the
policy derives an exact H50 cursor from the current timestamp and the prior
reference's source timestamp; both must be strictly monotonic on the 20 ms
grid. It conservatively maps that cursor to the first not-yet-started H30 row
with `ceil(cursor * 30 / 50)`. Thus a late cursor uses a shorter real suffix,
and an exhausted chunk supplies a zero-padded buffer with no frozen prefix.

The delay predictor starts at six 30 Hz rows and keeps the maximum of a
six-observation window. Realized feedback reports the initial
predecessor-reference ticks on SONIC's 50 Hz clock; the policy converts that
overlap conservatively with `ceil(ticks * 30 / 50)`. The predicted delay is
bounded by both the actual remaining suffix and the checkpoint's attested
exclusive `max_delay`. For the pinned step-3203 checkpoint, verified
`run_config.json` declares `max_delay = 8`, so only prefix lengths 0 through 7
are valid; the loaded actor and the native reference adapter must agree on that
contract before sampling. Relative pose
dimensions 29:38 are yaw-only rebased from the preceding chunk base into the
current same-shot qpos base.

AlpaSim supplies the other half of the protocol: it issues exactly one
asynchronous policy request while dynamics continues the active H50 buffer,
then returns the realized mixed-reference trace. The policy never infers delay
from host wall time and never polls by resampling. The combined path is an RTC
simulation-time execution model; production wall-clock pacing and distributed
placement remain separate qualification concerns.

When a late H50 arrives at or beyond the nominal cursor 25 trigger, AlpaSim
executes one tick from that source before taking the next policy snapshot. This
keeps every continuing replay interval causally attached to at least one
realized source tick; predecessor-only feedback is reserved for terminal or
outer-truncated actions that never reached the controller.

## Flow-PPO density contract

The sampled policy action is the complete `30 x 38` Flow chunk. Replay retains
the full latent chain, selected stochastic denoise transition, all 1,140
element log-probabilities, their scalar sum, critic value, processed visual
condition, and raw action tensor. The trainer forms one importance ratio and
one PPO clip from the joint chunk density; it does not independently clip rows
or action dimensions. Only the exponent used by the surrogate ratio is bounded
to the shared `[-5, 5]` numerical-stability contract. Post-update approximate
KL uses the raw joint log-ratio in float64 and fails closed if it becomes
non-finite, so the stability bound cannot hide policy divergence.

RTC hard-prefix replay is supported by the sampler and scorer: fixed prefix
elements remain unchanged and contribute exactly zero Gaussian density during
both rollout and replay. Replay stores the exact normalized prefix and mask.
The first plan is necessarily unconditioned; later plans derive their predictor
history from realized controller receipts rather than fabricated wall time.

The reward clock is an intentional macro-transition contract, not standard
discounted intra-option SMDP reward accumulation. Primitive 50 Hz rewards in
one realized controller interval are summed without intra-interval discount.
Only the bootstrap term uses `gamma ** (duration_ticks / 25)`, and GAE lambda
is applied once per sampled policy decision. Changing either convention would
change the training objective and requires a separately qualified experiment.

## Configuration

The [policy preset](src/alpagym_g1_wenhao_vla/configs/policy/g1_wenhao_vla.yaml)
and [single-GPU experiment](src/alpagym_g1_wenhao_vla/configs/experiment/g1_wenhao_vla_hq_stairs_local_1gpu.yaml)
select the `motion_reference` execution profile, the shared
`humanoid_reference` runtime domain, H50, and the 0.5-second replan trigger.
All workstation paths are mandatory overrides. The policy preset contains
content pins for the two dynamically loaded humanoid adapter sources; update
those pins only together with a reviewed humanoid-repository revision. The
local Wenhao experiment disables `export_safetensors` because the current
attested loader still requires the original source bundle and does not produce
a standalone Hugging Face checkpoint. This disables only the HF-style export;
when checkpointing is enabled, Cosmos still writes its model/optimizer/scheduler
resume checkpoint, including at the final step. Psi modules are compiled
directly from the verified `.py` manifest; interpreter bytecode caches are
neither read nor written by that import boundary.

```bash
uv run --no-sync --all-packages python -m alpagym_host.cli \
  experiment=g1_wenhao_vla_hq_stairs_local_1gpu \
  policy.model.path=/abs/path/to/alpa_policy_eval/models/qwen3vl-wenhao2b-s1-step3203 \
  alpasim.repo_path=/abs/path/to/alpasim \
  alpasim.humanoid.repo_path=/abs/path/to/humanoid-rl-joint-sim \
  alpasim.humanoid.scene_store_path=/abs/path/to/scene-store \
  alpasim.humanoid.grail_root_path=/abs/path/to/grail_distill \
  alpasim.humanoid.scene_cache_path=/abs/path/to/scene-cache
```

For Slurm, identity-mount the complete `alpa_policy_eval` root rather than only
the model leaf. Also identity-mount the humanoid repository, SceneStore, GRAIL
root, writable scene cache, and the selected AlpaSim checkout or checkout cache.
Host preflight rejects a Wenhao motion-reference submission when any worker
path would appear at a different absolute path inside the container. Local
execution does not require these Slurm mounts. This path audit is preparatory:
distributed humanoid placement is still unqualified, so the current host
contract continues to reject Slurm/disaggregated humanoid execution after the
mount audit succeeds.

## Verification

```bash
uv run --no-sync --all-packages pytest -q \
  packages/policies/g1_wenhao_vla/tests

uv run --no-sync --all-packages ruff check \
  packages/policies/g1_wenhao_vla

git diff --check
```

The optional real-checkpoint smoke verifies a true Flow-PPO backward pass,
separate action/critic optimizer groups, and behavior-version weight sync:

```bash
uv run --no-sync --all-packages python \
  packages/policies/g1_wenhao_vla/scripts/smoke_single_gpu_weight_sync.py \
  --model-root=/abs/path/to/alpa_policy_eval/models/qwen3vl-wenhao2b-s1-step3203
```
