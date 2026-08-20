# G1 visual-policy RL on AlpaGym, AlpaSim, and Cosmos-RL

Status date: 2026-08-20

## Ownership

- AlpaGym owns Flow-PPO training, replay, behavior-policy versions, weight
  publication, and launch topology.
- AlpaSim owns rollout scheduling, same-shot camera transport, H50
  motion-reference dispatch, dynamics feedback, and metrics transport.
- humanoid-rl-joint-sim owns SceneStore resolution, NuRec camera rendering,
  MuJoCo dynamics, the GRAIL/SONIC controller backend, rewards, and termination.

The currently qualified development topology is one GPU, colocated, and one
humanoid environment. Distributed asynchronous placement is not yet qualified.

## Policy/controller boundary

`g1_wenhao_vla` consumes one receipt-checked D455 frame and the image-paired
robot state. The policy samples a one-second `30 x 38` Flow-SDE action chunk at
30 Hz. A deterministic adapter resamples it into
`g1_motion_reference_29d_50hz_h50.v1`:

- frames 0..49 are policy targets at 50 Hz;
- frame 0 remains a policy target and is not overwritten by live joint state;
- at source cursor 25 (0.5 seconds), AlpaSim launches one asynchronous request
  for the next policy sample;
- the active H50 is not truncated at that trigger: dynamics may continue it
  across requests;
- after the H50 is exhausted, the backend repeats its terminal pose with zero
  joint velocity until a new plan arrives;
- a new H50 replaces the active buffer atomically, with no crossfade.
- if a late replacement is already at or beyond cursor 25, it executes one
  source tick before the next policy snapshot; continuing predecessor-only
  feedback is rejected.

The fixed 25 ticks are therefore a nominal replan/transport interval, not a
fake-plan horizon and not a controller-buffer lifetime. There is no shadow
rollout, completion policy, native chunk RPC, LowCmd transport, or
AlpaGym-owned SONIC process supervisor.

The motion-reference runner keeps exactly one policy request in flight. It
freezes the request's image/state observation, advances the already-active H50
one SONIC tick per zero-update dynamics call, and installs the returned H50 once
at the source-time cursor implied by the delay. Repeated calls to the policy are
not used as a polling mechanism: each call remains one model sample and one
replay identity. This is simulation-time asynchronous overlap, not wall-clock
pacing; realized overlap ticks reflect the relative policy/dynamics service
throughput.

## Flow PPO

The raw stochastic action is always the `30 x 38` Flow sample. H50 is a derived
controller representation, never a second policy action. Replay stores the
selected denoise transition, element log-probabilities, rollout value,
D455/BATS condition, raw action, and realized controller feedback. All 1,140
element log-probabilities are summed into one joint chunk density before PPO
forms its ratio.

RTC hard-prefix sampling and replay are implemented: fixed prefix elements
remain unchanged and contribute zero Gaussian density. The policy derives its
delay history from realized mixed-reference feedback, records the exact prefix
and mask, and replay checks the source/applied reference receipts and action
indices before training.

The Wenhao configuration uses asymmetric clipping 0.2/0.28, dual clip 3.0,
value-loss coefficient 1.0, value clip 0.2, Huber delta 10.0, normalized
advantages, gamma 0.99, lambda 0.95, and no KL penalty. The trainer is
`alpagym_flow_ppo`; Gaussian PPO and GRPO are not substitutes. The action head
and critic use separate AdamW learning rates of `5e-6` and `1e-4`, with epsilon
`1e-8`, betas `(0.9, 0.999)`, weight decay `0.01`, and gradient clipping at
`1.0`.

Primitive rewards from one realized controller interval are deliberately
summed without intra-interval discount. The nominal duration is 25 ticks;
variable-duration bootstrap discount is `gamma ** (duration_ticks / 25)`,
while GAE lambda is applied once per sampled policy decision. This is the
chosen undiscounted macro-sum objective, not standard discounted intra-option
SMDP accumulation; changing it requires separate qualification.

If termination or the outer horizon occurs while a replan is still in flight,
AlpaSim waits for that single RPC before FINALIZE so its replay row is not lost.
A predecessor-only interval marks the newly sampled action `actor_valid=false`:
its reward and boundary value remain in SMDP GAE/value training, but PPO actor,
KL, ratio, and clip diagnostics exclude it because the new reference never
reached the controller.

## Integrity checks

The stack rejects mismatched behavior-policy versions, SceneStore fingerprints,
camera render receipts, observation decision IDs, reference identities,
non-contiguous 20 ms controller ticks, and malformed feedback. The camera used
by the policy must be a zero-shutter frame rendered from the same qpos as its
observation.

Formal execution accepts only H50. Historical H70 helpers may still exist in
standalone VideoMimic tooling, but they are not selectable by the AlpaGym /
AlpaSim Wenhao path.

## Local run

Use the `g1_wenhao_vla_hq_stairs_local_1gpu` experiment and provide:

- the Wenhao model root;
- the AlpaSim and humanoid repository paths;
- SceneStore and GRAIL paths.

The lockfile installs the matching humanoid camera ABI from the exact AlpaSim
revision. Set `ALPASIM_GRPC_ROOT` only while developing uncommitted gRPC changes
in a local AlpaSim checkout.

Assets remain external and are not embedded in AlpaGym.
