# G1 SceneStore RL on standalone AlpaGym, AlpaSim, and Cosmos-RL

Handoff for Yuxiao  
Status date: 2026-08-19

## Supported scope

This branch integrates the SceneStore-backed G1 stack without copying AlpaSim into AlpaGym:

- AlpaGym owns launch topology, rollout versions, replay packing, PPO, weight publication, and checkpoints.
- AlpaSim owns RuntimeService scheduling, policy/dynamics sessions, tick joins, and metrics transport.
- humanoid-rl-joint-sim owns SceneStore resolution, MuJoCo/MJLab dynamics, the VideoMimic observation contract, GRAIL tracking, reward profiles, and termination.

The qualified deployment shape is one-GPU colocated mode with one humanoid environment. Distributed asynchronous placement and vectorized humanoid lanes remain unsupported and fail closed in host validation.

NuRec RGB is outside this RL loop. Direct VideoMimic uses the simulator state and raycast heightmap. The motion-reference path uses realized MuJoCo state and the frozen GRAIL controller's terrain observations.

## Repository layout

Use the matching thomast/humanoid-alpagym-integration branches of:

- humanoid-rl-joint-sim
- alpasim
- standalone NVlabs/alpagym

AlpaGym remains a normal standalone repository. For a local unpublished AlpaSim humanoid ABI, set ALPASIM_GRPC_ROOT to that checkout explicitly. A configured but missing path is an error.

## Canonical policy/controller boundary

The supported planner preset is g1_videomimic_planner_hq_stairs_current_policy. It emits the g1_motion_reference_29d_50hz_h70.v1 wire schema:

- one boundary pose plus 69 autoregressive VideoMimic actions at 20 ms;
- a 70-frame reference consumed by unchanged GRAIL/SONIC tracking;
- a fixed K25 execution window, so the outer planner rate is 2 Hz;
- 60 macro decisions, 1,500 controller ticks, and 30 seconds per episode.

At each real macro-boundary observation, the same current actor generates all 69 actions while a private MuJoCo state advances autoregressively. In the qualified colocated mode, rollout generation and weight publication are synchronous, and the policy captures the live model once for the complete H70 plan. The old fake-planner behavior—one current-policy action followed by a separately loaded frozen V9 completion actor—is not supported.

Replay, the AlpaSim response, the episode artifact, and Cosmos must all agree on the behavior-policy version. Session-scoped immutable model leases are reserved for a future disaggregated mode; host validation rejects that mode until its asynchronous weight synchronization is qualified end to end.

H70 provides the complete look-ahead required by GRAIL. With K25 execution, the controller can consume through reference frame 69. There is no repeated-action fill, terminal-hold padding, or completion-policy path.

## Replay and PPO

Replay stores all 69 shadow observations, raw sampled actions, applied clip(action, -8, 8) actions, and per-token old log probabilities. Their sum must equal the scalar audit log probability.

The physical plant commits a realized prefix K in [1, 25]. Replay retains:

- all K primitive rewards in temporal order;
- duration_ticks=K and a matching contiguous reward mask;
- termination/truncation facts and final realized state;
- a critic-only bootstrap value for truncation.

Termination bootstraps with zero. Truncation evaluates the critic without sampling an action, advancing planner RNG, or creating another reference.

For an early terminal prefix, the causal action-token count is 44 + K. A full K25 transition credits all 69 action tokens. PPO clips each token likelihood ratio independently and averages only over valid causal tokens; it does not exponentiate a summed 69-action log ratio or duplicate one macro reward into 69 transitions.

Semi-Markov GAE interprets gamma and lambda per 50 Hz controller tick. For duration d, the transition reward is sum_i gamma^i r_i, the bootstrap factor is gamma^d, and the GAE continuation factor is (gamma * lambda)^d.

The canonical configuration uses reference_route_centered.v3, gamma=0.99, lambda=0.95, action standard-deviation bounds [0.05, 0.15], and a fixed initial-policy KL reference. These are the maintained defaults, not an experimental recipe matrix.

## Checkpoint and provenance contract

Run preparation snapshots the supplied planner bundle and records its immutable identity. The raw V9 export records source-checkpoint identity, deterministic actor parity, critic initialization identity, and actor lineage.

Current-policy H70 training requires checkpointing on every applied update. Each checkpoint contains:

- Cosmos model, optimizer, scheduler, and data-position state;
- a policy-native config.json plus model.safetensors export;
- actor update lineage linked to the parent actor and source attestation.

Resume fails closed when checkpoint lineage, actor-state identity, or checkpoint step disagree. Evaluation and training must load the policy-native exported bundle rather than infer weights from an unrelated path.

## Runtime integrity checks

The stack rejects:

- missing or mixed behavior-policy versions;
- mismatched reset, environment, step, timestamp, or transition metrics;
- missing truncation bootstrap values;
- a non-K25 motion-reference control timestep;
- H70 references whose applied hash differs from the emitted reference;
- raw Wizard overrides of host-owned runtime/controller settings;
- SceneStore identity or fingerprint drift.

Padding rows retain the correct behavior version and remain masked. They are fixed-shape transport rows, not transitions.

## Local colocated run

First export a planner bundle from the raw VideoMimic V9 checkpoint:

    UV_NO_SYNC=1 .venv/bin/python \
      packages/policies/g1_videomimic_planner/scripts/export_v9_planner_checkpoint.py \
      --source-checkpoint /absolute/path/to/ppo_ft_best_v9_raw.pt \
      --output-dir /tmp/g1_videomimic_current_h70 \
      --critic-seed 0

Then launch the canonical preset:

    CUDA_VISIBLE_DEVICES=0 UV_NO_SYNC=1 .venv/bin/python -m alpagym_host.cli \
      experiment=g1_videomimic_planner_hq_stairs_current_policy \
      policy.model.path=/tmp/g1_videomimic_current_h70 \
      alpasim.repo_path=/absolute/path/to/alpasim \
      alpasim.humanoid.repo_path=/absolute/path/to/humanoid-rl-joint-sim \
      alpasim.humanoid.scene_store_path=/absolute/path/to/scene_store \
      alpasim.humanoid.grail_root_path=/absolute/path/to/GRAIL \
      run_root=/tmp/alpagym-humanoid-runs

For more than one rollout per update, change cosmos.rollout.n_generation, cosmos.train.train_batch_per_replica, and step_mini_batch together. One 30-second episode contributes 60 valid macro transitions.

## Qualification boundary

The retained tests establish the current-actor H70 plan, K25 controller receipt, token replay geometry, pure critic bootstrap, behavior-version agreement, checkpoint cadence, actor/full-model lineage, and policy-native export. They do not establish convergence, model improvement, multi-node throughput, or VLA-from-RGB training.

Assets remain external: the raw VideoMimic checkpoint, exported planner bundle, GRAIL checkout, SceneStore, and matching AlpaSim humanoid ABI are not embedded in AlpaGym.
