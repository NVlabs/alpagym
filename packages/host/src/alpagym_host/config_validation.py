# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import math
import re
from pathlib import Path

from alpagym_host.alpasim_dependency import validate_alpasim_checkout_cache
from alpagym_host.config import (
    AllInOneSlurmTopologyConfig,
    AlpaSimConfig,
    CosmosRLConfig,
    CosmosRLMode,
    CosmosRLPolicyParallelismConfig,
    CosmosRLRolloutParallelismConfig,
    DatasetConfig,
    ExecutionBackend,
    HumanoidExecutionProfile,
    HumanoidPolicyCameraProfile,
    RunConfig,
    SeparateNodesSlurmTopologyConfig,
    SlurmConfig,
    SlurmLayout,
    TransportKind,
)
from alpagym_host.run_artifacts import is_supported_hf_bundle_dir
from alpagym_host.run_topology import build_slurm_topology
from alpagym_host.slurm import validate_slurm_config


def validate_run_config(
    config: RunConfig,
    requested_command: str,
) -> None:
    """Validate a host run config before execution or submission.

    Args:
        config: Host run configuration.
        requested_command: Host command requested by Hydra.

    Raises:
        ValueError: The config cannot be executed safely.
    """
    _validate_wizard_startup_config(
        config=config.alpasim,
        dataset=config.dataset,
    )
    # Run the VLA Slurm visibility audit before the currently intentional
    # humanoid colocated-only rejection, so an eventual qualification cannot
    # inherit latent host/container path drift.
    _validate_vla_slurm_worker_mounts(config)
    _validate_humanoid_config(config)
    _validate_training_policy_config(config)
    _validate_cosmos_grpo_batch_geometry(config.cosmos)
    _validate_transport_config(config)
    _validate_policy_model_path(config)
    _validate_cosmos_mode(config)
    execution_backend = ExecutionBackend(config.execution.backend)
    if requested_command == "submit" and not execution_backend.is_slurm_run:
        raise ValueError(
            "command=submit requires execution.backend to be a Slurm backend"
        )
    if execution_backend is ExecutionBackend.slurm:
        if (
            config.alpasim.repo_path is None
            and config.alpasim.checkout_cache_dir is None
        ):
            raise ValueError(
                "alpasim.checkout_cache_dir must be set for a Slurm run when "
                "alpasim.repo_path is not set"
            )
        # Check absolute paths first: validate_slurm_config and validate_alpasim_checkout_cache
        # below mkdir uv_cache_dir/checkout_cache_dir, and a null cache_root_dir resolves them
        # to relative "None/..." strings that would otherwise create stray dirs under the cwd
        # before this guard runs.
        for config_key, value in [
            ("execution.slurm.uv_cache_dir", config.execution.slurm.uv_cache_dir),
            ("alpasim.checkout_cache_dir", config.alpasim.checkout_cache_dir),
        ]:
            if value is not None and not Path(value).expanduser().is_absolute():
                raise ValueError(
                    f"{config_key} must be an absolute path, got {value!r} "
                    "(check that cache_root_dir is set)"
                )
        _validate_slurm_topology_config(config.execution.slurm)
        validate_slurm_config(config.execution)
        _validate_slurm_cosmos_gpu_capacity(config)
        _validate_shared_cosmos_2_3_shape(config)
        validate_alpasim_checkout_cache(config.alpasim)
        _require_host_path_identity_mounted(
            config=config,
            path=Path(config.policy.model.path),
            label="policy.model.path",
        )
        _require_host_path_identity_mounted(
            config=config,
            path=Path(config.run_root),
            label="run_root",
        )


def _validate_wizard_startup_config(
    config: AlpaSimConfig,
    dataset: DatasetConfig,
) -> None:
    """Validate host-authored AlpaSim Wizard config before startup side effects."""
    if not config.wizard_args.deploy:
        raise ValueError("config.wizard_args.deploy must be non-empty")
    if not config.wizard_args.topology:
        raise ValueError("config.wizard_args.topology must be non-empty")
    if not config.wizard_args.driver_source:
        raise ValueError("config.wizard_args.driver_source must be non-empty")
    min_force_gt_duration_us = 0 if config.simulation_domain == "humanoid" else 1
    if config.wizard_args.force_gt_duration_us < min_force_gt_duration_us:
        expectation = (
            "non-negative" if config.simulation_domain == "humanoid" else "positive"
        )
        raise ValueError(
            f"config.wizard_args.force_gt_duration_us must be {expectation}"
        )
    if config.wizard_args.driver is not None and not config.wizard_args.driver:
        raise ValueError("config.wizard_args.driver must be non-empty when set")
    if config.wizard_args.renderer is not None and not config.wizard_args.renderer:
        raise ValueError("config.wizard_args.renderer must be non-empty when set")
    selectors = [
        dataset.scene_ids is not None,
        dataset.test_suite_id is not None,
    ]
    if sum(selectors) != 1:
        raise ValueError("dataset must set exactly one of scene_ids or test_suite_id")
    if dataset.scene_ids is not None and not dataset.scene_ids:
        raise ValueError("dataset.scene_ids must be non-empty when set")
    if dataset.test_suite_id is not None and not dataset.test_suite_id:
        raise ValueError("dataset.test_suite_id must be non-empty when set")


def _validate_humanoid_config(config: RunConfig) -> None:
    """Fail closed on humanoid routing and version-unsafe prefetch."""
    if config.policy.model.kind == "g1_vla":
        if config.policy.kind != "humanoid":
            raise ValueError("g1_vla requires policy.kind=humanoid")
        if config.alpasim.simulation_domain != "humanoid":
            raise ValueError("g1_vla requires alpasim.simulation_domain=humanoid")
        humanoid = config.alpasim.humanoid
        if humanoid is None:
            raise ValueError("g1_vla requires alpasim.humanoid")
        if humanoid.execution_profile is not HumanoidExecutionProfile.motion_reference:
            raise ValueError("g1_vla requires execution_profile=motion_reference")
        bundle_config = config.policy.model.bundle_config
        if bundle_config.get("humanoid_policy_factory") != (
            "alpagym_g1_vla.humanoid_policy:build_humanoid_policy_factory"
        ):
            raise ValueError("g1_vla requires its native humanoid_policy_factory")
        if bundle_config.get("require_policy_camera") is not True:
            raise ValueError("g1_vla requires require_policy_camera=true")
    if config.alpasim.simulation_domain != "humanoid":
        return
    humanoid = config.alpasim.humanoid
    if humanoid is None:
        raise ValueError("humanoid simulation requires alpasim.humanoid")
    if config.dataset.scene_ids is None:
        raise ValueError("humanoid simulation requires explicit dataset.scene_ids")
    expected_scenes = set(config.dataset.scene_ids)
    mapped_scenes = set(humanoid.scenario_ids_by_scene)
    if mapped_scenes != expected_scenes:
        raise ValueError(
            "alpasim.humanoid.scenario_ids_by_scene must match dataset.scene_ids: "
            f"missing={sorted(expected_scenes - mapped_scenes)}, "
            f"unexpected={sorted(mapped_scenes - expected_scenes)}"
        )
    if humanoid.expected_scene_fingerprints:
        fingerprint_scenes = set(humanoid.expected_scene_fingerprints)
        if fingerprint_scenes != expected_scenes:
            raise ValueError(
                "alpasim.humanoid.expected_scene_fingerprints must match "
                f"dataset.scene_ids: missing={sorted(expected_scenes - fingerprint_scenes)}, "
                f"unexpected={sorted(fingerprint_scenes - expected_scenes)}"
            )
        if any(
            re.fullmatch(r"[0-9a-f]{64}", digest) is None
            for digest in humanoid.expected_scene_fingerprints.values()
        ):
            raise ValueError("humanoid scene fingerprints must be lowercase SHA256")
    if humanoid.execution_profile is HumanoidExecutionProfile.motion_reference:
        if not humanoid.expected_scene_fingerprints:
            raise ValueError(
                "motion_reference requires a frozen expected_scene_fingerprints map"
            )
        expected_json = json.dumps(
            humanoid.expected_scene_fingerprints,
            sort_keys=True,
            separators=(",", ":"),
        )
        if (
            config.policy.model.bundle_config.get("expected_scene_fingerprints_json")
            != expected_json
        ):
            raise ValueError(
                "motion_reference policy and AlpaSim scene fingerprint snapshots differ"
            )
        expected_bundle_paths = {
            "humanoid_repo_path": str(Path(humanoid.repo_path).expanduser().resolve()),
            "scene_store_path": str(
                Path(humanoid.scene_store_path).expanduser().resolve()
            ),
        }
        for key, expected_path in expected_bundle_paths.items():
            if config.policy.model.bundle_config.get(key) != expected_path:
                raise ValueError(
                    f"motion_reference policy bundle {key} must come from "
                    "alpasim.humanoid"
                )
        if humanoid.reference_frame_count != 50:
            raise ValueError(
                "motion_reference requires the one-second H50 wire contract"
            )
        outer_period_us = int(config.alpasim.wizard_args.control_timestep_us)
        if outer_period_us != 500_000:
            raise ValueError(
                "motion_reference requires the native 500000us replan trigger"
            )
        if config.policy.model.step_dt_us != outer_period_us:
            raise ValueError(
                "motion_reference policy.model.step_dt_us must equal the "
                "AlpaSim outer policy period"
            )
        if config.alpasim.wizard_args.force_gt_duration_us != 0:
            raise ValueError("motion_reference requires force_gt_duration_us=0")
        if config.alpasim.wizard_args.n_sim_steps != config.expected_valid_steps:
            raise ValueError(
                "motion_reference n_sim_steps must equal expected_valid_steps"
            )
        if (
            humanoid.reward_profile_id == "direct_v9_shaped.v1"
            and config.alpasim.wizard_args.n_sim_steps * outer_period_us != 15_000_000
        ):
            raise ValueError(
                "motion_reference direct_v9_shaped.v1 requires the source "
                "750-tick / 15-second horizon"
            )
        if config.policy.model.kind == "g1_vla":
            if (
                humanoid.policy_camera_profile
                is not HumanoidPolicyCameraProfile.vla_d455
            ):
                raise ValueError("g1_vla requires policy_camera_profile=vla_d455")
            if config.policy.model.use_cameras != ["vla_d455_policy_rgb"]:
                raise ValueError("g1_vla requires only vla_d455_policy_rgb")
            train_policy = config.cosmos.train.train_policy
            required_flow_values = {
                "grpo_ratio_clip_low": 0.2,
                "grpo_ratio_clip_high": 0.28,
                "ppo_value_loss_coef": 1.0,
                "ppo_value_clip_range": 0.2,
                "ppo_gamma": 0.99,
                "ppo_gae_lambda": 0.95,
                "ppo_dual_clip_ratio": 3.0,
                "ppo_value_huber_delta": 10.0,
                "kl_beta": 0.0,
            }
            actual_flow_values = {
                "grpo_ratio_clip_low": train_policy.grpo_ratio_clip_low,
                "grpo_ratio_clip_high": train_policy.grpo_ratio_clip_high,
                "ppo_value_loss_coef": train_policy.ppo_value_loss_coef,
                "ppo_value_clip_range": train_policy.ppo_value_clip_range,
                "ppo_gamma": train_policy.ppo_gamma,
                "ppo_gae_lambda": train_policy.ppo_gae_lambda,
                "ppo_dual_clip_ratio": train_policy.ppo_dual_clip_ratio,
                "ppo_value_huber_delta": train_policy.ppo_value_huber_delta,
                "kl_beta": train_policy.kl_beta,
            }
            if train_policy.trainer_type != "alpagym_flow_ppo":
                raise ValueError("g1_vla motion_reference requires alpagym_flow_ppo")
            for name, expected in required_flow_values.items():
                if actual_flow_values[name] != expected:
                    raise ValueError(f"g1_vla requires {name}={expected}")
            if train_policy.ppo_normalize_advantages is not True:
                raise ValueError("g1_vla requires ppo_normalize_advantages=true")
            required_optimizer_values = {
                "optm_part_lrs": [5.0e-6, 1.0e-4],
                "epsilon": 1.0e-8,
                "optm_weight_decay": 0.01,
                "optm_betas": [0.9, 0.999],
                "optm_grad_norm_clip": 1.0,
                "optm_warmup_steps": 0,
            }
            actual_optimizer_values = {
                "optm_part_lrs": config.cosmos.train.optm_part_lrs,
                "epsilon": config.cosmos.train.epsilon,
                "optm_weight_decay": config.cosmos.train.optm_weight_decay,
                "optm_betas": config.cosmos.train.optm_betas,
                "optm_grad_norm_clip": config.cosmos.train.optm_grad_norm_clip,
                "optm_warmup_steps": config.cosmos.train.optm_warmup_steps,
            }
            for name, expected in required_optimizer_values.items():
                if actual_optimizer_values[name] != expected:
                    raise ValueError(f"g1_vla requires {name}={expected}")
    if config.cosmos.rollout.prefetch_rollout:
        raise ValueError(
            "humanoid rollouts require prefetch_rollout=false until Cosmos passes "
            "current_weight_version to its prefetch hook"
        )
    if config.cosmos.mode is not CosmosRLMode.colocated:
        raise ValueError(
            "standalone humanoid rollouts currently support cosmos.mode=colocated only; "
            "distributed async requires start-version reporting and session-boundary "
            "weight synchronization"
        )


def _validate_cosmos_mode(config: RunConfig) -> None:
    """Validate backend-specific Cosmos placement mode requirements."""
    execution_backend = ExecutionBackend(config.execution.backend)
    if (
        execution_backend.is_slurm_run
        and config.cosmos.mode is not CosmosRLMode.disaggregated
    ):
        raise ValueError("cosmos.mode must be 'disaggregated' for slurm execution")


def _validate_slurm_topology_config(slurm: SlurmConfig) -> None:
    """Validate authored Slurm topology settings before execution."""
    match SlurmLayout(slurm.topology.kind):
        case SlurmLayout.all_in_one:
            if not isinstance(slurm.topology, AllInOneSlurmTopologyConfig):
                raise TypeError(type(slurm.topology))
            if slurm.nodes != 1:
                raise ValueError("all_in_one requires nodes=1")
            if slurm.topology.alpasim_gpus < 1:
                raise ValueError("all_in_one requires at least one AlpaSim GPU")
            if slurm.topology.alpasim_gpus >= slurm.gpus_per_node:
                raise ValueError(
                    "all_in_one requires alpasim_gpus to leave a Cosmos GPU"
                )
        case SlurmLayout.separate_nodes:
            if not isinstance(slurm.topology, SeparateNodesSlurmTopologyConfig):
                raise TypeError(type(slurm.topology))
            if slurm.topology.cosmos_nodes < 1:
                raise ValueError("separate_nodes requires at least one Cosmos node")
            if slurm.topology.alpasim_nodes < 1:
                raise ValueError("separate_nodes requires at least one AlpaSim node")
            if slurm.nodes != (
                slurm.topology.cosmos_nodes + slurm.topology.alpasim_nodes
            ):
                raise ValueError(
                    "separate_nodes requires nodes to equal cosmos_nodes + alpasim_nodes"
                )


def _validate_policy_model_path(config: RunConfig) -> None:
    """Validate real Alpamayo model bundles before starting external processes."""
    model_path = Path(config.policy.model.path)
    if not model_path.exists():
        raise ValueError(
            f"policy.model.path does not exist: {model_path}. "
            "Download or export an HF bundle directory containing config.json."
        )
    if not model_path.is_dir():
        raise ValueError(
            f"policy.model.path must be an extracted HF bundle directory: {model_path}. "
            "Generated runs may start from a tarball, but resolved configs must point to "
            "artifact_paths.policy_model_bundle_dir. Regenerate run artifacts from the "
            "Hydra config."
        )
    if config.policy.model.kind == "g1_vla":
        if not (model_path / "run_config.json").is_file():
            raise ValueError("g1_vla policy.model.path must contain run_config.json")
        return
    if not (model_path / "config.json").is_file():
        raise ValueError(
            f"policy.model.path is a directory without config.json: {model_path}. "
            "Pass the HF bundle directory itself, not its parent."
        )
    if not is_supported_hf_bundle_dir(model_path):
        raise ValueError(
            "policy.model.path HF bundle directory must contain config.json and a "
            f"supported checkpoint weight file or shard index such as model*.safetensors, "
            f"pytorch_model*.bin, or model.safetensors.index.json: {model_path}"
        )


def _validate_transport_config(config: RunConfig) -> None:
    """Validate launcher and transport geometry before starting workers."""
    if config.transport.kind == TransportKind.nccl:
        if config.cosmos.mode != CosmosRLMode.disaggregated:
            raise ValueError(
                "transport=nccl requires cosmos.mode=disaggregated; NCCL transfers "
                "tensors between separate rollout and policy processes, but "
                f"cosmos.mode={config.cosmos.mode} colocates them."
            )
        _validate_nccl_parallelism(config)
        _validate_nccl_env(config)


def _validate_nccl_parallelism(config: RunConfig) -> None:
    """Reject NCCL configs that the current transport does not support."""
    policy_parallelism = config.cosmos.policy.parallelism
    rollout_parallelism = config.cosmos.rollout.parallelism
    if config.cosmos.launch.policy_replicas <= 0:
        raise ValueError("NCCL transport requires cosmos.launch.policy_replicas >= 1")
    if policy_parallelism.dp_shard_size <= 0:
        raise ValueError(
            "NCCL transport requires policy.parallelism.dp_shard_size >= 1"
        )
    unsupported_policy_axes = {
        "tp_size": policy_parallelism.tp_size,
        "cp_size": policy_parallelism.cp_size,
        "ep_size": policy_parallelism.ep_size,
        "pp_size": policy_parallelism.pp_size,
        "pp_micro_batch_size": policy_parallelism.pp_micro_batch_size,
        "dp_replicate_size": policy_parallelism.dp_replicate_size,
    }
    bad_policy_axes = {
        name: value for name, value in unsupported_policy_axes.items() if value != 1
    }
    if bad_policy_axes:
        raise ValueError(
            "NCCL transport sizes policy workers from "
            "cosmos.launch.policy_replicas * policy.parallelism.dp_shard_size; "
            f"unsupported policy parallelism axes must be 1, got {bad_policy_axes}."
        )
    if config.cosmos.launch.rollout_replicas <= 0:
        raise ValueError("NCCL transport requires cosmos.launch.rollout_replicas >= 1")
    if rollout_parallelism.tp_size != 1 or rollout_parallelism.pp_size != 1:
        raise ValueError(
            "NCCL transport currently supports one process per rollout replica; "
            f"got rollout.parallelism.tp_size={rollout_parallelism.tp_size} and "
            f"rollout.parallelism.pp_size={rollout_parallelism.pp_size}."
        )


def _validate_nccl_env(config: RunConfig) -> None:
    """Reject NCCL runs whose env lacks a positive, finite NCCL_TIMEOUT.

    Workers read ``transport.nccl_env['NCCL_TIMEOUT']`` to size the NCCL timeouts.
    Validate it at preflight so a missing, non-positive, or non-finite value fails
    here rather than crashing a worker subprocess when it converts the seconds to
    milliseconds (``int(nan * 1000)`` / ``int(inf * 1000)`` raise).
    """
    timeout = config.transport.nccl_env.get("NCCL_TIMEOUT")
    if timeout is None:
        raise ValueError(
            "transport=nccl requires transport.nccl_env['NCCL_TIMEOUT']; it is missing "
            "(conf/transport/nccl.yaml ships a default)."
        )
    try:
        timeout_seconds = float(timeout)
        is_valid = math.isfinite(timeout_seconds) and timeout_seconds > 0
    except ValueError:
        is_valid = False
    if not is_valid:
        raise ValueError(
            f"transport.nccl_env['NCCL_TIMEOUT'] must be a positive, finite number, got {timeout!r}"
        )


def _validate_training_policy_config(config: RunConfig) -> None:
    """Validate policy settings required by Cosmos replay training."""
    if config.expected_valid_steps <= 0:
        raise ValueError(
            "expected_valid_steps must be positive for AlpaGym Cosmos replay training."
        )
    # The rollout horizon must match the trainer packer's per-rollout budget: the
    # policy runs only after the force-GT warmup, so a rollout yields
    # (n_sim_steps - warmup) closed-loop steps.
    wizard_args = config.alpasim.wizard_args
    warmup_steps = wizard_args.force_gt_duration_us // wizard_args.control_timestep_us
    closed_loop_steps = wizard_args.n_sim_steps - warmup_steps
    if closed_loop_steps != config.expected_valid_steps:
        raise ValueError(
            "AlpaSim rollout horizon does not match the trainer packer budget: "
            f"n_sim_steps={wizard_args.n_sim_steps} minus the force-GT warmup "
            f"({warmup_steps} = force_gt_duration_us={wizard_args.force_gt_duration_us} // "
            f"control_timestep_us={wizard_args.control_timestep_us}) = {closed_loop_steps} "
            f"closed-loop policy steps, but expected_valid_steps={config.expected_valid_steps}. "
            "Adjust expected_valid_steps (n_sim_steps follows via the config resolver), or "
            "override runtime.simulation_config.n_sim_steps via alpasim.wizard_args."
            "extra_overrides for non-per-step policies."
        )
    if config.policy.model.num_context_frames <= 0:
        raise ValueError(
            "policy.model.num_context_frames must be positive for AlpaGym Cosmos replay training."
        )
    if not config.policy.inference.return_trace_for_rl:
        raise ValueError(
            "policy.inference.return_trace_for_rl must be true for current AlpaGym "
            "Cosmos replay training. It can be false only for a rollout-only "
            "entrypoint."
        )
    if (
        isinstance(config.cosmos.train.optm_lr, bool)
        or not math.isfinite(config.cosmos.train.optm_lr)
        or config.cosmos.train.optm_lr <= 0.0
    ):
        raise ValueError("cosmos.train.optm_lr must be finite and positive")
    if any(
        isinstance(value, bool) or not math.isfinite(value) or value <= 0.0
        for value in config.cosmos.train.optm_part_lrs
    ):
        raise ValueError(
            "cosmos.train.optm_part_lrs must contain positive finite values"
        )
    if (
        not math.isfinite(config.cosmos.train.epsilon)
        or config.cosmos.train.epsilon <= 0.0
    ):
        raise ValueError("cosmos.train.epsilon must be finite and positive")
    if (
        not math.isfinite(config.cosmos.train.optm_weight_decay)
        or config.cosmos.train.optm_weight_decay < 0.0
    ):
        raise ValueError(
            "cosmos.train.optm_weight_decay must be finite and non-negative"
        )
    betas = config.cosmos.train.optm_betas
    if len(betas) != 2 or any(
        not math.isfinite(value) or not 0.0 <= value < 1.0 for value in betas
    ):
        raise ValueError("cosmos.train.optm_betas must contain two values in [0, 1)")
    if not math.isfinite(config.cosmos.train.optm_grad_norm_clip):
        raise ValueError("cosmos.train.optm_grad_norm_clip must be finite")
    train_policy = config.cosmos.train.train_policy
    if train_policy.trainer_type not in {
        "alpagym_grpo",
        "alpagym_ppo",
        "alpagym_flow_ppo",
    }:
        raise ValueError(
            "cosmos.train.train_policy.trainer_type must be alpagym_grpo, "
            "alpagym_ppo, or alpagym_flow_ppo"
        )
    if train_policy.step_mini_batch is not None and (
        isinstance(train_policy.step_mini_batch, bool)
        or train_policy.step_mini_batch <= 0
    ):
        raise ValueError("PPO step_mini_batch must be a positive integer when set")
    if (
        not math.isfinite(train_policy.ppo_value_loss_coef)
        or train_policy.ppo_value_loss_coef < 0.0
    ):
        raise ValueError("PPO ppo_value_loss_coef must be finite and non-negative")
    if train_policy.ppo_value_clip_range is not None and (
        not math.isfinite(train_policy.ppo_value_clip_range)
        or train_policy.ppo_value_clip_range <= 0.0
    ):
        raise ValueError(
            "PPO ppo_value_clip_range must be finite and positive when set"
        )
    if train_policy.ppo_dual_clip_ratio is not None and (
        not math.isfinite(train_policy.ppo_dual_clip_ratio)
        or train_policy.ppo_dual_clip_ratio <= 1.0
    ):
        raise ValueError(
            "PPO ppo_dual_clip_ratio must be finite and greater than one when set"
        )
    if train_policy.ppo_value_huber_delta is not None and (
        not math.isfinite(train_policy.ppo_value_huber_delta)
        or train_policy.ppo_value_huber_delta <= 0.0
    ):
        raise ValueError(
            "PPO ppo_value_huber_delta must be finite and positive when set"
        )
    if not 0.0 <= train_policy.ppo_gamma <= 1.0:
        raise ValueError("PPO ppo_gamma must be in [0, 1]")
    if not 0.0 <= train_policy.ppo_gae_lambda <= 1.0:
        raise ValueError("PPO ppo_gae_lambda must be in [0, 1]")
    if not (
        math.isfinite(train_policy.ppo_min_action_std)
        and math.isfinite(train_policy.ppo_max_action_std)
        and 0.0 < train_policy.ppo_min_action_std <= train_policy.ppo_max_action_std
    ):
        raise ValueError(
            "PPO action std bounds must satisfy 0 < ppo_min_action_std <= "
            "ppo_max_action_std"
        )


def _validate_cosmos_grpo_batch_geometry(cosmos: CosmosRLConfig) -> None:
    """Validate GRPO batch geometry before writing a Cosmos config."""
    policy_replicas = cosmos.launch.policy_replicas
    rollout_replicas_launch = cosmos.launch.rollout_replicas
    train_batch = cosmos.train.train_batch_per_replica
    mini_batch = cosmos.train.train_policy.mini_batch
    dp_shard_size = cosmos.policy.parallelism.dp_shard_size
    n_generation = cosmos.rollout.n_generation
    errors: list[str] = []

    if policy_replicas <= 0:
        errors.append(f"policy_replicas must be > 0, got {policy_replicas}")
    if rollout_replicas_launch <= 0:
        errors.append(f"rollout_replicas must be > 0, got {rollout_replicas_launch}")
    if train_batch <= 0:
        errors.append(f"train_batch_per_replica must be > 0, got {train_batch}")
    if mini_batch <= 0:
        errors.append(f"mini_batch must be > 0, got {mini_batch}")
    if dp_shard_size <= 0:
        errors.append(f"dp_shard_size must be > 0, got {dp_shard_size}")
    if n_generation <= 0:
        errors.append(f"n_generation must be > 0, got {n_generation}")

    if train_batch > 0 and mini_batch > 0 and train_batch % mini_batch != 0:
        errors.append(
            f"train_batch_per_replica({train_batch}) must be divisible by mini_batch({mini_batch})"
        )
    if (
        train_batch > 0
        and dp_shard_size > 0
        and mini_batch > 0
        and train_batch % (dp_shard_size * mini_batch) != 0
    ):
        errors.append(
            f"train_batch_per_replica({train_batch}) must be divisible by "
            f"dp_shard_size({dp_shard_size}) * mini_batch({mini_batch})"
        )

    if errors:
        raise ValueError("; ".join(errors))


def _validate_slurm_cosmos_gpu_capacity(config: RunConfig) -> None:
    """Validate that the Cosmos launcher plan fits the Slurm Cosmos GPU pool."""
    slurm = config.execution.slurm
    topology = build_slurm_topology(
        backend=config.execution.backend,
        hostnames=[f"node-{host_index}" for host_index in range(slurm.nodes)],
        gpus_per_node=slurm.gpus_per_node,
        topology=slurm.topology,
    )
    cosmos_hosts = topology.cosmos_host_plans
    cosmos_gpus_per_host = cosmos_hosts[0].cosmos_gpu_count
    policy_gpus_per_replica = _policy_gpus_per_replica(config.cosmos.policy.parallelism)
    rollout_gpus_per_replica = _rollout_gpus_per_replica(
        config.cosmos.rollout.parallelism
    )

    errors: list[str] = []
    if policy_gpus_per_replica > cosmos_gpus_per_host:
        errors.append(
            f"policy replica requires {policy_gpus_per_replica} GPUs but each Cosmos "
            f"Slurm worker exposes {cosmos_gpus_per_host}"
        )
    if rollout_gpus_per_replica > cosmos_gpus_per_host:
        errors.append(
            f"rollout replica requires {rollout_gpus_per_replica} GPUs but each Cosmos "
            f"Slurm worker exposes {cosmos_gpus_per_host}"
        )

    required_gpus = (
        config.cosmos.launch.policy_replicas * policy_gpus_per_replica
        + config.cosmos.launch.rollout_replicas * rollout_gpus_per_replica
    )
    available_gpus = sum(host.cosmos_gpu_count for host in cosmos_hosts)
    if required_gpus > available_gpus:
        errors.append(
            f"Cosmos Slurm GPU capacity is {available_gpus}, but policy and rollout "
            f"replicas require {required_gpus}"
        )
    if errors:
        raise ValueError("; ".join(errors))


def _validate_shared_cosmos_2_3_shape(config: RunConfig) -> None:
    """Keep the five-node 2-Cosmos / 3-AlpaSim preset's coupled dispatch shape."""
    slurm = config.execution.slurm
    if not isinstance(slurm.topology, SeparateNodesSlurmTopologyConfig):
        return
    if (
        slurm.nodes,
        slurm.topology.cosmos_nodes,
        slurm.topology.alpasim_nodes,
    ) != (5, 2, 3):
        return

    expected: list[tuple[str, object, object]] = [
        ("transport", config.transport.kind, TransportKind.disk),
        ("cosmos.launch.policy_replicas", config.cosmos.launch.policy_replicas, 4),
        ("cosmos.launch.rollout_replicas", config.cosmos.launch.rollout_replicas, 12),
        ("cosmos.rollout.batch_size", config.cosmos.rollout.batch_size, 2),
        ("cosmos.rollout.n_generation", config.cosmos.rollout.n_generation, 2),
        (
            "cosmos.train.train_batch_per_replica",
            config.cosmos.train.train_batch_per_replica,
            12,
        ),
        (
            "cosmos.train.train_policy.mini_batch",
            config.cosmos.train.train_policy.mini_batch,
            1,
        ),
        ("expected_valid_steps", config.expected_valid_steps, 22),
    ]
    mismatches = [
        f"{key}={actual!r} (expected {want!r})"
        for key, actual, want in expected
        if actual != want
    ]
    if mismatches:
        raise ValueError(
            "slurm_distributed_shared_cosmos_2_3 requires its coupled dispatch shape; "
            + "; ".join(mismatches)
        )


def _policy_gpus_per_replica(parallelism: CosmosRLPolicyParallelismConfig) -> int:
    """Return the GPU count Cosmos-RL assigns to one policy replica."""
    return (
        parallelism.tp_size
        * parallelism.dp_replicate_size
        * parallelism.pp_size
        * parallelism.cp_size
        * parallelism.dp_shard_size
    )


def _rollout_gpus_per_replica(parallelism: CosmosRLRolloutParallelismConfig) -> int:
    """Return the GPU count Cosmos-RL assigns to one rollout replica."""
    return parallelism.tp_size * parallelism.pp_size


def _require_host_path_identity_mounted(
    config: RunConfig,
    path: Path,
    label: str,
) -> None:
    """Reject host paths the Slurm container cannot open at the same absolute path."""
    resolved_path = _resolve_path(path)
    mount_srcs: list[Path] = []
    covering_non_identity_mounts: list[str] = []
    for mount in config.execution.slurm.container_mounts:
        parsed = _parse_container_mount(mount)
        if parsed is None:
            continue
        src_path, dst_path = parsed
        mount_srcs.append(src_path)
        if _is_under(resolved_path, src_path) and dst_path != src_path:
            covering_non_identity_mounts.append(mount)
    if not any(_is_under(resolved_path, src) for src in mount_srcs):
        raise ValueError(
            f"{label}={resolved_path} is not under any execution.slurm.container_mounts "
            "source path; cosmos workers in the Slurm container cannot read it. "
            f"Configured mount sources: {[str(source) for source in mount_srcs]}"
        )
    if covering_non_identity_mounts:
        raise ValueError(
            f"{label}={resolved_path} is under non-identity container mounts. "
            "AlpaGym passes absolute host paths to cosmos workers, so the container "
            "mount destination must match the host source path. "
            f"Non-identity covering mounts: {covering_non_identity_mounts}"
        )


def _validate_vla_slurm_worker_mounts(config: RunConfig) -> None:
    """Require every host-authored VLA worker path to survive Slurm unchanged."""
    humanoid = config.alpasim.humanoid
    if (
        ExecutionBackend(config.execution.backend) is not ExecutionBackend.slurm
        or config.policy.model.kind != "g1_vla"
        or humanoid is None
        or humanoid.execution_profile is not HumanoidExecutionProfile.motion_reference
    ):
        return

    model_path = Path(config.policy.model.path)
    if not model_path.expanduser().is_absolute():
        raise ValueError(
            "policy.model.path must be an absolute path for VLA Slurm runs, "
            f"got {model_path!s}"
        )
    model_root = _resolve_path(model_path)
    required_paths: list[tuple[str, Path]] = [
        ("VLA policy_eval_root", model_root.parent.parent),
        ("alpasim.humanoid.repo_path", Path(humanoid.repo_path)),
        ("alpasim.humanoid.scene_store_path", Path(humanoid.scene_store_path)),
    ]
    if humanoid.grail_root_path is None:
        raise ValueError("VLA motion_reference requires grail_root_path")
    required_paths.append(
        ("alpasim.humanoid.grail_root_path", Path(humanoid.grail_root_path))
    )
    if humanoid.scene_cache_path is None:
        raise ValueError("VLA policy camera requires scene_cache_path")
    required_paths.append(
        ("alpasim.humanoid.scene_cache_path", Path(humanoid.scene_cache_path))
    )
    if config.alpasim.repo_path is not None:
        required_paths.append(("alpasim.repo_path", Path(config.alpasim.repo_path)))
    elif config.alpasim.checkout_cache_dir is not None:
        required_paths.append(
            (
                "alpasim.checkout_cache_dir",
                Path(config.alpasim.checkout_cache_dir),
            )
        )
    else:
        raise ValueError(
            "VLA Slurm runs require either alpasim.repo_path or "
            "alpasim.checkout_cache_dir"
        )

    for label, path in required_paths:
        if not path.expanduser().is_absolute():
            raise ValueError(
                f"{label} must be an absolute path for Slurm, got {path!s}"
            )
        _require_host_path_identity_mounted(
            config=config,
            path=path,
            label=label,
        )


def _parse_container_mount(mount: str) -> tuple[Path, Path] | None:
    """Parse an enroot ``src[:dst[:flags]]`` mount into resolved source and destination."""
    parts = mount.split(":")
    src = parts[0]
    if not src or src == "none":
        return None
    dst = parts[1] if len(parts) >= 2 and parts[1] else src
    return _resolve_path(Path(src)), _resolve_path(Path(dst))


def _is_under(path: Path, prefix: Path) -> bool:
    """Return whether ``path`` is equal to or nested under ``prefix``."""
    try:
        path.relative_to(prefix)
        return True
    except ValueError:
        return False


def _resolve_path(path: Path) -> Path:
    """Resolve a path for mount comparisons without requiring it to exist."""
    return path.expanduser().resolve(strict=False)
