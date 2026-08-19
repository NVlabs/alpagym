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
    RunConfig,
    SeparateNodesSlurmTopologyConfig,
    SlurmConfig,
    SlurmLayout,
    TransportKind,
)
from alpagym_host.run_artifacts import is_supported_hf_bundle_dir
from alpagym_host.humanoid_scene_identity import validate_frozen_policy_model_bundle
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
    _validate_humanoid_config(config)
    validate_frozen_policy_model_bundle(config)
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
        planner_mode = str(
            config.policy.model.bundle_config.get("planner_mode", "shadow_rollout")
        )
        if planner_mode != "shadow_rollout":
            raise ValueError(
                "VideoMimic fake plans require planner_mode='shadow_rollout' so "
                "all H70 actions come from the current policy"
            )
        if any(
            key in config.policy.model.bundle_config
            for key in ("completion_policy_path", "completion_policy_sha256")
        ):
            raise ValueError(
                "VideoMimic fake plans must not configure a frozen completion policy"
            )
        if humanoid.reference_frame_count != 70:
            raise ValueError(
                f"planner_mode={planner_mode!r} requires motion-reference H=70"
            )
        outer_period_us = int(config.alpasim.wizard_args.control_timestep_us)
        if outer_period_us != 500_000:
            raise ValueError(
                "motion_reference outer policy period is unsupported for "
                f"planner_mode={planner_mode!r}; expected 500000us"
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
    train_policy = config.cosmos.train.train_policy
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
