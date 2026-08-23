# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402

"""Trainer-free local qualification lifecycle for humanoid VLA rollouts."""

from __future__ import annotations

import json
import logging
import math
import shutil
import socket
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, cast

import grpc
import yaml

from alpagym_host.alpasim_dependency import resolve_alpasim_checkout
from alpagym_host.alpasim_wizard import (
    ensure_process_terminated,
    start_wizard,
    wait_for_runtime_ready,
)
from alpagym_host.config import (
    ExecutionBackend,
    ProvenanceMode,
    RunConfig,
    alpagym_project_root,
)
from alpagym_host.formal_run_provenance import FormalRunProvenance, RepositorySource
from alpagym_host.run_lifecycle import (
    _cleanup_wizard_processes,
    _configure_formal_subprocess_environment,
    _raise_lifecycle_failures,
    _run_formal_import_probe,
    fetch_runtime_info,
)
from alpagym_runtime.transport import DiskEpisodeWriter
from alpagym_runtime.types import EpisodeMetrics, EpisodeOutput, RewardResult

if TYPE_CHECKING:
    from alpagym_runtime.alpasim.humanoid_policy_server import HumanoidSessionRecord

logger = logging.getLogger(__name__)

_MAX_GRPC_MSG_SIZE = 256 * 1024 * 1024
_QUALIFICATION_REPLAY_SCHEMA = "g1_vla.native_ode_qualification.v1"
_QUALIFICATION_ARTIFACT_SCHEMA = "alpagym.humanoid_qualification.v1"

# These ABI-sensitive dependencies are loaded only after formal qualification has
# pinned ALPASIM_GRPC_ROOT. Tests replace them before execution, which keeps the
# host lifecycle testable without importing a GPU model or generated protobufs.
RuntimeServiceStub: Any | None = None
HumanoidPolicyServer: Any | None = None
build_simulation_request_proto: Any | None = None
build_humanoid_policy_factory: Any | None = None
build_inference_engine: Any | None = None
humanoid_policy_camera_required: Any | None = None


def _load_qualification_runtime() -> None:
    """Load policy/runtime modules after the formal gRPC source is pinned."""
    global RuntimeServiceStub
    global HumanoidPolicyServer
    global build_simulation_request_proto
    global build_humanoid_policy_factory
    global build_inference_engine
    global humanoid_policy_camera_required

    dependencies = (
        RuntimeServiceStub,
        HumanoidPolicyServer,
        build_simulation_request_proto,
        build_humanoid_policy_factory,
        build_inference_engine,
        humanoid_policy_camera_required,
    )
    if all(dependency is not None for dependency in dependencies):
        return

    from alpagym_runtime.alpasim.grpc_import import ensure_alpasim_grpc_source

    ensure_alpasim_grpc_source()
    from alpagym_runtime.alpasim.humanoid_policy_server import (
        HumanoidPolicyServer as PolicyServer,
    )
    from alpagym_runtime.alpasim.proto_conversion import (
        build_simulation_request_proto as build_request,
    )
    from alpagym_runtime.policies.factory import (
        build_humanoid_policy_factory as build_policy_factory,
    )
    from alpagym_runtime.policies.factory import (
        build_inference_engine as build_engine,
    )
    from alpagym_runtime.policies.factory import (
        humanoid_policy_camera_required as policy_camera_required,
    )
    from alpasim_grpc.v0.runtime_pb2_grpc import (
        RuntimeServiceStub as RuntimeStub,
    )

    RuntimeServiceStub = RuntimeStub
    HumanoidPolicyServer = PolicyServer
    build_simulation_request_proto = build_request
    build_humanoid_policy_factory = build_policy_factory
    build_inference_engine = build_engine
    humanoid_policy_camera_required = policy_camera_required


@dataclass(frozen=True)
class QualificationArtifacts:
    """Durable outputs produced by one trainer-free qualification rollout."""

    episode_manifest: Path
    metrics_manifest: Path


def validate_local_qualification_prerequisites() -> None:
    """Require only Wizard's Docker prerequisites, never Cosmos/Redis."""
    if shutil.which("docker") is None:
        raise ValueError(
            "command=rollout requires Docker in PATH because AlpaSim Wizard "
            "uses Docker Compose for local launches"
        )
    try:
        subprocess.run(
            ["docker", "compose", "version"],
            check=True,
            text=True,
            capture_output=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise ValueError(
            "command=rollout requires `docker compose` to be available"
        ) from exc


def execute_qualification_rollout(config: RunConfig) -> QualificationArtifacts:
    """Run one local native-ODE humanoid episode without Cosmos or a trainer."""
    if ExecutionBackend(config.execution.backend) is not ExecutionBackend.local_process:
        raise ValueError("qualification lifecycle supports local_process only")
    if config.alpasim.simulation_domain != "humanoid":
        raise ValueError("qualification lifecycle supports humanoid only")
    humanoid = config.alpasim.humanoid
    if humanoid is None:
        raise ValueError("qualification lifecycle requires alpasim.humanoid")
    scene_ids = config.dataset.scene_ids
    if scene_ids is None or len(scene_ids) != 1:
        raise ValueError("qualification lifecycle requires exactly one scene")
    scene_id = str(scene_ids[0])
    scenario_id = humanoid.scenario_ids_by_scene[scene_id]

    validate_local_qualification_prerequisites()
    checkout_root = resolve_alpasim_checkout(config=config.alpasim)
    wizard_dir = (config.artifact_paths.alpasim_log_dir / "wizard_0").resolve()
    wizard_dir.mkdir(parents=True, exist_ok=True)

    provenance: FormalRunProvenance | None = None
    critical_environment: dict[str, str] | None = None
    if config.execution.provenance_mode is ProvenanceMode.required:
        critical_environment = _configure_formal_subprocess_environment(
            config=config,
            alpasim_checkout_root=checkout_root,
        )
        provenance = FormalRunProvenance.capture_prelaunch(
            provenance_dir=config.artifact_paths.run_dir / "provenance",
            repositories=(
                RepositorySource(name="alpagym", root=alpagym_project_root()),
                RepositorySource(
                    name="alpasim",
                    root=checkout_root,
                    capture_ignored_generated_pb2=True,
                ),
                RepositorySource(name="humanoid", root=Path(humanoid.repo_path)),
            ),
            resolved_config_path=config.artifact_paths.resolved_config_path,
            cosmos_config_path=config.artifact_paths.cosmos_config_path,
            critical_environment=critical_environment,
        )

    wizard_process: subprocess.Popen[str] | None = None
    inference_engine: Any | None = None
    policy_model_source: Mapping[str, Any] | None = None
    engine_thread: threading.Thread | None = None
    policy_server: HumanoidPolicyServer | None = None
    channel: grpc.Channel | None = None
    session_uuid: str | None = None
    session_drained = False
    qualification_completed = False
    qualification_error: BaseException | None = None
    artifacts: QualificationArtifacts | None = None
    try:
        _load_qualification_runtime()
        if any(
            dependency is None
            for dependency in (
                RuntimeServiceStub,
                HumanoidPolicyServer,
                build_simulation_request_proto,
                build_humanoid_policy_factory,
                build_inference_engine,
                humanoid_policy_camera_required,
            )
        ):
            raise RuntimeError(
                "qualification runtime dependency loading was incomplete"
            )
        runtime_service_stub_type = cast(Any, RuntimeServiceStub)
        policy_server_type = cast(Any, HumanoidPolicyServer)
        build_request = cast(Any, build_simulation_request_proto)
        build_policy_factory = cast(Any, build_humanoid_policy_factory)
        build_engine = cast(Any, build_inference_engine)
        policy_camera_required = cast(Any, humanoid_policy_camera_required)
        wizard_process = start_wizard(
            config=config.alpasim,
            execution_backend=ExecutionBackend.local_process,
            dataset=config.dataset,
            alpasim_run_dir=wizard_dir,
            cwd=checkout_root,
        )
        runtime_host, runtime_port = wait_for_runtime_ready(
            wizard_process=wizard_process,
            runtime_server_path=wizard_dir / "generated-runtime-server.yaml",
            timeout_s=config.alpasim.startup_timeout_s,
            published_host="localhost",
        )
        runtime_capacity, runtime_scene_ids = fetch_runtime_info(
            runtime_host,
            runtime_port,
            timeout_s=config.alpasim.startup_timeout_s,
        )
        if runtime_capacity < 1:
            raise ValueError("AlpaSim qualification runtime reported zero capacity")
        if scene_id not in runtime_scene_ids:
            raise ValueError(
                f"AlpaSim qualification runtime does not contain scene {scene_id!r}: "
                f"{runtime_scene_ids!r}"
            )
        config.artifact_paths.alpasim_scene_ids_path.write_text(
            yaml.safe_dump({"scene_ids": runtime_scene_ids}, sort_keys=False),
            encoding="utf-8",
        )
        if provenance is not None:
            assert critical_environment is not None
            import_probe = _run_formal_import_probe(
                cosmos_command=[
                    "uv",
                    "run",
                    "--no-sync",
                    "--project",
                    str(alpagym_project_root()),
                    "--package",
                    "alpagym-host",
                    "python",
                ],
                critical_environment=critical_environment,
            )
            provenance.capture_runtime_ready(
                wizard_log_dirs=(wizard_dir,),
                workload_kind="qualification_rollout",
                workload_command=[str(Path(sys.executable).resolve()), *sys.argv],
                scene_store_root=Path(humanoid.scene_store_path),
                import_probe=import_probe,
            )

        inference_engine = build_engine(config)
        if inference_engine.requires_session_model_leases:
            raise ValueError(
                "local qualification must not require distributed model leases"
            )
        policy_model_source = _loaded_policy_model_source(inference_engine)
        _require_loaded_source_matches_config(
            config=config,
            loaded_source=policy_model_source,
        )
        policy_factory = build_policy_factory(config, inference_engine)
        endpoint_id = (
            f"humanoid-qualification-{socket.gethostname()}-{uuid.uuid4().hex}"
        )
        policy_server = policy_server_type(
            name=endpoint_id,
            max_concurrent_rollouts=1,
            policy_factory=policy_factory,
            publish_host="localhost",
            require_policy_camera=policy_camera_required(config),
        )
        policy_server.start()
        engine_thread = threading.Thread(
            target=inference_engine.run_loop,
            name="alpagym-qualification-infer",
            daemon=True,
        )
        engine_thread.start()

        channel = grpc.insecure_channel(
            f"{runtime_host}:{runtime_port}",
            options=[
                ("grpc.max_send_message_length", _MAX_GRPC_MSG_SIZE),
                ("grpc.max_receive_message_length", _MAX_GRPC_MSG_SIZE),
            ],
        )
        grpc.channel_ready_future(channel).result(
            timeout=config.alpasim.startup_timeout_s
        )
        runtime_stub = runtime_service_stub_type(channel)
        session_uuid = uuid.uuid4().hex
        behavior_policy_version = 0
        policy_server.servicer.reserve_session(
            session_uuid,
            behavior_policy_version,
        )
        request = build_request(
            scene_ids=(scene_id,),
            n_generation=1,
            humanoid_policy_host=policy_server.topology_endpoint.host,
            humanoid_policy_port=policy_server.topology_endpoint.port,
            n_concurrent_per_humanoid_policy=1,
            session_uuid=session_uuid,
            random_seed=humanoid.rollout_seed_base,
            expected_behavior_policy_version=behavior_policy_version,
            humanoid_scenario_ids=(scenario_id,),
        )
        sim_return = runtime_stub.simulate(
            request,
            timeout=config.alpasim.simulation_timeout_s,
        )
        rollout_returns = tuple(sim_return.rollout_returns)
        if len(rollout_returns) != 1:
            raise ValueError(
                "qualification simulate must return exactly one rollout, got "
                f"{len(rollout_returns)}"
            )
        rollout_return = rollout_returns[0]
        if not rollout_return.success:
            raise RuntimeError(rollout_return.error or "AlpaSim qualification failed")
        record = policy_server.servicer.pop_session_record(session_uuid)
        session_drained = True
        episode = _build_qualification_episode(
            scene_id=scene_id,
            session_uuid=session_uuid,
            rollout_seed=humanoid.rollout_seed_base,
            record=record,
            rollout_return=rollout_return,
        )
        episode_handle = Path(
            DiskEpisodeWriter(config.artifact_paths.artifacts_dir).write(episode)
        )
        metrics_path = (
            config.artifact_paths.artifacts_dir / "qualification_metrics.json"
        )
        _write_metrics_manifest(
            path=metrics_path,
            config=config,
            episode_manifest=episode_handle,
            episode=episode,
            record=record,
            policy_model_source=policy_model_source,
        )
        logger.info(
            "Qualification completed: scene=%s steps=%d episode=%s metrics=%s",
            scene_id,
            episode.num_steps,
            episode_handle,
            metrics_path,
        )
        artifacts = QualificationArtifacts(
            episode_manifest=episode_handle,
            metrics_manifest=metrics_path,
        )
        qualification_completed = True
    except BaseException as exc:
        qualification_error = exc
        raise
    finally:
        cleanup_failures: list[BaseException] = []
        if (
            session_uuid is not None
            and policy_server is not None
            and not session_drained
        ):
            try:
                policy_server.servicer.discard_session(session_uuid)
            except BaseException as exc:
                cleanup_failures.append(exc)
        if channel is not None:
            try:
                channel.close()
            except BaseException as exc:
                cleanup_failures.append(exc)
        if policy_server is not None:
            try:
                policy_server.stop()
            except BaseException as exc:
                cleanup_failures.append(exc)
        if inference_engine is not None:
            try:
                inference_engine.shutdown()
            except BaseException as exc:
                cleanup_failures.append(exc)
        if engine_thread is not None:
            try:
                engine_thread.join(timeout=30.0)
            except BaseException as exc:
                cleanup_failures.append(exc)
        if wizard_process is not None:
            try:
                if provenance is None:
                    ensure_process_terminated(wizard_process)
                else:
                    _cleanup_wizard_processes(
                        config=config,
                        execution_backend=ExecutionBackend.local_process,
                        wizard_processes=[wizard_process],
                        provenance=provenance,
                    )
            except BaseException as exc:
                cleanup_failures.append(exc)
        cleanup_error: BaseException | None = None
        if len(cleanup_failures) == 1:
            cleanup_error = cleanup_failures[0]
        elif cleanup_failures:
            cleanup_error = BaseExceptionGroup(
                "qualification cleanup failed",
                cleanup_failures,
            )
        provenance_error: BaseException | None = None
        if provenance is not None:
            try:
                provenance.finalize(
                    run_completed=qualification_completed,
                    cleanup_error=cleanup_error,
                )
            except BaseException as exc:
                provenance_error = exc
        _raise_lifecycle_failures(
            run_error=qualification_error,
            cleanup_error=cleanup_error,
            provenance_error=provenance_error,
        )
    if artifacts is None:
        raise AssertionError("successful qualification produced no artifacts")
    return artifacts


def _build_qualification_episode(
    *,
    scene_id: str,
    session_uuid: str,
    rollout_seed: int | None,
    record: HumanoidSessionRecord,
    rollout_return: object,
) -> EpisodeOutput:
    """Preserve raw native-ODE rows without attaching PPO transitions."""
    if record.behavior_policy_version != 0:
        raise ValueError("qualification record behavior_policy_version must be zero")
    returned_version = str(getattr(rollout_return, "behavior_policy_version", ""))
    if returned_version != "0":
        raise ValueError(
            "AlpaSim qualification behavior version mismatch: "
            f"returned={returned_version!r}, expected='0'"
        )
    if not record.outputs:
        raise ValueError("qualification record contains no policy outputs")
    for index, output in enumerate(record.outputs):
        replay = output.replay_data
        if replay is None or replay.payload_schema != _QUALIFICATION_REPLAY_SCHEMA:
            raise ValueError(
                f"qualification output {index} is missing the native ODE replay schema"
            )
        if output.chosen_logprob is not None or replay.old_logprob is not None:
            raise ValueError(f"qualification output {index} contains PPO logprob data")
        if replay.payload.get("sampling_mode") != "native_ode_qualification":
            raise ValueError(f"qualification output {index} changed sampling mode")
        forbidden = {
            "latent_chain",
            "old_element_logprobs",
            "transition",
        }
        leaked = forbidden.intersection(replay.payload)
        if leaked:
            raise ValueError(
                f"qualification output {index} contains trainer fields {sorted(leaked)}"
            )
        if "feedback_trace" not in replay.payload:
            raise ValueError(
                f"qualification output {index} is missing physical feedback"
            )

    aggregated = {
        str(name): float(value)
        for name, value in dict(
            getattr(rollout_return, "aggregated_metrics", {})
        ).items()
    }
    if "humanoid_total_return" not in aggregated:
        raise ValueError("qualification rollout is missing humanoid_total_return")
    reward_total = aggregated["humanoid_total_return"]
    if not math.isfinite(reward_total):
        raise ValueError("qualification humanoid_total_return is non-finite")
    dense = _dense_metrics_from_rollout_return(rollout_return)
    return EpisodeOutput(
        scene_id=scene_id,
        session_uuid=session_uuid,
        num_steps=len(record.outputs),
        policy_outputs=record.outputs,
        rollout_seed=rollout_seed,
        metrics=EpisodeMetrics(aggregated=aggregated, dense=dense),
        reward=RewardResult(total=reward_total, report_metrics=aggregated),
    )


def _dense_metrics_from_rollout_return(rollout_return: object) -> dict[str, Any]:
    """Convert RuntimeService timestep metrics into lossless JSON data."""
    dense: dict[str, Any] = {}
    for metric in getattr(rollout_return, "timestep_metrics", ()):
        name = str(metric.name)
        if name in dense:
            raise ValueError(f"qualification returned duplicate dense metric {name!r}")
        dense[name] = {
            "timestamps_us": [int(value) for value in metric.timestamps_us],
            "values": [float(value) for value in metric.values],
            "valid": [bool(value) for value in metric.valid],
        }
    return dense


def _write_metrics_manifest(
    *,
    path: Path,
    config: RunConfig,
    episode_manifest: Path,
    episode: EpisodeOutput,
    record: HumanoidSessionRecord,
    policy_model_source: Mapping[str, Any],
) -> None:
    """Write the human-readable aggregate/dense qualification summary."""
    metrics = episode.metrics
    if metrics is None:
        raise ValueError("qualification episode has no metrics")
    run_dir = config.artifact_paths.run_dir.resolve()
    try:
        episode_path = str(episode_manifest.resolve().relative_to(run_dir))
    except ValueError:
        episode_path = str(episode_manifest.resolve())
    payload: Mapping[str, Any] = {
        "artifact_schema": _QUALIFICATION_ARTIFACT_SCHEMA,
        "scene_id": episode.scene_id,
        "session_uuid": episode.session_uuid,
        "rollout_seed": episode.rollout_seed,
        "behavior_policy_version": record.behavior_policy_version,
        "policy_model_source": dict(policy_model_source),
        "num_policy_steps": episode.num_steps,
        "episode_manifest": episode_path,
        "aggregate_metrics": dict(metrics.aggregated),
        "dense_metrics": dict(metrics.dense),
        "final_bootstrap_values": {
            str(env_id): float(value)
            for env_id, value in record.final_bootstrap_values.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _loaded_policy_model_source(
    inference_engine: Any,
) -> Mapping[str, Any]:
    """Read and JSON-check identity from the model that will actually serve."""
    get_model = getattr(inference_engine, "get_model", None)
    if not callable(get_model):
        raise TypeError("qualification inference engine does not expose get_model()")
    model = get_model()
    identity = getattr(model, "alpagym_inference_source_identity", None)
    if not isinstance(identity, Mapping):
        raise ValueError(
            "qualification model is missing its verified inference-source identity"
        )
    source_kind = identity.get("source_kind")
    if source_kind not in {"base_attested", "candidate_overlay"}:
        raise ValueError("qualification model source kind is not admitted")
    try:
        # Round-trip to detach mutable model-owned containers and prove the
        # exact payload can be durably written without NaN or custom objects.
        normalized = json.loads(
            json.dumps(identity, ensure_ascii=False, allow_nan=False, sort_keys=True)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "qualification model source identity is not strict JSON"
        ) from exc
    if not isinstance(normalized, dict):
        raise TypeError("qualification model source identity must be an object")
    return normalized


def _require_loaded_source_matches_config(
    *,
    config: RunConfig,
    loaded_source: Mapping[str, Any],
) -> None:
    """Prevent a candidate request from continuing with silently loaded base."""
    bundle_config = config.policy.model.bundle_config
    expected_kind = bundle_config.get("qualification_model_source")
    if loaded_source.get("source_kind") != expected_kind:
        raise ValueError(
            "qualification loaded policy source differs from the explicit config"
        )
    if expected_kind != "candidate_overlay":
        return
    candidate = loaded_source.get("candidate_overlay")
    if not isinstance(candidate, Mapping):
        raise ValueError("qualification candidate source identity is incomplete")
    loaded_path = candidate.get("path")
    configured_path = bundle_config.get("candidate_overlay_path")
    if not isinstance(loaded_path, str) or not isinstance(configured_path, str):
        raise ValueError("qualification candidate source path is missing")
    if Path(loaded_path).resolve(strict=True) != Path(configured_path).resolve(
        strict=True
    ):
        raise ValueError(
            "qualification loaded candidate path differs from the explicit config"
        )
    loaded_sha256 = candidate.get("candidate_sha256")
    expected_sha256 = bundle_config.get("expected_candidate_sha256")
    if loaded_sha256 != expected_sha256:
        raise ValueError(
            "qualification loaded candidate SHA256 differs from the explicit config"
        )
