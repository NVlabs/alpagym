# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402

import logging
import os
import shutil
import subprocess
from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast

import grpc
import yaml
from alpagym_runtime.alpasim.grpc_import import ensure_alpasim_grpc_source

ensure_alpasim_grpc_source()
from alpasim_grpc.v0.common_pb2 import Empty
from alpasim_grpc.v0.runtime_pb2_grpc import RuntimeServiceStub

from alpagym_host.alpasim_dependency import resolve_alpasim_checkout
from alpagym_host.alpasim_wizard import (
    _build_wizard_command,
    ensure_process_terminated,
    start_wizard,
    wait_for_runtime_ready,
    wizard_compose_project,
)
from alpagym_host.config import (
    ExecutionBackend,
    ProvenanceMode,
    RunConfig,
    alpagym_project_root,
)
from alpagym_host.endpoint_registry import FileTopologyRegistry, TopologyEndpoint
from alpagym_host.formal_run_provenance import (
    FormalRunProvenance,
    RepositorySource,
    build_import_probe_receipt,
)
from alpagym_host.log_organizer import organize_role_logs, tee_role_logs
from alpagym_host.run_topology import (
    RunHostPlan,
    RunTopologyPlan,
    build_local_topology,
    build_slurm_topology,
)
from alpagym_host.slurm import (
    allocated_hostnames,
    build_cosmos_srun_command,
    build_wizard_srun_command,
    prepare_container_image,
)
from alpagym_host.transport_env import apply_transport_env_vars


_LOCAL_COMPOSE_DOWN_TIMEOUT_S = 30.0
_FORMAL_IMPORT_PROBE_MARKER = "ALPAGYM_FORMAL_IMPORT_PROBE="
_FORMAL_IMPORT_PROBE_SCRIPT = """
import importlib
import json

module_names = (
    "alpagym_host",
    "alpagym_runtime",
    "alpagym_g1_vla",
    "alpasim_grpc.v0.humanoid_pb2",
    "alpasim_grpc.v0.humanoid_pb2_grpc",
)
modules = {name: importlib.import_module(name) for name in module_names}
humanoid_pb2 = modules["alpasim_grpc.v0.humanoid_pb2"]
humanoid_pb2_grpc = modules["alpasim_grpc.v0.humanoid_pb2_grpc"]

class _ProbeChannel:
    def unary_unary(self, *args, **kwargs):
        del args, kwargs
        return object()

policy_stub = humanoid_pb2_grpc.HumanoidPolicyServiceStub(_ProbeChannel())
dynamics_stub = humanoid_pb2_grpc.HumanoidDynamicsServiceStub(_ProbeChannel())
payload = {
    "module_origins": {
        name: str(module.__file__) for name, module in modules.items()
    },
    "descriptor_fields": {
        name: {field.name: field.number for field in descriptor.fields}
        for name, descriptor in humanoid_pb2.DESCRIPTOR.message_types_by_name.items()
    },
    "descriptor_services": {
        name: {
            method.name: method.input_type.full_name
            for method in descriptor.methods
        }
        for name, descriptor in humanoid_pb2.DESCRIPTOR.services_by_name.items()
    },
    "grpc_bindings": {
        "policy_stub_abort_session": hasattr(policy_stub, "abort_session"),
        "dynamics_stub_abort_session": hasattr(dynamics_stub, "abort_session"),
        "policy_servicer_abort_session": callable(getattr(
            humanoid_pb2_grpc.HumanoidPolicyServiceServicer,
            "abort_session",
            None,
        )),
        "dynamics_servicer_abort_session": callable(getattr(
            humanoid_pb2_grpc.HumanoidDynamicsServiceServicer,
            "abort_session",
            None,
        )),
    },
}
print("ALPAGYM_FORMAL_IMPORT_PROBE=" + json.dumps(payload, sort_keys=True))
""".strip()


def fetch_runtime_info(host: str, port: int, timeout_s: float) -> tuple[int, list[str]]:
    """Fetch capacity and resolved scenes from an AlpaSim RuntimeService.

    Args:
        host: RuntimeService host.
        port: RuntimeService port.
        timeout_s: Connection and request timeout in seconds.

    Returns:
        Tuple of maximum supported concurrent rollouts and resolved scene ids.
    """
    target = f"{host}:{port}"
    with grpc.insecure_channel(target) as channel:
        grpc.channel_ready_future(channel).result(timeout=timeout_s)
        info = RuntimeServiceStub(channel).get_runtime_info(Empty(), timeout=timeout_s)
    return int(info.max_supported_concurrent_rollouts), [
        str(scene.scene_id) for scene in info.scenes
    ]


def validate_local_process_config(execution_backend: ExecutionBackend) -> None:
    """Validate local-process prerequisites before AlpaSim Wizard startup."""
    if execution_backend is not ExecutionBackend.local_process:
        return
    if shutil.which("docker") is None:
        raise ValueError(
            "execution.backend=local_process requires Docker in PATH because AlpaSim Wizard "
            "uses Docker Compose for local launches."
        )
    if shutil.which("redis-server") is None:
        raise ValueError(
            "execution.backend=local_process requires redis-server in PATH because "
            "Cosmos-RL starts a local Redis process."
        )
    if shutil.which("redis-cli") is None:
        raise ValueError(
            "execution.backend=local_process requires redis-cli in PATH because "
            "Cosmos-RL uses it to shut down its local Redis process."
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
            "execution.backend=local_process requires `docker compose` to be available."
        ) from exc


def execute_run(config: RunConfig) -> None:
    """Execute a run through the shared local and Slurm lifecycle."""
    execution_backend = ExecutionBackend(config.execution.backend)
    validate_local_process_config(execution_backend)
    scene_selector = (
        f"{len(config.dataset.scene_ids)} scene ids"
        if config.dataset.scene_ids is not None
        else f"test suite {config.dataset.test_suite_id!r}"
    )
    logging.info(
        "Starting AlpaGym run: backend=%s run_dir=%s dataset=%s policy_replicas=%d "
        "rollout_replicas=%d",
        execution_backend.value,
        config.artifact_paths.run_dir,
        scene_selector,
        config.cosmos.launch.policy_replicas,
        config.cosmos.launch.rollout_replicas,
    )
    if execution_backend.is_slurm_run:
        hostnames = allocated_hostnames()
        logging.info(
            "Resolved Slurm allocation: hostnames=%s gpus_per_node=%d",
            ",".join(hostnames),
            config.execution.slurm.gpus_per_node,
        )
        topology = build_slurm_topology(
            backend=execution_backend,
            hostnames=hostnames,
            gpus_per_node=config.execution.slurm.gpus_per_node,
            topology=config.execution.slurm.topology,
            policy_replicas=config.cosmos.launch.policy_replicas,
            rollout_replicas=config.cosmos.launch.rollout_replicas,
        )
        logging.info(
            "Preparing Slurm container image: image=%s cache_root=%s",
            config.execution.slurm.container_image,
            config.execution.slurm.container_cache_root,
        )
        container_image = prepare_container_image(
            container_image=cast(str, config.execution.slurm.container_image),
            container_cache_root=config.execution.slurm.container_cache_root,
        )
        logging.info("Using Slurm container image: %s", container_image)
    else:
        topology = build_local_topology()
        container_image = None

    _log_topology(topology)
    alpasim_checkout_root = resolve_alpasim_checkout(
        config=config.alpasim,
        prepare_local_env=not execution_backend.is_slurm_run,
    )
    if config.cosmos.train.deterministic:
        workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if workspace_config not in (None, ":4096:8"):
            raise ValueError(
                "deterministic training requires CUBLAS_WORKSPACE_CONFIG=:4096:8"
            )
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    provenance: FormalRunProvenance | None = None
    critical_environment: dict[str, str] | None = None
    if config.execution.provenance_mode is ProvenanceMode.required:
        humanoid = config.alpasim.humanoid
        if humanoid is None:
            raise AssertionError("required provenance escaped humanoid validation")
        critical_environment = _configure_formal_subprocess_environment(
            config=config,
            alpasim_checkout_root=alpasim_checkout_root,
        )
        provenance = FormalRunProvenance.capture_prelaunch(
            provenance_dir=config.artifact_paths.run_dir / "provenance",
            repositories=(
                RepositorySource(name="alpagym", root=alpagym_project_root()),
                RepositorySource(
                    name="alpasim",
                    root=alpasim_checkout_root,
                    capture_ignored_generated_pb2=True,
                ),
                RepositorySource(name="humanoid", root=Path(humanoid.repo_path)),
            ),
            resolved_config_path=config.artifact_paths.resolved_config_path,
            cosmos_config_path=config.artifact_paths.cosmos_config_path,
            critical_environment=critical_environment,
        )
    registry = FileTopologyRegistry(config.artifact_paths.topology_registry_dir)
    wizard_processes: list[subprocess.Popen[str]] = []
    alpasim_hosts = topology.alpasim_host_plans
    run_completed = False
    run_error: BaseException | None = None
    try:
        logging.info("Starting %d AlpaSim Wizard process(es)", len(alpasim_hosts))
        for runtime_index, host in enumerate(alpasim_hosts):
            wizard_processes.append(
                _start_wizard_process(
                    config=config,
                    execution_backend=execution_backend,
                    host=host,
                    runtime_index=runtime_index,
                    alpasim_checkout_root=alpasim_checkout_root,
                )
            )

        logging.info("Waiting for %d AlpaSim runtime endpoint(s)", len(alpasim_hosts))
        runtime_scene_ids: list[str] | None = None
        for runtime_index, (host, process) in enumerate(
            zip(alpasim_hosts, wizard_processes, strict=True)
        ):
            _ensure_wizard_processes_running(wizard_processes)
            wizard_log_dir = _wizard_log_dir(config=config, runtime_index=runtime_index)
            runtime_host, runtime_port = wait_for_runtime_ready(
                wizard_process=process,
                runtime_server_path=wizard_log_dir / "generated-runtime-server.yaml",
                timeout_s=config.alpasim.startup_timeout_s,
                published_host=host.hostname,
            )
            runtime_capacity, scene_ids = fetch_runtime_info(
                runtime_host,
                runtime_port,
                timeout_s=config.alpasim.startup_timeout_s,
            )
            if not scene_ids:
                raise ValueError(f"AlpaSim runtime {runtime_index} reported no scenes")
            if runtime_scene_ids is None:
                runtime_scene_ids = scene_ids
            elif scene_ids != runtime_scene_ids:
                raise ValueError(
                    "AlpaSim runtimes reported different scene lists: "
                    f"{runtime_scene_ids!r} != {scene_ids!r}"
                )
            endpoint = TopologyEndpoint(
                id=f"alpasim-runtime-{runtime_index}",
                host=runtime_host,
                port=runtime_port,
                capacity=runtime_capacity,
            )
            registry.publish_alpasim_runtime(endpoint)
            logging.info(
                "Published AlpaSim runtime: id=alpasim-runtime-%d host=%s port=%d capacity=%d",
                runtime_index,
                runtime_host,
                runtime_port,
                runtime_capacity,
            )

        config.artifact_paths.alpasim_scene_ids_path.write_text(
            yaml.safe_dump(
                {"scene_ids": runtime_scene_ids or []},
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        cosmos_command = _build_cosmos_command(
            config=config,
            execution_backend=execution_backend,
            topology=topology,
            container_image=container_image,
        )
        if provenance is not None:
            assert critical_environment is not None
            import_probe = _run_formal_import_probe(
                cosmos_command=cosmos_command,
                critical_environment=critical_environment,
            )
            provenance.capture_runtime_ready(
                wizard_log_dirs=tuple(
                    _wizard_log_dir(config=config, runtime_index=runtime_index)
                    for runtime_index in range(len(alpasim_hosts))
                ),
                workload_kind="cosmos_training",
                workload_command=cosmos_command,
                scene_store_root=Path(humanoid.scene_store_path),
                scene_cache_root=(
                    Path(humanoid.scene_cache_path)
                    if humanoid.scene_cache_path is not None
                    else None
                ),
                runtime_cache_root=(
                    Path(humanoid.runtime_cache_path)
                    if humanoid.runtime_cache_path is not None
                    else None
                ),
                controller_release_root=(
                    Path(humanoid.visual_controller_release_path)
                    if humanoid.visual_controller_release_path is not None
                    else None
                ),
                import_probe=import_probe,
            )
        logging.info(
            "Starting Cosmos launcher: backend=%s cosmos_hosts=%s log_dir=%s",
            execution_backend.value,
            ",".join(topology.cosmos_hosts),
            config.artifact_paths.log_dir,
        )
        logging.info("Starting Cosmos launcher command: %s", cosmos_command)
        # Write the NCCL fabric env to the host os.environ right before the Cosmos
        # launch. Locally the subprocess inherits it; on Slurm `srun --export=ALL`
        # carries it to every Policy/Rollout/Controller worker (and `bash -lc` does
        # not strip NCCL_*). It governs both the AlpaGym data plane and cosmos-rl's
        # native weight-sync mesh. Disk runs leave nccl_env empty, so this is a no-op.
        apply_transport_env_vars(config.transport)
        if config.transport.nccl_env:
            logging.info(
                "Applied NCCL fabric env to os.environ: %s",
                dict(config.transport.nccl_env),
            )
        # Unbuffer worker stdout so cosmos-rl's import-time print() output (e.g. its
        # model auto-discovery) is interleaved in order instead of being flushed in a
        # block when the process exits.
        os.environ["PYTHONUNBUFFERED"] = "1"
        # Only local runs inherit this terminal; Slurm replicas write their own logs.
        tee_logs = (
            tee_role_logs(config.artifact_paths.log_dir)
            if not execution_backend.is_slurm_run
            else nullcontext()
        )
        with organize_role_logs(config.artifact_paths.log_dir), tee_logs:
            subprocess.run(
                cosmos_command,
                check=True,
                text=True,
            )
        run_completed = True
        logging.info("Cosmos launcher completed")
    except BaseException as exc:
        run_error = exc
        raise
    finally:
        if wizard_processes:
            logging.info(
                "Stopping %d AlpaSim Wizard process(es)", len(wizard_processes)
            )
        cleanup_error: BaseException | None = None
        try:
            _cleanup_wizard_processes(
                config=config,
                execution_backend=execution_backend,
                wizard_processes=wizard_processes,
                provenance=provenance,
            )
        except BaseException as exc:
            cleanup_error = exc
        provenance_error: BaseException | None = None
        if provenance is not None:
            try:
                provenance.finalize(
                    run_completed=run_completed,
                    cleanup_error=cleanup_error,
                )
            except BaseException as exc:
                provenance_error = exc
        _raise_lifecycle_failures(
            run_error=run_error,
            cleanup_error=cleanup_error,
            provenance_error=provenance_error,
        )


def _configure_formal_subprocess_environment(
    *, config: RunConfig, alpasim_checkout_root: Path
) -> dict[str, str]:
    """Pin source resolution and bytecode behavior before any formal subprocess."""
    grpc_source = (alpasim_checkout_root / "src" / "grpc").resolve(strict=True)
    pycache_prefix = (config.artifact_paths.run_dir / "formal_python_cache").resolve()
    if os.path.lexists(pycache_prefix):
        raise FileExistsError(
            f"formal Python cache prefix must be fresh: {pycache_prefix}"
        )
    pycache_prefix.mkdir(parents=True)
    environment = {
        "ALPASIM_GRPC_ROOT": str(grpc_source),
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONPATH": str(grpc_source),
        "PYTHONPYCACHEPREFIX": str(pycache_prefix),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    os.environ.update(environment)
    return environment


def _run_formal_import_probe(
    *, cosmos_command: list[str], critical_environment: dict[str, str]
) -> dict[str, Any]:
    """Probe imports through the exact no-sync Python prefix used by Cosmos."""
    try:
        python_index = cosmos_command.index("python")
    except ValueError as exc:
        raise ValueError("formal Cosmos command has no Python launcher") from exc
    probe_command = [
        *cosmos_command[: python_index + 1],
        "-c",
        _FORMAL_IMPORT_PROBE_SCRIPT,
    ]
    if probe_command[:3] != ["uv", "run", "--no-sync"]:
        raise RuntimeError(
            "formal import probe requires the exact no-sync Cosmos prefix"
        )
    environment = os.environ.copy()
    for name, expected in critical_environment.items():
        if environment.get(name) != expected:
            raise RuntimeError(
                f"formal import probe environment {name} differs from prelaunch"
            )
    result = subprocess.run(
        probe_command,
        check=True,
        text=True,
        capture_output=True,
        env=environment,
    )
    payload_lines = [
        line.removeprefix(_FORMAL_IMPORT_PROBE_MARKER)
        for line in result.stdout.splitlines()
        if line.startswith(_FORMAL_IMPORT_PROBE_MARKER)
    ]
    if len(payload_lines) != 1:
        raise RuntimeError(
            "formal import probe did not return exactly one structured payload"
        )
    payload = yaml.safe_load(payload_lines[0])
    if not isinstance(payload, dict):
        raise TypeError("formal import probe payload must be an object")
    return build_import_probe_receipt(
        command=probe_command,
        environment=critical_environment,
        payload=payload,
    )


def _cleanup_wizard_processes(
    *,
    config: RunConfig,
    execution_backend: ExecutionBackend,
    wizard_processes: list[subprocess.Popen[str]],
    provenance: FormalRunProvenance | None = None,
) -> None:
    """Terminate Wizards and down only their exact local Compose projects.

    Process-group termination alone is not sufficient: interrupting Wizard can
    leave Compose containers alive. For local runs, every generated Compose file
    belonging to this lifecycle is therefore passed to an explicit, bounded
    ``docker compose --file ... down``. Failures are collected so one broken
    Wizard cannot prevent cleanup of the remaining exact projects.
    """
    failures: list[BaseException] = []
    for process in wizard_processes:
        try:
            ensure_process_terminated(process)
        except BaseException as exc:
            failures.append(exc)

    if not execution_backend.is_slurm_run:
        for runtime_index in range(len(wizard_processes)):
            wizard_log_dir = _wizard_log_dir(config=config, runtime_index=runtime_index)
            compose_path = wizard_log_dir / "docker-compose.yaml"
            compose_project = wizard_compose_project(wizard_log_dir)
            if not os.path.lexists(compose_path):
                failures.append(
                    FileNotFoundError(
                        "started local Wizard has no exact generated Compose file: "
                        f"{compose_path}"
                    )
                )
            elif compose_path.is_symlink() or not compose_path.is_file():
                failures.append(
                    RuntimeError(
                        "local Wizard cleanup found a non-regular exact Compose "
                        f"path: {compose_path}"
                    )
                )

            if provenance is not None:
                try:
                    _compose_sha256, admitted_project = (
                        provenance.cleanup_compose_identity(compose_path)
                    )
                    if admitted_project != compose_project:
                        raise RuntimeError(
                            "formal runtime Compose project differs from its immutable "
                            f"launch identity: {admitted_project!r} != "
                            f"{compose_project!r}"
                        )
                except BaseException as exc:
                    failures.append(exc)

            compose_prefix = [
                "docker",
                "compose",
                "--project-name",
                compose_project,
                "--file",
                str(compose_path),
            ]
            try:
                subprocess.run(
                    [*compose_prefix, "down"],
                    check=True,
                    text=True,
                    capture_output=True,
                    timeout=_LOCAL_COMPOSE_DOWN_TIMEOUT_S,
                )
            except BaseException as exc:
                failures.append(exc)

            try:
                remaining_container_ids = _compose_project_container_ids(
                    compose_project
                )
            except BaseException as exc:
                failures.append(exc)
                remaining_container_ids = ()
            if remaining_container_ids:
                failures.append(
                    RuntimeError(
                        "exact Wizard Compose down required label-scoped forced "
                        f"container removal: {compose_project}: "
                        f"{list(remaining_container_ids)}"
                    )
                )
                try:
                    subprocess.run(
                        ["docker", "rm", "--force", *remaining_container_ids],
                        check=True,
                        text=True,
                        capture_output=True,
                        timeout=_LOCAL_COMPOSE_DOWN_TIMEOUT_S,
                    )
                except BaseException as exc:
                    failures.append(exc)
                try:
                    remaining_after_removal = _compose_project_container_ids(
                        compose_project
                    )
                    if remaining_after_removal:
                        raise RuntimeError(
                            "exact Wizard Compose project still has containers after "
                            "label-scoped forced removal: "
                            f"{compose_project}: {list(remaining_after_removal)}"
                        )
                except BaseException as exc:
                    failures.append(exc)

    if len(failures) == 1:
        raise failures[0]
    if failures:
        raise BaseExceptionGroup(
            "one or more Wizard cleanup operations failed", failures
        )


def _compose_project_container_ids(compose_project: str) -> tuple[str, ...]:
    """Return containers carrying one immutable Wizard launch-project label."""
    result = subprocess.run(
        [
            "docker",
            "ps",
            "--all",
            "--quiet",
            "--filter",
            f"label=com.docker.compose.project={compose_project}",
        ],
        check=True,
        text=True,
        capture_output=True,
        timeout=_LOCAL_COMPOSE_DOWN_TIMEOUT_S,
    )
    container_ids = tuple(
        line.strip() for line in (result.stdout or "").splitlines() if line.strip()
    )
    if any(
        len(container_id) < 12
        or any(character not in "0123456789abcdef" for character in container_id)
        for container_id in container_ids
    ):
        raise RuntimeError(
            "Docker returned an invalid container ID for exact project cleanup: "
            f"{container_ids}"
        )
    return container_ids


def _raise_lifecycle_failures(
    *,
    run_error: BaseException | None,
    cleanup_error: BaseException | None,
    provenance_error: BaseException | None,
) -> None:
    """Preserve every lifecycle failure instead of masking teardown evidence."""
    additional_failures = [
        failure for failure in (cleanup_error, provenance_error) if failure is not None
    ]
    if run_error is not None:
        if additional_failures:
            raise BaseExceptionGroup(
                "run and finalization both failed",
                [run_error, *additional_failures],
            ) from None
        return
    if len(additional_failures) == 2:
        raise BaseExceptionGroup(
            "Wizard cleanup and formal provenance finalization both failed",
            additional_failures,
        ) from None
    if additional_failures:
        raise additional_failures[0]


def _start_wizard_process(
    config: RunConfig,
    execution_backend: ExecutionBackend,
    host: RunHostPlan,
    runtime_index: int,
    alpasim_checkout_root: Path,
) -> subprocess.Popen[str]:
    """Start one Wizard process for a topology host."""
    wizard_log_dir = _wizard_log_dir(config=config, runtime_index=runtime_index)
    wizard_log_dir.mkdir(parents=True, exist_ok=True)
    logging.info(
        "Starting AlpaSim Wizard: runtime_index=%d host=%s log_dir=%s",
        runtime_index,
        host.hostname,
        wizard_log_dir,
    )
    if not execution_backend.is_slurm_run:
        return start_wizard(
            config=config.alpasim,
            execution_backend=execution_backend,
            dataset=config.dataset,
            alpasim_run_dir=wizard_log_dir,
            cwd=alpasim_checkout_root,
        )

    wizard_baseport = config.alpasim.wizard_args.baseport + runtime_index * 100
    if wizard_baseport > 65535:
        raise ValueError(f"AlpaSim runtime {runtime_index} baseport exceeds 65535")
    wizard_command = _build_wizard_command(
        config=config.alpasim,
        execution_backend=execution_backend,
        dataset=config.dataset,
        alpasim_run_dir=wizard_log_dir,
        checkout_root=alpasim_checkout_root,
        python_executable=Path("/opt/venv/bin/python"),
        baseport=wizard_baseport,
    )
    if len(host.alpasim_gpu_ids) == 1:
        gpu_ids = f"[{host.alpasim_gpu_ids[0]}]"
        wizard_command.append(f"defines.humanoid_dynamics_gpus={gpu_ids}")
    command = build_wizard_srun_command(
        host=host,
        slurm=config.execution.slurm,
        wizard_command=wizard_command,
        wizard_workdir=alpasim_checkout_root,
        log_path=(
            config.artifact_paths.log_dir / f"wizard_{runtime_index}.log"
        ).resolve(),
    )
    logging.info(
        "Submitting AlpaSim Wizard through srun: runtime_index=%d host=%s slurm_log=%s",
        runtime_index,
        host.hostname,
        config.artifact_paths.log_dir / f"wizard_{runtime_index}.log",
    )
    logging.info("Submitting AlpaSim Wizard command: %s", command)
    # `start_new_session=True` makes the srun client its own process-group leader, so the
    # shared `ensure_process_terminated` can `os.killpg` it on cleanup (srun then forwards
    # the signal to the remote Wizard step). Without it killpg targets a non-existent group
    # and silently no-ops, leaking the Wizard srun. This mirrors the local `start_wizard`.
    return subprocess.Popen(
        command, cwd=alpasim_checkout_root, start_new_session=True, text=True
    )


def _wizard_log_dir(config: RunConfig, runtime_index: int) -> Path:
    """Return the Wizard run directory for one runtime index."""
    return (config.artifact_paths.alpasim_log_dir / f"wizard_{runtime_index}").resolve()


def _build_cosmos_command(
    config: RunConfig,
    execution_backend: ExecutionBackend,
    topology: RunTopologyPlan,
    container_image: str | None,
) -> list[str]:
    """Build the Cosmos launcher command for the selected execution backend."""
    if not execution_backend.is_slurm_run:
        return _build_cosmos_launcher_command(
            config,
            project_root=alpagym_project_root(),
            no_sync=True,
            worker_count=1,
            worker_index=0,
            controller_port=config.cosmos.launch.controller_port,
        )

    Path(config.execution.slurm.uv_cache_dir).mkdir(parents=True, exist_ok=True)
    cosmos_hosts = topology.cosmos_host_plans
    controller_host = cosmos_hosts[0].hostname
    worker_count = sum(len(host.cosmos_workers) for host in cosmos_hosts)
    worker_commands: list[tuple[list[str], ...]] = []
    for host in cosmos_hosts:
        host_commands: list[list[str]] = []
        for worker in host.cosmos_workers:
            worker_index = worker.global_worker_index
            controller_url = None
            if worker_index != 0:
                controller_address = (
                    "localhost" if host.hostname == controller_host else controller_host
                )
                controller_url = (
                    f"{controller_address}:{config.cosmos.launch.controller_port}"
                )
            host_commands.append(
                _build_cosmos_launcher_command(
                    config,
                    project_root=Path(config.execution.slurm.container_workdir),
                    no_sync=True,
                    controller_port=(
                        config.cosmos.launch.controller_port
                        if worker_index == 0
                        else None
                    ),
                    controller_url=controller_url,
                    worker_count=worker_count,
                    worker_index=worker_index,
                )
            )
        worker_commands.append(tuple(host_commands))
    return build_cosmos_srun_command(
        cosmos_hosts=cosmos_hosts,
        slurm=config.execution.slurm,
        container_image=cast(str, container_image),
        runtime_check_command=["true"],
        worker_commands=tuple(worker_commands),
        log_dir=config.artifact_paths.log_dir,
    )


def _log_topology(topology: RunTopologyPlan) -> None:
    """Log the planned host roles and GPU placement."""
    logging.info(
        "Run topology: hosts=%d cosmos_hosts=%s alpasim_hosts=%s",
        len(topology.hosts),
        ",".join(topology.cosmos_hosts),
        ",".join(topology.alpasim_hosts),
    )
    for host in topology.hosts:
        logging.info(
            "Run host plan: host_index=%d hostname=%s cosmos_gpus=%d "
            "alpasim_gpus=%d cosmos_gpu_ids=%s alpasim_gpu_ids=%s",
            host.host_index,
            host.hostname,
            host.cosmos_gpus,
            host.alpasim_gpus,
            ",".join(str(gpu_id) for gpu_id in host.cosmos_gpu_ids) or "-",
            ",".join(str(gpu_id) for gpu_id in host.alpasim_gpu_ids) or "-",
        )


def _build_cosmos_launcher_command(
    config: RunConfig,
    project_root: Path,
    no_sync: bool,
    worker_count: int,
    worker_index: int,
    controller_port: int | None = None,
    controller_url: str | None = None,
) -> list[str]:
    """Build the Cosmos-RL launcher command."""
    if controller_port is not None and controller_url is not None:
        raise ValueError("Cosmos launcher command cannot set both port and url")

    if not no_sync:
        raise ValueError("Cosmos launcher must use uv --no-sync")
    command = ["uv", "run", "--no-sync"]
    launcher_args = [
        "--project",
        str(project_root),
        "--package",
        "alpagym-runtime",
        "python",
        "-m",
        "cosmos_rl.launcher.launch_all",
        "--config",
        str(config.artifact_paths.cosmos_config_path),
        "--policy",
        str(config.cosmos.launch.policy_replicas),
        "--rollout",
        str(config.cosmos.launch.rollout_replicas),
        "--num-workers",
        str(worker_count),
        "--worker-idx",
        str(worker_index),
    ]
    if controller_port is not None:
        launcher_args.extend(["--port", str(controller_port)])
    if controller_url is not None:
        launcher_args.extend(["--url", controller_url])
    launcher_args.extend(
        [
            "--log-dir",
            str(config.artifact_paths.log_dir),
            "alpagym_runtime.cosmos.entrypoint",
        ]
    )
    command.extend(launcher_args)
    return command


def _ensure_wizard_processes_running(processes: list[subprocess.Popen[str]]) -> None:
    """Raise if any Wizard process exited before runtime readiness."""
    for process in processes:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"AlpaSim Wizard exited before readiness with code {return_code}"
            )
