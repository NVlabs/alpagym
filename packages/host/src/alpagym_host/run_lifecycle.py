# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import shlex
import shutil
import subprocess
from contextlib import nullcontext
from pathlib import Path
from typing import cast

import grpc
import yaml
from alpasim_grpc.v0.common_pb2 import Empty
from alpasim_grpc.v0.runtime_pb2_grpc import RuntimeServiceStub

from alpagym_host.alpasim_dependency import resolve_alpasim_checkout
from alpagym_host.alpasim_wizard import (
    _build_wizard_command,
    ensure_process_terminated,
    start_wizard,
    wait_for_runtime_ready,
)
from alpagym_host.config import ExecutionBackend, RunConfig, alpagym_project_root
from alpagym_host.endpoint_registry import FileTopologyRegistry, TopologyEndpoint
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
    alpasim_checkout_root = resolve_alpasim_checkout(config=config.alpasim)
    registry = FileTopologyRegistry(config.artifact_paths.topology_registry_dir)
    wizard_processes: list[subprocess.Popen[str]] = []
    alpasim_hosts = topology.alpasim_host_plans
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
            alpasim_checkout_root=alpasim_checkout_root,
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
                "Applied NCCL fabric env to os.environ: %s", dict(config.transport.nccl_env)
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
        logging.info("Cosmos launcher completed")
    finally:
        if wizard_processes:
            logging.info("Stopping %d AlpaSim Wizard process(es)", len(wizard_processes))
        for process in wizard_processes:
            ensure_process_terminated(process)


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

    wizard_command = _build_wizard_command(
        config=config.alpasim,
        execution_backend=execution_backend,
        dataset=config.dataset,
        alpasim_run_dir=wizard_log_dir,
        checkout_root=alpasim_checkout_root,
    )
    command = build_wizard_srun_command(
        host=host,
        slurm=config.execution.slurm,
        wizard_command=wizard_command,
        log_path=(config.artifact_paths.log_dir / f"wizard_{runtime_index}.log").resolve(),
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
    return subprocess.Popen(command, cwd=alpasim_checkout_root, start_new_session=True, text=True)


def _wizard_log_dir(config: RunConfig, runtime_index: int) -> Path:
    """Return the Wizard run directory for one runtime index."""
    return (config.artifact_paths.alpasim_log_dir / f"wizard_{runtime_index}").resolve()


def _build_cosmos_command(
    config: RunConfig,
    execution_backend: ExecutionBackend,
    topology: RunTopologyPlan,
    container_image: str | None,
    alpasim_checkout_root: Path,
) -> list[str]:
    """Build the Cosmos launcher command for the selected execution backend."""
    if not execution_backend.is_slurm_run:
        return _build_cosmos_launcher_command(
            config,
            project_root=alpagym_project_root(),
            no_sync=False,
            worker_count=1,
            worker_index=0,
            controller_port=config.cosmos.launch.controller_port,
        )

    Path(config.execution.slurm.uv_cache_dir).mkdir(parents=True, exist_ok=True)
    cosmos_hosts = topology.cosmos_host_plans
    controller_url = f"{cosmos_hosts[0].hostname}:{config.cosmos.launch.controller_port}"
    worker_commands: list[list[str]] = []
    for worker_index, _host in enumerate(cosmos_hosts):
        worker_commands.append(
            _build_cosmos_launcher_command(
                config,
                project_root=Path(config.execution.slurm.container_workdir),
                no_sync=True,
                controller_port=(
                    config.cosmos.launch.controller_port if worker_index == 0 else None
                ),
                controller_url=controller_url if worker_index != 0 else None,
                worker_count=len(cosmos_hosts),
                worker_index=worker_index,
            )
        )
    workspace_setup_commands = [
        [
            "uv",
            "sync",
            "--frozen",
            "--inexact",
            "--all-packages",
            "--project",
            str(config.execution.slurm.container_workdir),
        ]
    ]
    if config.alpasim.simulation_domain == "humanoid":
        workspace_setup_commands.append(
            [
                "uv",
                "pip",
                "install",
                "--python",
                "/opt/venv/bin/python",
                "--reinstall",
                "--no-deps",
                str(alpasim_checkout_root / "src" / "grpc"),
            ]
        )
    return build_cosmos_srun_command(
        cosmos_hosts=cosmos_hosts,
        slurm=config.execution.slurm,
        container_image=cast(str, container_image),
        workspace_sync_command=[
            "bash",
            "-lc",
            " && ".join(shlex.join(command) for command in workspace_setup_commands),
        ],
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

    command = ["uv", "run"]
    if no_sync:
        command.append("--no-sync")
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
            raise RuntimeError(f"AlpaSim Wizard exited before readiness with code {return_code}")
