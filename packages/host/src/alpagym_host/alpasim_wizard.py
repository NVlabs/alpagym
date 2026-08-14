# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import shlex
import signal
import socket
import subprocess
import time
from pathlib import Path

import yaml

from alpagym_host.config import AlpaSimConfig, DatasetConfig, ExecutionBackend, alpagym_project_root


def _build_wizard_command(
    config: AlpaSimConfig,
    execution_backend: ExecutionBackend,
    dataset: DatasetConfig,
    alpasim_run_dir: Path,
    checkout_root: Path,
) -> list[str]:
    """Build argv that runs the AlpaSim Wizard from the checkout venv's interpreter."""
    wizard_args = config.wizard_args
    # AlpaGym ships its Wizard config groups (e.g. topology=alpagym_4gpu) as plain
    # YAML in this package. Supply them through Hydra's config search path at launch
    # rather than installing them into the checkout venv, so the shared checkout stays
    # pure AlpaSim (see resolve_alpasim_checkout).
    configs_dir = (
        alpagym_project_root()
        / "packages"
        / "alpasim_configs"
        / "src"
        / "alpagym_alpasim_configs"
        / "configs"
    )
    # Launch Wizard with the checkout venv's interpreter directly, not `uv run`.
    # See `start_wizard` for why we avoid `uv run` and its env vars here.
    argv = [
        str(checkout_root / ".venv" / "bin" / "python"),
        "-m",
        "alpasim_wizard",
        f"hydra.searchpath=[file://{configs_dir}]",
        f"deploy={wizard_args.deploy}",
        f"topology={wizard_args.topology}",
        f"driver_source={wizard_args.driver_source}",
        f"wizard.run_method={execution_backend.wizard_run_method}",
        "wizard.run_mode=SERVER",
        "wizard.debug_flags.use_localhost=true",
        "runtime.simulation_config.send_recording_ground_truth=true",
        "runtime.simulation_config.skip_driver_during_force_gt=true",
        f"runtime.simulation_domain={config.simulation_domain}",
        f"runtime.simulation_config.force_gt_duration_us={wizard_args.force_gt_duration_us}",
        f"runtime.simulation_config.control_timestep_us={wizard_args.control_timestep_us}",
        f"runtime.simulation_config.n_sim_steps={wizard_args.n_sim_steps}",
    ]
    if wizard_args.driver is not None:
        argv.append(f"driver={wizard_args.driver}")
    if wizard_args.renderer is not None:
        argv.append(f"renderer={wizard_args.renderer}")
    argv.extend(shlex.split(wizard_args.extra_overrides))
    argv.append(f"wizard.log_dir={alpasim_run_dir}")
    if dataset.scene_ids is not None:
        argv.append(f"scenes.scene_ids={json.dumps(list(dataset.scene_ids))}")
    else:
        argv.append(f"scenes.test_suite_id={dataset.test_suite_id}")
    return argv


def start_wizard(
    config: AlpaSimConfig,
    execution_backend: ExecutionBackend,
    dataset: DatasetConfig,
    alpasim_run_dir: Path,
    cwd: Path,
) -> subprocess.Popen[str]:
    """Start Wizard as a subprocess."""
    alpasim_run_dir = alpasim_run_dir.resolve()
    argv = _build_wizard_command(
        config=config,
        execution_backend=execution_backend,
        dataset=dataset,
        alpasim_run_dir=alpasim_run_dir,
        checkout_root=cwd,
    )
    # Hand Wizard a clean environment: no UV_PROJECT_ENVIRONMENT, no VIRTUAL_ENV.
    # The interpreter in `argv` already selects the checkout venv, so neither
    # variable is needed. Removing them matters because Wizard starts the sim
    # services with `srun`, which copies this environment into each service
    # container. The checkout venv is not mounted there, so a leaked
    # UV_PROJECT_ENVIRONMENT would make the in-container `uv run` build an empty
    # venv and the services would fail to start.
    env = os.environ.copy()
    env.pop("UV_PROJECT_ENVIRONMENT", None)
    env.pop("VIRTUAL_ENV", None)
    logging.info("Starting AlpaSim Wizard in %s: %s", cwd, argv)
    # `start_new_session=True` puts Wizard in its own process group so the caller
    # can SIGTERM the whole tree (Wizard + docker-compose children) via
    # `os.killpg` from `ensure_process_terminated` below.
    return subprocess.Popen(
        argv,
        cwd=cwd,
        start_new_session=True,
        env=env,
        text=True,
    )


def ensure_process_terminated(
    process: subprocess.Popen[str],
    timeout_s: float = 10.0,
) -> None:
    """Ensure a subprocess has exited, killing it if graceful termination times out."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        process.wait()


def wait_for_runtime_ready(
    wizard_process: subprocess.Popen[str],
    runtime_server_path: Path,
    timeout_s: float,
    published_host: str,
) -> tuple[str, int]:
    """Wait until Wizard's generated RuntimeService endpoint is reachable.

    Args:
        wizard_process: Wizard subprocess to monitor while waiting. If it exits
            before the endpoint becomes reachable, this function raises
            immediately instead of waiting for the timeout.
        runtime_server_path: Path to Wizard's generated runtime endpoint YAML.
            The file may not exist yet; it is polled until Wizard writes a
            parseable host and port.
        timeout_s: Maximum number of seconds to wait for the endpoint file and
            TCP readiness.
        published_host: Host that rollout workers should use for the runtime.
            Wizard's endpoint file provides the port; the lifecycle provides
            the host because it owns the launch topology.

    Returns:
        The host and port that should be published to rollout workers.
    """
    logging.info("Waiting for AlpaSim RuntimeService endpoint at %s", runtime_server_path)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        return_code = wizard_process.poll()
        if return_code is not None:
            raise RuntimeError(f"AlpaSim Wizard exited before readiness with code {return_code}")

        endpoint = _read_runtime_endpoint(runtime_server_path)
        if endpoint is not None:
            _, port = endpoint
            try:
                with socket.create_connection((published_host, port), timeout=1.0):
                    logging.info(
                        "AlpaSim RuntimeService is ready at %s:%s",
                        published_host,
                        port,
                    )
                    return published_host, port
            except OSError:
                pass

        time.sleep(1.0)

    raise TimeoutError(f"Timed out waiting for AlpaSim RuntimeService at {runtime_server_path}")


def _read_runtime_endpoint(runtime_server_path: Path) -> tuple[str, int] | None:
    """Read Wizard's generated runtime endpoint if it has been written fully."""
    if not runtime_server_path.is_file():
        return None
    try:
        data = yaml.safe_load(runtime_server_path.read_text(encoding="utf-8")) or {}
        host = data["host"]
        port = int(data["port"])
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError):
        return None
    return str(host), port
