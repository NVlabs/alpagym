# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import subprocess
from pathlib import Path
from subprocess import CompletedProcess
from types import SimpleNamespace

import pytest
import yaml
from alpagym_host.config import (
    ExecutionBackend,
    HumanoidAlpaSimConfig,
    HumanoidExecutionProfile,
    HumanoidReferenceControllerProfile,
    ProvenanceMode,
    RunConfig,
    register_config_schema,
)
from alpagym_host.run_artifacts import build_artifact_paths, build_run_config
from alpagym_host.run_lifecycle import (
    _cleanup_wizard_processes,
    _raise_lifecycle_failures,
    execute_run,
    validate_local_process_config,
)
from hydra import compose, initialize_config_module


def test_local_process_preflight_requires_redis_server(monkeypatch) -> None:
    """Fail before Wizard startup when Cosmos-RL cannot launch its Redis process."""

    def fake_which(executable: str) -> str | None:
        return "/usr/bin/docker" if executable == "docker" else None

    monkeypatch.setattr("alpagym_host.run_lifecycle.shutil.which", fake_which)

    with pytest.raises(ValueError, match="requires redis-server in PATH"):
        validate_local_process_config(ExecutionBackend.local_process)


def test_local_process_preflight_requires_redis_cli(monkeypatch) -> None:
    """Do not launch a Redis server that Cosmos-RL cannot cleanly shut down."""

    def fake_which(executable: str) -> str | None:
        if executable in {"docker", "redis-server"}:
            return f"/usr/bin/{executable}"
        return None

    monkeypatch.setattr("alpagym_host.run_lifecycle.shutil.which", fake_which)

    with pytest.raises(ValueError, match="requires redis-cli in PATH"):
        validate_local_process_config(ExecutionBackend.local_process)


@pytest.mark.parametrize("formal_provenance", (False, True))
def test_execute_run_runs_local_process_lifecycle(
    tmp_path: Path,
    caplog,
    monkeypatch,
    formal_provenance: bool,
) -> None:
    """Local execution starts one Wizard and one Cosmos controller."""
    from alpagym_host import run_lifecycle

    model_path = _write_model_bundle_dir(tmp_path)
    register_config_schema()
    with initialize_config_module(version_base=None, config_module="alpagym_host.conf"):
        cfg = compose(
            config_name="default",
            overrides=[
                f"run_root={tmp_path.as_posix()}",
                "deploy=local",
                "topology=local_colocated_1gpu",
                "policy.model.kind=alpamayo_r1",
                f"policy.model.path={model_path.as_posix()}",
                "cosmos.launch.policy_replicas=1",
                "cosmos.launch.rollout_replicas=3",
                "cosmos.launch.controller_port=29500",
                f"alpasim.repo_path={tmp_path / 'alpasim'}",
                "alpasim.repo_url=null",
                "alpasim.repo_ref=null",
            ],
        )
    artifact_paths = build_artifact_paths(cfg)
    artifact_paths.log_dir.mkdir(parents=True)
    artifact_paths.alpasim_log_dir.mkdir(parents=True)
    config: RunConfig = build_run_config(cfg, artifact_paths)
    # The host must not replace the locked alpasim-grpc ABI with Wizard checkout
    # sources, even when the selected runtime domain is humanoid.
    config.alpasim.simulation_domain = "humanoid"
    controller_release = tmp_path / "controller_release"
    if formal_provenance:
        config.alpasim.humanoid = HumanoidAlpaSimConfig(
            repo_path=str(tmp_path / "humanoid"),
            scene_store_path=str(tmp_path / "scene_store"),
            scenario_ids_by_scene={"scene_a": "ascend", "scene_b": "ascend"},
            execution_profile=HumanoidExecutionProfile.motion_reference,
            reference_controller_profile=HumanoidReferenceControllerProfile.sonic_visual,
            visual_controller_release_path=str(controller_release),
            robot_physics_profile=(
                "sonic.isaac_training.g1_cylinder_model_12.mujoco_port.v1"
            ),
            scene_cache_path=str(tmp_path / "scene_cache"),
            runtime_cache_path=str(tmp_path / "runtime_cache"),
            service_image="alpasim-humanoid-nurec:local",
            reward_profile_id="stable_support_route.v2",
        )
        config.execution.provenance_mode = ProvenanceMode.required
    monkeypatch.delenv("ALPASIM_GRPC_ROOT", raising=False)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    config.cosmos.train.deterministic = True
    commands: list[list[str]] = []
    captured_paths: dict[str, Path] = {}
    provenance_calls: list[tuple[str, dict[str, object]]] = []

    monkeypatch.setattr(
        run_lifecycle, "validate_local_process_config", lambda backend: None
    )
    monkeypatch.setattr(
        run_lifecycle,
        "resolve_alpasim_checkout",
        lambda config, **kwargs: tmp_path / "alpasim",
    )
    if formal_provenance:
        monkeypatch.setattr(
            run_lifecycle,
            "_configure_formal_subprocess_environment",
            lambda **kwargs: {"formal": "environment"},
        )

        class Provenance:
            """Capture formal lifecycle arguments without touching the filesystem."""

            def capture_runtime_ready(self, **kwargs: object) -> None:
                """Record runtime admission arguments."""
                provenance_calls.append(("runtime_ready", kwargs))

            def cleanup_compose_identity(self, compose_path: Path) -> tuple[str, str]:
                """Return the project identity expected by exact cleanup."""
                return "unused-sha256", run_lifecycle.wizard_compose_project(
                    compose_path.parent
                )

            def finalize(self, **kwargs: object) -> None:
                """Record postrun finalization arguments."""
                provenance_calls.append(("finalize", kwargs))

        provenance = Provenance()

        class ProvenanceFactory:
            """Return the test provenance owner at prelaunch."""

            @staticmethod
            def capture_prelaunch(**kwargs: object) -> Provenance:
                """Record prelaunch arguments and return the lifecycle owner."""
                provenance_calls.append(("prelaunch", kwargs))
                return provenance

        monkeypatch.setattr(run_lifecycle, "FormalRunProvenance", ProvenanceFactory)
        monkeypatch.setattr(
            run_lifecycle,
            "_run_formal_import_probe",
            lambda **kwargs: {"probe": "receipt"},
        )

    def fake_wait_for_runtime_ready(**kwargs: object) -> tuple[str, int]:
        captured_paths["runtime_server_path"] = kwargs["runtime_server_path"]
        return kwargs["published_host"], 30051

    monkeypatch.setattr(
        run_lifecycle, "wait_for_runtime_ready", fake_wait_for_runtime_ready
    )
    monkeypatch.setattr(
        run_lifecycle,
        "fetch_runtime_info",
        lambda host, port, timeout_s: (7, ["scene_b", "scene_a"]),
        raising=False,
    )
    monkeypatch.setattr(
        run_lifecycle,
        "ensure_process_terminated",
        lambda process: process.terminate(),
    )

    class FakePopen:
        """Behave like a live local Wizard process."""

        def poll(self):
            """Return None to indicate the process is running."""
            return None

        def terminate(self) -> None:
            """Accept graceful termination."""

        def wait(self, timeout=None):
            """Accept process waiting."""
            del timeout
            return 0

        def kill(self) -> None:
            """Accept forced termination."""

    def fake_start_wizard(**kwargs: object) -> FakePopen:
        captured_paths["alpasim_run_dir"] = kwargs["alpasim_run_dir"]
        (Path(kwargs["alpasim_run_dir"]) / "docker-compose.yaml").write_text(
            "services: {}\n", encoding="utf-8"
        )
        return FakePopen()

    monkeypatch.setattr(run_lifecycle, "start_wizard", fake_start_wizard)

    def fake_run(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        del kwargs
        commands.append(command)
        if command[0] == "uv":
            assert run_lifecycle.os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
            # Mimic Cosmos-RL writing its flat per-role logs while the launcher runs.
            cosmos_logs = artifact_paths.log_dir / "logs_20260101-000000"
            cosmos_logs.mkdir(parents=True, exist_ok=True)
            (cosmos_logs / "controller.log").write_text("ctrl", encoding="utf-8")
            (cosmos_logs / "policy_0.log").write_text("p0", encoding="utf-8")
            (cosmos_logs / "rollout_0.log").write_text("r0", encoding="utf-8")
        return CompletedProcess(args=command, returncode=0, stdout="")

    monkeypatch.setattr(run_lifecycle.subprocess, "run", fake_run)

    caplog.set_level(logging.INFO)
    execute_run(config)

    assert "ALPASIM_GRPC_ROOT" not in run_lifecycle.os.environ

    assert (
        captured_paths["alpasim_run_dir"] == artifact_paths.alpasim_log_dir / "wizard_0"
    )
    assert (
        captured_paths["runtime_server_path"]
        == artifact_paths.alpasim_log_dir / "wizard_0" / "generated-runtime-server.yaml"
    )
    scene_ids = yaml.safe_load(artifact_paths.alpasim_scene_ids_path.read_text())
    assert scene_ids == {"scene_ids": ["scene_b", "scene_a"]}
    cosmos_command = [
        "uv",
        "run",
        "--no-sync",
        "--project",
        str(run_lifecycle.alpagym_project_root()),
        "--package",
        "alpagym-runtime",
        "python",
        "-m",
        "cosmos_rl.launcher.launch_all",
        "--config",
        str(config.artifact_paths.cosmos_config_path),
        "--policy",
        "1",
        "--rollout",
        "3",
        "--num-workers",
        "1",
        "--worker-idx",
        "0",
        "--port",
        "29500",
        "--log-dir",
        str(config.artifact_paths.log_dir),
        "alpagym_runtime.cosmos.entrypoint",
    ]
    compose_path = artifact_paths.alpasim_log_dir / "wizard_0" / "docker-compose.yaml"
    compose_project = run_lifecycle.wizard_compose_project(compose_path.parent)
    assert commands == [
        cosmos_command,
        [
            "docker",
            "compose",
            "--project-name",
            compose_project,
            "--file",
            str(compose_path),
            "down",
        ],
        [
            "docker",
            "ps",
            "--all",
            "--quiet",
            "--filter",
            f"label=com.docker.compose.project={compose_project}",
        ],
    ]
    assert f"Starting Cosmos launcher command: {cosmos_command}" in caplog.messages
    if formal_provenance:
        runtime_ready = next(
            kwargs for name, kwargs in provenance_calls if name == "runtime_ready"
        )
        assert runtime_ready["workload_kind"] == "cosmos_training"
        assert runtime_ready["controller_release_root"] == controller_release
        assert runtime_ready["scene_cache_root"] == tmp_path / "scene_cache"
        assert runtime_ready["runtime_cache_root"] == tmp_path / "runtime_cache"
    else:
        assert not provenance_calls

    # organize_role_logs wraps the Cosmos launch: its exit-path final pass must have
    # linked the flat per-role logs into the role-grouped layout by the time execute_run
    # returns.
    log_dir = artifact_paths.log_dir
    assert (log_dir / "controller.log").read_text() == "ctrl"
    assert (log_dir / "policy" / "policy_0.log").read_text() == "p0"
    assert (log_dir / "rollout" / "rollout_0.log").read_text() == "r0"


def test_execute_run_runs_distributed_slurm_topology(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Distributed Slurm starts only AlpaSim hosts as Wizards and Cosmos hosts as workers."""
    from alpagym_host import run_lifecycle

    uv_cache_dir = tmp_path / "uv-cache"
    model_path = _write_model_bundle_dir(tmp_path)
    register_config_schema()
    with initialize_config_module(version_base=None, config_module="alpagym_host.conf"):
        cfg = compose(
            config_name="default",
            overrides=[
                f"run_root={tmp_path.as_posix()}",
                "deploy=local",
                "topology=slurm_distributed_1_1_1",
                "policy.model.kind=alpamayo_r1",
                f"policy.model.path={model_path.as_posix()}",
                "execution.slurm.partition=batch",
                "execution.slurm.account=research",
                "execution.slurm.cpus_per_task=16",
                "execution.slurm.container_image=/containers/alpagym.sqsh",
                f"execution.slurm.uv_cache_dir={uv_cache_dir.as_posix()}",
                f"alpasim.repo_path={tmp_path / 'alpasim'}",
                "alpasim.repo_url=null",
                "alpasim.repo_ref=null",
                "alpasim.simulation_domain=humanoid",
            ],
        )
    artifact_paths = build_artifact_paths(cfg)
    artifact_paths.log_dir.mkdir(parents=True)
    artifact_paths.alpasim_log_dir.mkdir(parents=True)
    config: RunConfig = build_run_config(cfg, artifact_paths)
    commands: list[list[str]] = []

    monkeypatch.setattr(
        run_lifecycle,
        "allocated_hostnames",
        lambda: ["cosmos-0", "cosmos-1", "alpasim-0"],
    )
    monkeypatch.setattr(
        run_lifecycle,
        "resolve_alpasim_checkout",
        lambda config, **kwargs: tmp_path / "alpasim",
    )
    monkeypatch.setattr(
        run_lifecycle,
        "prepare_container_image",
        lambda container_image, container_cache_root: "/containers/alpagym.sqsh",
    )
    monkeypatch.setattr(
        run_lifecycle,
        "wait_for_runtime_ready",
        lambda **kwargs: (kwargs["published_host"], 30051),
    )
    monkeypatch.setattr(
        run_lifecycle,
        "fetch_runtime_info",
        lambda host, port, timeout_s: (9, ["scene_a", "scene_b"]),
        raising=False,
    )
    monkeypatch.setattr(
        run_lifecycle,
        "ensure_process_terminated",
        lambda process: process.terminate(),
    )

    class FakeDistributedPopen:
        """Capture Wizard commands while behaving like a live process."""

        def __init__(self, command: list[str], **kwargs: object) -> None:
            """Record the started command and its Popen kwargs."""
            self.command = command
            self.start_new_session = kwargs.get("start_new_session")
            self.terminated = False
            commands.append(command)
            wizard_processes.append(self)

        def poll(self):
            """Return None to indicate the process is running."""
            return None

        def terminate(self) -> None:
            """Record graceful termination."""
            self.terminated = True

        def wait(self, timeout=None):
            """Accept process waiting."""
            del timeout
            return 0

        def kill(self) -> None:
            """Record forced termination."""

    wizard_processes: list[FakeDistributedPopen] = []
    monkeypatch.setattr(run_lifecycle.subprocess, "Popen", FakeDistributedPopen)

    def fake_run(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        del kwargs
        commands.append(command)
        return CompletedProcess(args=command, returncode=0)

    monkeypatch.setattr(run_lifecycle.subprocess, "run", fake_run)

    execute_run(config)

    wizard_commands = [process.command for process in wizard_processes]
    assert len(wizard_commands) == 1
    assert "--nodelist=alpasim-0" in wizard_commands[0]
    assert all(process.terminated for process in wizard_processes)
    # Slurm Wizards must start their own session so cleanup's os.killpg can target them.
    assert all(process.start_new_session is True for process in wizard_processes)

    runtime_files = sorted(
        (artifact_paths.topology_registry_dir / "alpasim_runtimes").glob("*.yaml")
    )
    runtime_hosts = [
        runtime_file.read_text(encoding="utf-8") for runtime_file in runtime_files
    ]
    assert len(runtime_hosts) == 1
    assert any("host: alpasim-0" in runtime_host for runtime_host in runtime_hosts)
    assert any("capacity: 9" in runtime_host for runtime_host in runtime_hosts)

    cosmos_command = commands[1]
    assert "--nodelist=cosmos-0,cosmos-1" in cosmos_command
    assert "alpasim-0" not in " ".join(cosmos_command)
    assert str(tmp_path / "alpasim" / "src" / "grpc") not in cosmos_command[-1]
    assert "uv pip install" not in cosmos_command[-1]
    assert all("CUDA_VISIBLE_DEVICES" not in " ".join(command) for command in commands)


def test_execute_run_resolves_relative_slurm_wizard_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Relative run roots are made absolute before Slurm changes Wizard cwd."""
    from alpagym_host import run_lifecycle

    monkeypatch.chdir(tmp_path)
    model_path = _write_model_bundle_dir(tmp_path)
    register_config_schema()
    with initialize_config_module(version_base=None, config_module="alpagym_host.conf"):
        cfg = compose(
            config_name="default",
            overrides=[
                "run_root=relative-runs",
                "deploy=local",
                "topology=slurm_full_node_1_3_4",
                "policy.model.kind=alpamayo_r1",
                f"policy.model.path={model_path.as_posix()}",
                "execution.slurm.partition=batch",
                "execution.slurm.account=research",
                "execution.slurm.cpus_per_task=16",
                "execution.slurm.container_image=/containers/alpagym.sqsh",
                f"execution.slurm.uv_cache_dir={(tmp_path / 'uv-cache').as_posix()}",
                f"alpasim.repo_path={tmp_path / 'alpasim'}",
                "alpasim.repo_url=null",
                "alpasim.repo_ref=null",
            ],
        )
    artifact_paths = build_artifact_paths(cfg)
    artifact_paths.log_dir.mkdir(parents=True)
    artifact_paths.alpasim_log_dir.mkdir(parents=True)
    config: RunConfig = build_run_config(cfg, artifact_paths)
    captured_paths: dict[str, Path] = {}

    monkeypatch.setattr(run_lifecycle, "allocated_hostnames", lambda: ["single-0"])
    monkeypatch.setattr(
        run_lifecycle,
        "resolve_alpasim_checkout",
        lambda config, **kwargs: tmp_path / "alpasim",
    )
    monkeypatch.setattr(
        run_lifecycle,
        "prepare_container_image",
        lambda container_image, container_cache_root: "/containers/alpagym.sqsh",
    )

    def fake_build_wizard_command(**kwargs: object) -> list[str]:
        captured_paths["alpasim_run_dir"] = kwargs["alpasim_run_dir"]
        return ["uv", "run", "alpasim_wizard"]

    monkeypatch.setattr(
        run_lifecycle, "_build_wizard_command", fake_build_wizard_command
    )

    def fake_build_wizard_srun_command(**kwargs: object) -> list[str]:
        captured_paths["log_path"] = kwargs["log_path"]
        return ["srun", "wizard"]

    monkeypatch.setattr(
        run_lifecycle,
        "build_wizard_srun_command",
        fake_build_wizard_srun_command,
    )

    def fake_wait_for_runtime_ready(**kwargs: object) -> tuple[str, int]:
        captured_paths["runtime_server_path"] = kwargs["runtime_server_path"]
        return kwargs["published_host"], 30051

    monkeypatch.setattr(
        run_lifecycle, "wait_for_runtime_ready", fake_wait_for_runtime_ready
    )
    monkeypatch.setattr(
        run_lifecycle,
        "fetch_runtime_info",
        lambda host, port, timeout_s: (4, ["scene_a"]),
        raising=False,
    )
    monkeypatch.setattr(
        run_lifecycle,
        "ensure_process_terminated",
        lambda process: process.terminate(),
    )

    class FakePopen:
        """Behave like a running Wizard process."""

        def __init__(self, command: list[str], **kwargs: object) -> None:
            """Accept the started command."""
            del command, kwargs

        def poll(self):
            """Return None to indicate the process is running."""
            return None

        def terminate(self) -> None:
            """Accept graceful termination."""

        def wait(self, timeout=None):
            """Accept process waiting."""
            del timeout
            return 0

        def kill(self) -> None:
            """Accept forced termination."""

    monkeypatch.setattr(run_lifecycle.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(
        run_lifecycle.subprocess,
        "run",
        lambda command, **kwargs: CompletedProcess(args=command, returncode=0),
    )

    execute_run(config)

    assert captured_paths["alpasim_run_dir"].is_absolute()
    assert captured_paths["log_path"].is_absolute()
    assert captured_paths["runtime_server_path"].is_absolute()


def test_local_wizard_cleanup_downs_only_exact_generated_compose(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Successful local cleanup uses the exact this-run Compose file and a bound."""
    from alpagym_host import run_lifecycle

    config = _cleanup_test_config(tmp_path)
    compose_path = _write_cleanup_compose(config)
    compose_project = run_lifecycle.wizard_compose_project(compose_path.parent)
    other_project = run_lifecycle.wizard_compose_project(
        tmp_path / "other-run" / "alpasim" / "wizard_0"
    )
    process = object()
    terminated: list[object] = []
    commands: list[tuple[list[str], dict[str, object]]] = []
    monkeypatch.setattr(
        run_lifecycle,
        "ensure_process_terminated",
        lambda candidate: terminated.append(candidate),
    )

    def fake_run(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        commands.append((command, kwargs))
        return CompletedProcess(args=command, returncode=0)

    monkeypatch.setattr(run_lifecycle.subprocess, "run", fake_run)

    _cleanup_wizard_processes(
        config=config,
        execution_backend=ExecutionBackend.local_process,
        wizard_processes=[process],
    )

    assert terminated == [process]
    assert commands == [
        (
            [
                "docker",
                "compose",
                "--project-name",
                compose_project,
                "--file",
                str(compose_path),
                "down",
            ],
            {
                "check": True,
                "text": True,
                "capture_output": True,
                "timeout": run_lifecycle._LOCAL_COMPOSE_DOWN_TIMEOUT_S,
            },
        ),
        (
            [
                "docker",
                "ps",
                "--all",
                "--quiet",
                "--filter",
                f"label=com.docker.compose.project={compose_project}",
            ],
            {
                "check": True,
                "text": True,
                "capture_output": True,
                "timeout": run_lifecycle._LOCAL_COMPOSE_DOWN_TIMEOUT_S,
            },
        ),
    ]
    assert other_project not in repr(commands)


def test_local_wizard_cleanup_downs_compose_after_keyboard_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Interrupted process cleanup cannot skip exact Compose teardown."""
    from alpagym_host import run_lifecycle

    config = _cleanup_test_config(tmp_path)
    compose_path = _write_cleanup_compose(config)
    compose_project = run_lifecycle.wizard_compose_project(compose_path.parent)
    commands: list[list[str]] = []

    def interrupt(_process: object) -> None:
        raise KeyboardInterrupt("interrupted termination")

    monkeypatch.setattr(run_lifecycle, "ensure_process_terminated", interrupt)

    def fake_run(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        del kwargs
        commands.append(command)
        return CompletedProcess(args=command, returncode=0)

    monkeypatch.setattr(run_lifecycle.subprocess, "run", fake_run)

    with pytest.raises(KeyboardInterrupt, match="interrupted termination"):
        _cleanup_wizard_processes(
            config=config,
            execution_backend=ExecutionBackend.local_process,
            wizard_processes=[object()],
        )

    assert commands == [
        [
            "docker",
            "compose",
            "--project-name",
            compose_project,
            "--file",
            str(compose_path),
            "down",
        ],
        [
            "docker",
            "ps",
            "--all",
            "--quiet",
            "--filter",
            f"label=com.docker.compose.project={compose_project}",
        ],
    ]


def test_local_wizard_cleanup_missing_exact_compose_is_explicit_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A started local Wizard can never turn a missing Compose file into success."""
    from alpagym_host import run_lifecycle

    config = _cleanup_test_config(tmp_path)
    compose_project = run_lifecycle.wizard_compose_project(
        Path(config.artifact_paths.alpasim_log_dir) / "wizard_0"
    )
    monkeypatch.setattr(
        run_lifecycle, "ensure_process_terminated", lambda _process: None
    )
    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        del kwargs
        commands.append(command)
        return CompletedProcess(args=command, returncode=0, stdout="")

    monkeypatch.setattr(run_lifecycle.subprocess, "run", fake_run)

    with pytest.raises(FileNotFoundError, match="no exact generated Compose file"):
        _cleanup_wizard_processes(
            config=config,
            execution_backend=ExecutionBackend.local_process,
            wizard_processes=[object()],
        )

    assert commands[0][-1] == "down"
    assert commands[1] == [
        "docker",
        "ps",
        "--all",
        "--quiet",
        "--filter",
        f"label=com.docker.compose.project={compose_project}",
    ]


def test_local_wizard_cleanup_preserves_termination_and_compose_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A termination failure and exact Compose failure remain independently visible."""
    from alpagym_host import run_lifecycle

    config = _cleanup_test_config(tmp_path)
    compose_path = _write_cleanup_compose(config)
    termination_error = RuntimeError("termination failed")
    compose_error = subprocess.TimeoutExpired(
        ["docker", "compose", "--file", str(compose_path), "down"],
        timeout=30.0,
    )

    def fail_termination(_process: object) -> None:
        raise termination_error

    def fail_down(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        del kwargs
        if command[:2] == ["docker", "compose"]:
            raise compose_error
        return CompletedProcess(args=command, returncode=0, stdout="")

    monkeypatch.setattr(run_lifecycle, "ensure_process_terminated", fail_termination)
    monkeypatch.setattr(run_lifecycle.subprocess, "run", fail_down)

    with pytest.raises(BaseExceptionGroup) as captured:
        _cleanup_wizard_processes(
            config=config,
            execution_backend=ExecutionBackend.local_process,
            wizard_processes=[object()],
        )

    assert captured.value.exceptions == (termination_error, compose_error)


def test_lifecycle_preserves_cleanup_and_provenance_dual_failure() -> None:
    """Provenance failure cannot hide cleanup failure or vice versa."""
    cleanup_error = RuntimeError("cleanup failed")
    provenance_error = RuntimeError("provenance failed")

    with pytest.raises(BaseExceptionGroup) as captured:
        _raise_lifecycle_failures(
            run_error=None,
            cleanup_error=cleanup_error,
            provenance_error=provenance_error,
        )

    assert captured.value.exceptions == (cleanup_error, provenance_error)


def _cleanup_test_config(tmp_path: Path):
    """Return the minimal structural config required by exact local cleanup."""
    return SimpleNamespace(
        artifact_paths=SimpleNamespace(alpasim_log_dir=tmp_path / "alpasim")
    )


def _write_cleanup_compose(config) -> Path:
    """Create one exact Wizard-owned Compose path for cleanup tests."""
    compose_path = (
        Path(config.artifact_paths.alpasim_log_dir) / "wizard_0" / "docker-compose.yaml"
    ).resolve()
    compose_path.parent.mkdir(parents=True)
    compose_path.write_text("services: {}\n", encoding="utf-8")
    return compose_path


def _write_model_bundle_dir(tmp_path: Path) -> Path:
    """Write the minimal HF bundle shape needed by host config tests."""
    bundle_dir = tmp_path / "model_bundle"
    bundle_dir.mkdir()
    (bundle_dir / "config.json").write_text("{}", encoding="utf-8")
    (bundle_dir / "model.safetensors").write_text("weights", encoding="utf-8")
    return bundle_dir
