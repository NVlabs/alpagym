# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import signal
import subprocess
from pathlib import Path
from typing import Any

import pytest
from alpagym_host import alpasim_wizard
from alpagym_host.alpasim_wizard import (
    _build_wizard_command,
    _read_runtime_endpoint,
    ensure_process_terminated,
    start_wizard,
    wait_for_runtime_ready,
)
from alpagym_host.config import (
    AlpaSimConfig,
    AlpaSimWizardArgs,
    DatasetConfig,
    ExecutionBackend,
    HumanoidAlpaSimConfig,
    HumanoidExecutionProfile,
    HumanoidPolicyCameraProfile,
    HumanoidReferenceControllerProfile,
)


def _alpasim_config(wizard_args: AlpaSimWizardArgs | None = None) -> AlpaSimConfig:
    """Build the minimal AlpaSim config used by host-side Wizard tests."""
    return AlpaSimConfig(
        repo_url="ssh://git@example/alpasim.git",
        repo_ref="unused",
        startup_timeout_s=600.0,
        simulation_timeout_s=600.0,
        wizard_args=wizard_args
        or AlpaSimWizardArgs(
            deploy="local",
            topology="1gpu",
            driver_source="external_dynamic",
            force_gt_duration_us=3_000_000,
            control_timestep_us=100_000,
            n_sim_steps=38,
        ),
    )


def test_wizard_command_appends_host_derived_overrides_last(tmp_path: Path) -> None:
    """Keeps run-specific Wizard overrides authoritative."""
    alpasim_run_dir = tmp_path / "alpasim"

    command = _build_wizard_command(
        config=_alpasim_config(
            AlpaSimWizardArgs(
                deploy="local",
                topology="1gpu",
                driver_source="external_dynamic",
                force_gt_duration_us=1_500_000,
                control_timestep_us=100_000,
                n_sim_steps=23,
                extra_overrides=(
                    "wizard.log_dir=/ignored scenes.scene_ids='[\"ignored\"]' myparam=myvalue"
                ),
            )
        ),
        execution_backend=ExecutionBackend.local_process,
        dataset=DatasetConfig(scene_ids=["scene_a", "scene_b"], test_suite_id=None),
        alpasim_run_dir=alpasim_run_dir,
        checkout_root=tmp_path,
    )

    assert "myparam=myvalue" in command
    assert command[-4:] == [
        f"hydra.run.dir={alpasim_run_dir / 'hydra' / 'wizard'}",
        f"wizard.prometheus.file_sd_dir={alpasim_run_dir / 'prometheus' / 'file-sd'}",
        f"wizard.log_dir={alpasim_run_dir}",
        'scenes.scene_ids=["scene_a", "scene_b"]',
    ]
    assert "runtime.simulation_config.force_gt_duration_us=1500000" in command
    assert "runtime.rollout_timeout_s=540.0" in command


def test_wizard_command_can_select_test_suite(tmp_path: Path) -> None:
    """Forwards a suite selector without expanding it in AlpaGym."""
    command = _build_wizard_command(
        config=_alpasim_config(),
        execution_backend=ExecutionBackend.local_process,
        dataset=DatasetConfig(scene_ids=None, test_suite_id="alpagym_smoke"),
        alpasim_run_dir=tmp_path / "alpasim",
        checkout_root=tmp_path,
    )

    alpasim_run_dir = tmp_path / "alpasim"
    assert command[-4:] == [
        f"hydra.run.dir={alpasim_run_dir / 'hydra' / 'wizard'}",
        f"wizard.prometheus.file_sd_dir={alpasim_run_dir / 'prometheus' / 'file-sd'}",
        f"wizard.log_dir={tmp_path / 'alpasim'}",
        "scenes.test_suite_id=alpagym_smoke",
    ]
    assert not any(override.startswith("scenes.scene_ids=") for override in command)


def test_wizard_command_selects_strict_motion_reference_profile(
    tmp_path: Path,
) -> None:
    """Reference execution sends one frozen identity map and no direct-action knobs."""
    config = _alpasim_config(
        AlpaSimWizardArgs(
            deploy="local",
            topology="1gpu",
            driver_source="external_dynamic",
            force_gt_duration_us=0,
            control_timestep_us=500_000,
            n_sim_steps=60,
        )
    )
    config.simulation_domain = "humanoid"
    config.humanoid = HumanoidAlpaSimConfig(
        repo_path="/workspace/humanoid",
        scene_store_path="/workspace/scenes",
        scenario_ids_by_scene={"hq_stairs": "ascend"},
        execution_profile=HumanoidExecutionProfile.motion_reference,
        grail_root_path="/workspace/GRAIL",
        policy_camera_profile=HumanoidPolicyCameraProfile.vla_d455,
        scene_cache_path="/workspace/cache/hq_stairs",
        runtime_cache_path="/workspace/runtime-cache",
        service_image="alpasim-humanoid-nurec:local",
        reward_profile_id="reference_route_centered.v3",
        expected_scene_fingerprints={"hq_stairs": "a" * 64},
    )

    command = _build_wizard_command(
        config=config,
        execution_backend=ExecutionBackend.local_process,
        dataset=DatasetConfig(scene_ids=["hq_stairs"], test_suite_id=None),
        alpasim_run_dir=tmp_path / "alpasim",
        checkout_root=tmp_path,
    )

    assert "runtime_domain=humanoid_reference" in command
    assert "defines.humanoid_grail_root=/workspace/GRAIL" in command
    assert "defines.humanoid_image=alpasim-humanoid-nurec:local" in command
    assert (
        'defines.humanoid_scene_fingerprints_json="{\\"hq_stairs\\":\\"'
        + "a" * 64
        + '\\"}"'
    ) in command
    assert not any("registration_options" in item for item in command)
    assert "runtime.humanoid.reference.control_ticks_per_policy_step=25" in command
    assert (
        "+runtime.humanoid.controller.options.reward_profile_id="
        '"reference_route_centered.v3"'
    ) in command
    assert '+runtime.humanoid.controller.options.route_center_soft_m="0.1"' in command
    assert (
        '+runtime.humanoid.controller.options.route_progress_credit_m="0.3"' in command
    )
    assert (
        '+runtime.humanoid.controller.options.route_corridor_half_width_m="0.45"'
        in command
    )
    derived_horizon_override = (
        'runtime.humanoid.controller.options.max_control_ticks="1500"'
    )
    assert derived_horizon_override in command
    assert command.count("cameras=humanoid_vla_d455") == 1
    assert command.count("defines.humanoid_scene_cache=/workspace/cache/hq_stairs") == 1
    assert command.count("defines.humanoid_runtime_cache=/workspace/runtime-cache") == 1
    checkout_root = tmp_path.resolve()
    assert f"services.runtime.volumes.3={checkout_root / 'src'}:/repo/src:ro" in command
    assert (
        f"services.runtime.volumes.4={checkout_root / 'plugins'}:/repo/plugins:ro"
        in command
    )
    assert (
        "services.humanoid_dynamics.volumes.0="
        f"{checkout_root / 'src'}:/repo/src:ro" in command
    )
    assert (
        "services.humanoid_dynamics.volumes.1="
        f"{checkout_root / 'plugins'}:/repo/plugins:ro" in command
    )
    assert (
        f"scenes.scene_cache={tmp_path / 'alpasim' / 'scene-cache'}" in command
    )


def test_wizard_command_selects_visual_sonic_as_atomic_tracker_profile(
    tmp_path: Path,
) -> None:
    config = _alpasim_config(
        AlpaSimWizardArgs(
            deploy="local",
            topology="1gpu",
            driver_source="external_dynamic",
            force_gt_duration_us=0,
            control_timestep_us=100_000,
            n_sim_steps=150,
        )
    )
    config.simulation_domain = "humanoid"
    config.humanoid = HumanoidAlpaSimConfig(
        repo_path="/workspace/humanoid",
        scene_store_path="/workspace/scenes",
        scenario_ids_by_scene={"hq_stairs": "ascend"},
        execution_profile=HumanoidExecutionProfile.motion_reference,
        reference_controller_profile=(HumanoidReferenceControllerProfile.sonic_visual),
        visual_controller_release_path="/workspace/visual-sonic-release",
        robot_physics_profile=(
            "sonic.isaac_training.g1_cylinder_model_12.mujoco_port.v1"
        ),
        policy_camera_profile=HumanoidPolicyCameraProfile.vla_d435_native,
        scene_cache_path="/workspace/cache/hq_stairs",
        runtime_cache_path="/workspace/runtime-cache",
        service_image="alpasim-humanoid-nurec:local",
        reward_profile_id="direct_v9_shaped.v1",
        expected_scene_fingerprints={"hq_stairs": "a" * 64},
    )

    command = _build_wizard_command(
        config=config,
        execution_backend=ExecutionBackend.local_process,
        dataset=DatasetConfig(scene_ids=["hq_stairs"], test_suite_id=None),
        alpasim_run_dir=tmp_path / "alpasim",
        checkout_root=tmp_path,
    )

    assert "runtime_domain=humanoid_reference_visual" in command
    assert not any(
        item.startswith("defines.humanoid_grail_root=") for item in command
    )
    assert (
        "defines.humanoid_visual_controller_release=/workspace/visual-sonic-release"
    ) in command
    assert (
        "defines.humanoid_robot_physics_profile="
        '"sonic.isaac_training.g1_cylinder_model_12.mujoco_port.v1"'
    ) in command
    assert command.count("defines.humanoid_scene_cache=/workspace/cache/hq_stairs") == 1
    assert command.count("defines.humanoid_runtime_cache=/workspace/runtime-cache") == 1
    assert 'runtime.humanoid.controller.options.max_control_ticks="750"' in command
    assert "runtime.simulation_config.control_timestep_us=100000" in command
    assert "runtime.simulation_config.n_sim_steps=150" in command
    assert "runtime.humanoid.reference.control_ticks_per_policy_step=5" in command
    assert command.count("cameras=humanoid_vla_d435_native") == 1
    assert not any("cache_policy" in argument for argument in command)


def test_wizard_command_sends_v9_reward_without_route_center_options(
    tmp_path: Path,
) -> None:
    """The V9 scalar contract has no route-center threshold parameters."""
    config = _alpasim_config(
        AlpaSimWizardArgs(
            deploy="local",
            topology="1gpu",
            driver_source="external_dynamic",
            force_gt_duration_us=0,
            control_timestep_us=500_000,
            n_sim_steps=30,
        )
    )
    config.simulation_domain = "humanoid"
    config.humanoid = HumanoidAlpaSimConfig(
        repo_path="/workspace/humanoid",
        scene_store_path="/workspace/scenes",
        scenario_ids_by_scene={"hq_stairs": "ascend"},
        execution_profile=HumanoidExecutionProfile.motion_reference,
        grail_root_path="/workspace/GRAIL",
        reward_profile_id="direct_v9_shaped.v1",
        expected_scene_fingerprints={"hq_stairs": "a" * 64},
        runtime_spawn_root_x_m=0.9954772324738195,
        runtime_spawn_root_y_m=2.8733132015389695,
        runtime_spawn_root_yaw_rad=-0.7295696089600734,
    )

    command = _build_wizard_command(
        config=config,
        execution_backend=ExecutionBackend.local_process,
        dataset=DatasetConfig(scene_ids=["hq_stairs"], test_suite_id=None),
        alpasim_run_dir=tmp_path / "alpasim",
        checkout_root=tmp_path,
    )

    assert (
        '+runtime.humanoid.controller.options.reward_profile_id="direct_v9_shaped.v1"'
    ) in command
    assert not any("route_center_soft_m" in item for item in command)
    assert not any("route_progress_credit_m" in item for item in command)
    assert not any("route_corridor_half_width_m" in item for item in command)
    assert (
        "+runtime.humanoid.controller.options.runtime_spawn_override_schema="
        '"reference_tracking.runtime_spawn_xy_yaw.v1"'
    ) in command
    assert (
        '+runtime.humanoid.controller.options.runtime_spawn_root_x_m="0.9954772324738195"'
        in command
    )
    assert (
        '+runtime.humanoid.controller.options.runtime_spawn_root_y_m="2.8733132015389695"'
        in command
    )
    assert (
        '+runtime.humanoid.controller.options.runtime_spawn_root_yaw_rad="-0.7295696089600734"'
        in command
    )
    assert 'runtime.humanoid.controller.options.max_control_ticks="750"' in command


def test_wizard_command_rejects_one_pose_for_multiple_scenes(tmp_path: Path) -> None:
    """A scalar runtime reset override cannot ambiguously target many scenes."""
    config = _alpasim_config()
    config.simulation_domain = "humanoid"
    config.humanoid = HumanoidAlpaSimConfig(
        repo_path="/workspace/humanoid",
        scene_store_path="/workspace/scenes",
        scenario_ids_by_scene={"scene_a": "ascend", "scene_b": "ascend"},
        execution_profile=HumanoidExecutionProfile.motion_reference,
        grail_root_path="/workspace/GRAIL",
        reward_profile_id="direct_v9_shaped.v1",
        runtime_spawn_root_x_m=1.0,
        runtime_spawn_root_y_m=2.0,
        runtime_spawn_root_yaw_rad=0.0,
    )

    with pytest.raises(ValueError, match="exactly one selected scene"):
        _build_wizard_command(
            config=config,
            execution_backend=ExecutionBackend.local_process,
            dataset=DatasetConfig(scene_ids=["scene_a", "scene_b"], test_suite_id=None),
            alpasim_run_dir=tmp_path / "alpasim",
            checkout_root=tmp_path,
        )


@pytest.mark.parametrize(
    "reserved_override",
    [
        'runtime.humanoid.controller.options.reward_profile_id="other"',
        '+runtime.humanoid.controller.options.route_center_soft_m="0.2"',
        '++runtime.humanoid.controller.options.max_control_ticks="1"',
        "~runtime.humanoid.controller.options",
        "runtime.humanoid.controller.module=untrusted_backend",
        "runtime.humanoid.controller@runtime.humanoid.controller=untrusted_backend",
        "cameras=motion_reference_debug",
        "runtime.simulation_config.cameras=[]",
        "runtime.simulation_config.image_format=jpeg",
        "runtime.humanoid.policy_camera.schema=untrusted.v0",
        "runtime.humanoid.session_cleanup_timeout_s=9999",
        "defines.humanoid_scene_cache=/tmp/untrusted",
        "defines.humanoid_runtime_cache=/tmp/untrusted",
        "defines={humanoid_robot_physics_profile:untrusted}",
        "runtime.humanoid={num_envs:99}",
        "services.runtime.volumes=[]",
        "runtime_domain=humanoid",
    ],
)
def test_wizard_command_rejects_motion_reference_controller_overrides(
    tmp_path: Path,
    reserved_override: str,
) -> None:
    """Raw Hydra options cannot replace the trusted controller or reward contract."""
    config = _alpasim_config(
        AlpaSimWizardArgs(
            deploy="local",
            topology="1gpu",
            driver_source="external_dynamic",
            force_gt_duration_us=0,
            control_timestep_us=100_000,
            n_sim_steps=300,
            extra_overrides=reserved_override,
        )
    )
    config.simulation_domain = "humanoid"
    config.humanoid = HumanoidAlpaSimConfig(
        repo_path="/workspace/humanoid",
        scene_store_path="/workspace/scenes",
        scenario_ids_by_scene={"hq_stairs": "ascend"},
        execution_profile=HumanoidExecutionProfile.motion_reference,
        grail_root_path="/workspace/GRAIL",
        reward_profile_id="reference_route_centered.v3",
    )

    with pytest.raises(
        ValueError,
        match="motion_reference extra_overrides cannot modify host-owned",
    ):
        _build_wizard_command(
            config=config,
            execution_backend=ExecutionBackend.local_process,
            dataset=DatasetConfig(scene_ids=["hq_stairs"], test_suite_id=None),
            alpasim_run_dir=tmp_path / "alpasim",
            checkout_root=tmp_path,
        )


def test_start_wizard_uses_separate_process_group(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Launches Wizard in a new process group so cleanup reaches docker compose."""

    class FakeProcess:
        """Minimal `Popen` replacement for capturing launch kwargs."""

    calls: list[dict[str, Any]] = []

    def fake_popen(*args: Any, **kwargs: Any) -> FakeProcess:
        del args
        calls.append(kwargs)
        return FakeProcess()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    process = start_wizard(
        config=_alpasim_config(),
        execution_backend=ExecutionBackend.local_process,
        dataset=DatasetConfig(scene_ids=["scene_a"], test_suite_id=None),
        alpasim_run_dir=tmp_path / "alpasim",
        cwd=tmp_path,
    )

    assert isinstance(process, FakeProcess)
    assert calls[0]["start_new_session"] is True


def test_start_wizard_requires_scene_cache_and_creates_runtime_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Keeps scene inputs immutable while preparing mutable native caches."""

    class FakeProcess:
        """Minimal process returned by the patched launcher."""

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    config = _alpasim_config(
        AlpaSimWizardArgs(
            deploy="local",
            topology="1gpu",
            driver_source="external_dynamic",
            force_gt_duration_us=0,
            control_timestep_us=500_000,
            n_sim_steps=60,
        )
    )
    config.simulation_domain = "humanoid"
    cache_path = tmp_path / "scene-cache"
    cache_path.mkdir()
    runtime_cache_path = tmp_path / "runtime-cache"
    config.humanoid = HumanoidAlpaSimConfig(
        repo_path="/workspace/humanoid",
        scene_store_path="/workspace/scenes",
        scene_cache_path=str(cache_path),
        runtime_cache_path=str(runtime_cache_path),
        scenario_ids_by_scene={"hq_stairs": "ascend"},
        execution_profile=HumanoidExecutionProfile.motion_reference,
        grail_root_path="/workspace/GRAIL",
        policy_camera_profile=HumanoidPolicyCameraProfile.vla_d455,
        service_image="alpasim-humanoid-nurec:local",
        reward_profile_id="reference_route_centered.v3",
    )

    start_wizard(
        config=config,
        execution_backend=ExecutionBackend.local_process,
        dataset=DatasetConfig(scene_ids=["hq_stairs"], test_suite_id=None),
        alpasim_run_dir=tmp_path / "alpasim",
        cwd=tmp_path,
    )

    assert cache_path.is_dir()
    assert runtime_cache_path.is_dir()


def test_start_wizard_requires_existing_typed_scene_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Formal runs fail before launch instead of creating an empty cache root."""

    popen_called = False

    def fake_popen(*args: Any, **kwargs: Any) -> None:
        nonlocal popen_called
        del args, kwargs
        popen_called = True

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    config = _alpasim_config(
        AlpaSimWizardArgs(
            deploy="local",
            topology="1gpu",
            driver_source="external_dynamic",
            force_gt_duration_us=0,
            control_timestep_us=500_000,
            n_sim_steps=60,
        )
    )
    config.simulation_domain = "humanoid"
    cache_path = tmp_path / "missing-render-cache"
    runtime_cache_path = tmp_path / "runtime-cache"
    config.humanoid = HumanoidAlpaSimConfig(
        repo_path="/workspace/humanoid",
        scene_store_path="/workspace/scenes",
        scene_cache_path=str(cache_path),
        runtime_cache_path=str(runtime_cache_path),
        scenario_ids_by_scene={"hq_stairs": "ascend"},
        execution_profile=HumanoidExecutionProfile.motion_reference,
        grail_root_path="/workspace/GRAIL",
        policy_camera_profile=HumanoidPolicyCameraProfile.vla_d455,
        service_image="alpasim-humanoid-nurec:local",
        reward_profile_id="reference_route_centered.v3",
    )

    with pytest.raises(FileNotFoundError, match="required humanoid scene cache"):
        start_wizard(
            config=config,
            execution_backend=ExecutionBackend.local_process,
            dataset=DatasetConfig(scene_ids=["hq_stairs"], test_suite_id=None),
            alpasim_run_dir=tmp_path / "alpasim",
            cwd=tmp_path,
        )

    assert not cache_path.exists()
    assert not runtime_cache_path.exists()
    assert popen_called is False


def test_ensure_process_terminated_signals_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Terminates the Wizard process group instead of only the parent process."""

    class FakeProcess:
        """Minimal process handle that exits after the first wait."""

        pid = 1234

        def poll(self) -> None:
            """Report the process as still running."""
            return None

        def wait(self, timeout: float | None = None) -> int:
            """Record that cleanup waited for graceful termination."""
            del timeout
            return 0

    signals: list[tuple[int, signal.Signals]] = []

    def fake_killpg(pid: int, sig: signal.Signals) -> None:
        signals.append((pid, sig))

    monkeypatch.setattr(os, "killpg", fake_killpg)

    ensure_process_terminated(FakeProcess())

    assert signals == [(1234, signal.SIGTERM)]


def test_start_wizard_runs_checkout_interpreter_with_clean_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runs Wizard on the checkout interpreter and strips uv venv vars from its env.

    The sim services inherit this env via `srun`, so a leaked UV_PROJECT_ENVIRONMENT
    would point them at the unmounted checkout venv and break startup.
    """
    captured: dict[str, object] = {}

    class FakePopen:
        def __init__(self, argv, cwd=None, env=None, text=None, start_new_session=None):
            del cwd, text, start_new_session
            captured["argv"] = argv
            captured["env"] = env

    monkeypatch.setenv("VIRTUAL_ENV", "/opt/venv")
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/opt/venv")
    monkeypatch.setattr(alpasim_wizard.subprocess, "Popen", FakePopen)

    checkout_root = tmp_path / "alpasim"
    start_wizard(
        config=_alpasim_config(),
        execution_backend=ExecutionBackend.local_process,
        dataset=DatasetConfig(scene_ids=["scene_a"], test_suite_id=None),
        alpasim_run_dir=tmp_path / "run",
        cwd=checkout_root,
    )

    argv = captured["argv"]
    assert argv[:3] == [
        str(checkout_root / ".venv" / "bin" / "python"),
        "-m",
        "alpasim_wizard",
    ]
    env = captured["env"]
    assert env is not None
    assert "UV_PROJECT_ENVIRONMENT" not in env
    assert "VIRTUAL_ENV" not in env
    assert env["COMPOSE_PROJECT_NAME"] == alpasim_wizard.wizard_compose_project(
        tmp_path / "run"
    )


def test_wizard_compose_project_is_stable_and_run_unique(tmp_path: Path) -> None:
    """Identical runtime indices in different runs never share cleanup labels."""
    first = tmp_path / "run-a" / "alpasim" / "wizard_0"
    second = tmp_path / "run-b" / "alpasim" / "wizard_0"

    first_project = alpasim_wizard.wizard_compose_project(first)

    assert first_project == alpasim_wizard.wizard_compose_project(first)
    assert first_project != alpasim_wizard.wizard_compose_project(second)
    assert first_project.startswith("alpagym_")
    assert len(first_project) == len("alpagym_") + 24


@pytest.mark.parametrize("contents", ["host: localhost\n", "host: ["])
def test_runtime_endpoint_waits_for_partial_or_unparseable_file(
    tmp_path: Path,
    contents: str,
) -> None:
    """Treats partial endpoint writes as not ready yet."""
    runtime_server_path = tmp_path / "generated-runtime-server.yaml"
    runtime_server_path.write_text(contents, encoding="utf-8")

    assert _read_runtime_endpoint(runtime_server_path) is None


def test_wait_for_runtime_ready_publishes_caller_host(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use Wizard's port but publish the host selected by the lifecycle."""
    runtime_server_path = tmp_path / "generated-runtime-server.yaml"
    runtime_server_path.write_text(
        "host: wizard-internal\nport: 30051\n", encoding="utf-8"
    )
    connection_attempts: list[tuple[str, int]] = []

    class FakeWizardProcess:
        """Wizard process stand-in that keeps running."""

        def poll(self) -> None:
            """Return None to indicate the process is still running."""
            return None

    class FakeConnection:
        """Socket context manager stand-in."""

        def __enter__(self) -> "FakeConnection":
            """Enter the fake connection context."""
            return self

        def __exit__(self, *args: object) -> None:
            """Exit the fake connection context."""

    def fake_create_connection(
        address: tuple[str, int], timeout: float
    ) -> FakeConnection:
        """Capture the probed endpoint."""
        del timeout
        connection_attempts.append(address)
        return FakeConnection()

    monkeypatch.setattr(
        "alpagym_host.alpasim_wizard.socket.create_connection",
        fake_create_connection,
    )

    assert wait_for_runtime_ready(
        wizard_process=FakeWizardProcess(),
        runtime_server_path=runtime_server_path,
        timeout_s=1.0,
        published_host="runtime-node-0",
    ) == ("runtime-node-0", 30051)
    assert connection_attempts == [("runtime-node-0", 30051)]
