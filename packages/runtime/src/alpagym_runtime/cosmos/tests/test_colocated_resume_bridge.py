# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for colocated checkpoint-resume step bootstrapping."""

import importlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from alpagym_host.config import CosmosRLMode

from alpagym_runtime.cosmos.colocated_resume_bridge import (
    install_colocated_resume_bootstrap_bridge,
)


def _install_fake_cosmos_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[type, type, type]:
    """Install focused Cosmos command/controller fakes for one bridge test."""

    class FakeDataFetchCommand:
        """Minimal DataFetch protocol payload."""

        def __init__(self, *, global_step: object, total_steps: object) -> None:
            """Store the two fields consumed by the bridge."""
            self.global_step = global_step
            self.total_steps = total_steps

    class FakeRolloutToRolloutBroadcastCommand:
        """Minimal R2R protocol payload."""

        def __init__(self, *, weight_step: object, total_steps: object) -> None:
            """Store the two fields consumed by the bridge."""
            self.weight_step = weight_step
            self.total_steps = total_steps

    class FakeColocatedController:
        """Exercise the bridge around Cosmos-owned synchronization behavior."""

        def __init__(
            self,
            *,
            data_fetch_command: object,
            r2r_command: object,
            current_step: int = 0,
        ) -> None:
            """Create a controller with deterministic initial commands."""
            self.data_fetch_command = data_fetch_command
            self.r2r_command = r2r_command
            self.current_step = current_step
            self.total_steps = 100
            self.rollout = object()
            self.synchronization_state: tuple[int, int] | None = None

        def init_commands(self) -> None:
            """Represent the upstream method replaced by the bridge."""
            raise AssertionError("bridge was not installed")

        def policy_consume_one_step_commands_util_data_fetch(self) -> object:
            """Return the initial remote DataFetch command."""
            return self.data_fetch_command

        def wait_for_remote_command(
            self,
            replica: object,
            *args: object,
            **kwargs: object,
        ) -> object:
            """Return the initial remote R2R command."""
            del replica, args, kwargs
            return self.r2r_command

        def rollout_consume_one_step_commands_util_r2r(self) -> bool:
            """Mirror the upstream R2R wait and observe its local step metadata."""
            self.wait_for_remote_command(self.rollout)
            self.synchronization_state = (self.current_step, self.total_steps)
            return True

    colocated_package = types.ModuleType("cosmos_rl.colocated")
    colocated_package.__path__ = []
    controller_module = types.ModuleType("cosmos_rl.colocated.controller")
    controller_module.ColocatedController = FakeColocatedController
    command_module = types.ModuleType("cosmos_rl.dispatcher.command")
    command_module.DataFetchCommand = FakeDataFetchCommand
    command_module.RolloutToRolloutBroadcastCommand = (
        FakeRolloutToRolloutBroadcastCommand
    )
    monkeypatch.setitem(sys.modules, "cosmos_rl.colocated", colocated_package)
    monkeypatch.setitem(
        sys.modules,
        "cosmos_rl.colocated.controller",
        controller_module,
    )
    monkeypatch.setitem(sys.modules, "cosmos_rl.dispatcher.command", command_module)
    return (
        FakeColocatedController,
        FakeDataFetchCommand,
        FakeRolloutToRolloutBroadcastCommand,
    )


@pytest.mark.parametrize(
    ("global_step", "total_steps", "weight_step", "r2r_total_steps"),
    [
        (1, 4, 1, None),
        (6, 10, 6, 10),
    ],
)
def test_bootstrap_labels_fresh_and_resumed_weights_with_behavior_step(
    monkeypatch: pytest.MonkeyPatch,
    global_step: int,
    total_steps: int,
    weight_step: int,
    r2r_total_steps: int | None,
) -> None:
    """Upcoming-step R2R metadata bootstraps local behavior version N."""
    controller_type, data_fetch_type, r2r_type = _install_fake_cosmos_modules(
        monkeypatch
    )
    install_colocated_resume_bootstrap_bridge()
    data_fetch_command = data_fetch_type(
        global_step=global_step,
        total_steps=total_steps,
    )
    controller = controller_type(
        data_fetch_command=data_fetch_command,
        r2r_command=r2r_type(
            weight_step=weight_step,
            total_steps=r2r_total_steps,
        ),
    )

    controller.init_commands()

    assert controller.current_step == global_step - 1
    assert controller.total_steps == total_steps
    assert controller.synchronization_state == (global_step - 1, total_steps)
    assert controller.init_data_fetch_command is data_fetch_command
    assert "wait_for_remote_command" not in controller.__dict__


@pytest.mark.parametrize(
    ("global_step", "total_steps", "weight_step", "r2r_total_steps", "message"),
    [
        (0, 10, 0, None, "global_step >= 1"),
        (6, 5, 5, None, "total_steps >= DataFetch.global_step"),
        (6, 10, 5, None, "step mismatch"),
        (6, 10, 6, 9, "total-step mismatch"),
    ],
)
def test_bootstrap_rejects_inconsistent_remote_step_metadata(
    monkeypatch: pytest.MonkeyPatch,
    global_step: int,
    total_steps: int,
    weight_step: int,
    r2r_total_steps: int | None,
    message: str,
) -> None:
    """Invalid DataFetch/R2R relationships fail before rollout generation."""
    controller_type, data_fetch_type, r2r_type = _install_fake_cosmos_modules(
        monkeypatch
    )
    install_colocated_resume_bootstrap_bridge()
    controller = controller_type(
        data_fetch_command=data_fetch_type(
            global_step=global_step,
            total_steps=total_steps,
        ),
        r2r_command=r2r_type(
            weight_step=weight_step,
            total_steps=r2r_total_steps,
        ),
    )

    with pytest.raises(RuntimeError, match=message):
        controller.init_commands()


@pytest.mark.parametrize(
    ("mode", "expected_install_count"),
    [
        (CosmosRLMode.colocated, 1),
        (CosmosRLMode.disaggregated, 0),
    ],
)
def test_entrypoint_installs_bridge_only_for_colocated_policy_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: CosmosRLMode,
    expected_install_count: int,
) -> None:
    """Only the process that owns the local colocated controller installs the bridge."""
    entrypoint = importlib.import_module("alpagym_runtime.cosmos.entrypoint")
    cosmos_config_path = tmp_path / "cosmos.toml"
    cosmos_config_path.write_text(
        "[custom]\n"
        f"resolved_config_path = {json.dumps(str(tmp_path / 'resolved.yaml'))}\n",
        encoding="utf-8",
    )
    run_config = SimpleNamespace(
        logging_level="DEBUG",
        cosmos=SimpleNamespace(mode=mode),
        policy=SimpleNamespace(model=SimpleNamespace(kind="fake")),
    )
    install_calls: list[None] = []
    data_packer = SimpleNamespace(close=lambda: None)
    policy_bundle = SimpleNamespace(
        build_data_packer=lambda config, cosmos_role: data_packer,
    )
    monkeypatch.setenv("COSMOS_ROLE", "Policy")
    monkeypatch.setattr(entrypoint, "load_run_config", lambda path: run_config)
    monkeypatch.setattr(
        entrypoint,
        "install_colocated_resume_bootstrap_bridge",
        lambda: install_calls.append(None),
    )
    monkeypatch.setattr(
        entrypoint, "get_policy_bundle", lambda model_kind: policy_bundle
    )
    monkeypatch.setattr(entrypoint, "launch_worker", lambda **kwargs: None)

    entrypoint.main(["--config", str(cosmos_config_path)])

    assert len(install_calls) == expected_install_count
