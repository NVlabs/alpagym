# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import sys
import types

import pytest


def test_fresh_colocated_rollout_uses_dummy_load_and_preserves_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command_module = types.ModuleType("cosmos_rl.dispatcher.command")
    rollout_module = types.ModuleType(
        "cosmos_rl.rollout.worker.colocated.rollout_control"
    )

    class PolicyToRolloutUnicastCommand:
        dst_replica_name = "rollout-0"

    class ColocatedRolloutControlWorker:
        registry = {}

        @classmethod
        def register_rollout_command_handler(cls, command_type):
            def register(handler):
                cls.registry[command_type] = handler
                return handler

            return register

        def lazy_initialize_rollout_engine(self, load_format):
            self.load_formats.append(load_format)

        def policy_to_rollout_unicast(self, command):
            self.lazy_initialize_rollout_engine("auto")
            return command

    class FakeRollout:
        def set_underlying_model(self, model):
            self.model = model

    class FakeApiClient:
        def get_policy_model(self):
            return policy_model

    command_module.PolicyToRolloutUnicastCommand = PolicyToRolloutUnicastCommand
    rollout_module.ColocatedRolloutControlWorker = ColocatedRolloutControlWorker
    monkeypatch.setitem(sys.modules, "cosmos_rl.dispatcher.command", command_module)
    monkeypatch.setitem(
        sys.modules,
        "cosmos_rl.rollout.worker.colocated.rollout_control",
        rollout_module,
    )

    bridge = importlib.import_module(
        "alpagym_runtime.cosmos.colocated_fresh_rollout_bridge"
    )
    bridge.install_colocated_fresh_rollout_bridge()
    installed_handler = ColocatedRolloutControlWorker.policy_to_rollout_unicast
    bridge.install_colocated_fresh_rollout_bridge()

    worker = ColocatedRolloutControlWorker()
    worker.replica_name = "rollout-0"
    worker.rollout = FakeRollout()
    policy_model = object()
    worker.api_client = FakeApiClient()
    worker.load_formats = []
    command = PolicyToRolloutUnicastCommand()

    assert worker.policy_to_rollout_unicast(command) is command
    assert worker.load_formats == ["dummy"]
    assert worker.rollout.model is policy_model
    assert "lazy_initialize_rollout_engine" not in worker.__dict__
    assert ColocatedRolloutControlWorker.policy_to_rollout_unicast is installed_handler
    assert (
        ColocatedRolloutControlWorker.registry[PolicyToRolloutUnicastCommand]
        is installed_handler
    )
