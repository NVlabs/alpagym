# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Avoid redundant pretrained-weight loading for fresh colocated rollouts."""

from typing import Any


def install_colocated_fresh_rollout_bridge() -> None:
    """Initialize the colocated rollout architecture without loading weights twice.

    A fresh colocated run replaces the rollout model with the live policy model
    immediately after lazy initialization. Cosmos nevertheless requests
    ``load_format="auto"``, which reloads the full pretrained checkpoint before
    discarding that model. Force only that initialization call to ``"dummy"``;
    checkpoint-resume runs do not install this bridge.
    """
    from cosmos_rl.dispatcher.command import PolicyToRolloutUnicastCommand
    from cosmos_rl.utils.logging import logger
    from cosmos_rl.rollout.worker.colocated.rollout_control import (
        ColocatedRolloutControlWorker,
    )

    original_handler = ColocatedRolloutControlWorker.policy_to_rollout_unicast
    if getattr(original_handler, "_alpagym_fresh_rollout_bridge", False):
        return

    def policy_to_rollout_with_dummy_initialization(
        self: Any,
        command: Any,
    ) -> Any:
        """Delegate to Cosmos while forcing its disposable model to dummy load."""
        if command.dst_replica_name == self.replica_name:
            self.rollout.set_underlying_model(self.api_client.get_policy_model())
            logger.info("[alpagym] Injected colocated policy model before rollout init")
        original_initialize = self.lazy_initialize_rollout_engine
        had_instance_override = "lazy_initialize_rollout_engine" in self.__dict__
        instance_override = self.__dict__.get("lazy_initialize_rollout_engine")

        def initialize_without_pretrained_weights(_load_format: str) -> Any:
            return original_initialize("dummy")

        self.lazy_initialize_rollout_engine = initialize_without_pretrained_weights
        try:
            result = original_handler(self, command)
            logger.info("[alpagym] Fresh colocated rollout initialization complete")
            return result
        finally:
            if had_instance_override:
                self.lazy_initialize_rollout_engine = instance_override
            else:
                del self.__dict__["lazy_initialize_rollout_engine"]

    policy_to_rollout_with_dummy_initialization._alpagym_fresh_rollout_bridge = (  # type: ignore[attr-defined]
        True
    )
    ColocatedRolloutControlWorker.policy_to_rollout_unicast = (
        policy_to_rollout_with_dummy_initialization
    )
    ColocatedRolloutControlWorker.register_rollout_command_handler(
        PolicyToRolloutUnicastCommand
    )(policy_to_rollout_with_dummy_initialization)


__all__ = ["install_colocated_fresh_rollout_bridge"]
