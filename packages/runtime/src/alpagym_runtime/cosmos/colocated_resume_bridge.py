# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bootstrap Cosmos colocated rollout metadata from the resumed policy step."""

from typing import Any


def _required_int(value: object, *, field: str) -> int:
    """Return an integer protocol field or reject the command."""
    if type(value) is not int:
        raise RuntimeError(
            f"Cosmos colocated bootstrap requires integer {field}, got {value!r}"
        )
    return value


def install_colocated_resume_bootstrap_bridge() -> None:
    """Install the step-exact bootstrap used by a colocated Policy process.

    Cosmos restores checkpoint ``N`` before sending the first
    ``DataFetch(global_step=N+1)`` command. The colocated controller otherwise
    keeps its setup-time step at zero and labels the restored policy as behavior
    version zero. The dispatcher's registration-time R2R command carries the
    upcoming optimizer step ``N+1``, even though the transferred parameters are
    behavior version ``N``. This bridge verifies that control-plane convention,
    derives the live behavior version, and then delegates synchronization to
    Cosmos with the corrected local step.
    """
    from cosmos_rl.colocated.controller import ColocatedController
    from cosmos_rl.dispatcher.command import (
        DataFetchCommand,
        RolloutToRolloutBroadcastCommand,
    )

    original_init_commands = ColocatedController.init_commands
    if getattr(
        original_init_commands,
        "_alpagym_colocated_resume_bootstrap_bridge",
        False,
    ):
        return

    def init_commands_with_resume_bootstrap(self: Any) -> None:
        """Initialize Cosmos after reconciling DataFetch and R2R step metadata."""
        data_fetch_command = self.policy_consume_one_step_commands_util_data_fetch()
        if not isinstance(data_fetch_command, DataFetchCommand):
            raise RuntimeError(
                "Cosmos colocated bootstrap expected an initial DataFetchCommand, "
                f"got {type(data_fetch_command).__name__}"
            )

        global_step = _required_int(
            data_fetch_command.global_step,
            field="DataFetch.global_step",
        )
        total_steps = _required_int(
            data_fetch_command.total_steps,
            field="DataFetch.total_steps",
        )
        if global_step < 1:
            raise RuntimeError(
                "Cosmos colocated bootstrap requires DataFetch.global_step >= 1, "
                f"got {global_step}"
            )
        if total_steps < global_step:
            raise RuntimeError(
                "Cosmos colocated bootstrap requires DataFetch.total_steps >= "
                f"DataFetch.global_step, got {total_steps} < {global_step}"
            )

        behavior_step = global_step - 1
        local_step = _required_int(
            self.current_step,
            field="ColocatedController.current_step",
        )
        if local_step not in (0, behavior_step):
            raise RuntimeError(
                "Cosmos colocated bootstrap found inconsistent local and remote "
                f"steps: local={local_step}, expected 0 or {behavior_step}"
            )

        self.current_step = behavior_step
        self.total_steps = total_steps

        original_wait_for_remote_command = self.wait_for_remote_command
        had_instance_wait_override = "wait_for_remote_command" in self.__dict__
        instance_wait_override = self.__dict__.get("wait_for_remote_command")
        saw_initial_r2r = False

        def wait_for_remote_command_with_bootstrap_check(
            replica: Any,
            *args: object,
            **kwargs: object,
        ) -> Any:
            """Validate the initial remote R2R before Cosmos re-emits it locally."""
            nonlocal saw_initial_r2r
            command = original_wait_for_remote_command(replica, *args, **kwargs)
            if not isinstance(command, RolloutToRolloutBroadcastCommand):
                return command

            weight_step = _required_int(
                command.weight_step,
                field="initial R2R.weight_step",
            )
            if weight_step != global_step:
                raise RuntimeError(
                    "Cosmos colocated bootstrap step mismatch: "
                    f"DataFetch upcoming step is {global_step}, "
                    f"registration R2R carries {weight_step}"
                )
            if command.total_steps is not None:
                r2r_total_steps = _required_int(
                    command.total_steps,
                    field="initial R2R.total_steps",
                )
                if r2r_total_steps != total_steps:
                    raise RuntimeError(
                        "Cosmos colocated bootstrap total-step mismatch: "
                        f"DataFetch carries {total_steps}, initial R2R carries "
                        f"{r2r_total_steps}"
                    )
            saw_initial_r2r = True
            return command

        self.wait_for_remote_command = wait_for_remote_command_with_bootstrap_check
        try:
            synchronized = self.rollout_consume_one_step_commands_util_r2r()
        finally:
            if had_instance_wait_override:
                self.wait_for_remote_command = instance_wait_override
            else:
                del self.__dict__["wait_for_remote_command"]

        if synchronized is not True or not saw_initial_r2r:
            raise RuntimeError(
                "Cosmos colocated bootstrap requires one initial R2R weight sync"
            )
        if self.current_step != behavior_step or self.total_steps != total_steps:
            raise RuntimeError(
                "Cosmos colocated controller changed bootstrap step metadata during "
                "initial R2R synchronization"
            )
        self.init_data_fetch_command = data_fetch_command

    init_commands_with_resume_bootstrap._alpagym_colocated_resume_bootstrap_bridge = (  # type: ignore[attr-defined]
        True
    )
    ColocatedController.init_commands = init_commands_with_resume_bootstrap
