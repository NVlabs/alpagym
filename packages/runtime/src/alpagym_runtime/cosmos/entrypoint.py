# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import atexit
import logging
import os
import tomllib
from pathlib import Path
from typing import Any

import yaml
from alpagym_host.config import CosmosRLMode, TransportKind, load_run_config
from cosmos_rl.launcher.worker_entry import main as launch_worker
from torch.distributed.elastic.multiprocessing.errors import record

# Imports that register model, backend and trainer classes with the Cosmos registry.
from alpagym_runtime.cosmos import (
    rollout_backend as _rollout_backend,  # noqa: F401
    trainer as _trainer,  # noqa: F401
)
from alpagym_runtime.cosmos.colocated_fresh_rollout_bridge import (
    install_colocated_fresh_rollout_bridge,
)
from alpagym_runtime.cosmos.colocated_resume_bridge import (
    install_colocated_resume_bootstrap_bridge,
)
from alpagym_runtime.cosmos.dataset import AlpagymSceneDataset, repeat_scene_ids
from alpagym_runtime.cosmos.nccl_cleanup_hooks import (
    install_cosmos_nccl_cleanup_publisher_opt_in,
)
from alpagym_runtime.cosmos.nccl_store import start_nccl_store_master
from alpagym_runtime.cosmos.reward_fn import episode_reward_from_artifact
from alpagym_runtime.policies.registry import get_policy_bundle

# Controller-owned NCCL TCPStore master, kept alive for the whole job. Policy
# and Rollout workers rendezvous through it; dropping the reference at process
# exit tears it down (TCPStore has no close()).
_nccl_store_master = None


def _configure_logging(level: str) -> None:
    """Configure process-wide logging for Cosmos worker processes."""
    logging.basicConfig(
        level=logging.getLevelNamesMapping()[level],
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )


def _build_dataset(config: Any) -> AlpagymSceneDataset:
    """Build the scene dataset from the resolved AlpaGym config."""
    run_config = load_run_config(Path(config.custom["resolved_config_path"]))
    scene_id_data: dict[str, list[str]] = yaml.safe_load(
        run_config.artifact_paths.alpasim_scene_ids_path.read_text(encoding="utf-8")
    )
    scene_ids = [str(scene_id) for scene_id in scene_id_data["scene_ids"]]
    return AlpagymSceneDataset(
        scene_ids=repeat_scene_ids(scene_ids, run_config.dataset.scene_repetitions)
    )


def _install_policy_bundle_runtime_hooks(policy_bundle: Any, run_config: Any) -> None:
    """Install policy-owned Cosmos hooks before Cosmos constructs workers."""
    install_bridge = getattr(policy_bundle, "install_runtime_bridge", None)
    if callable(install_bridge):
        install_bridge()

    setup_bundle_tokenizer = getattr(policy_bundle, "setup_tokenizer", None)
    if not callable(setup_bundle_tokenizer):
        return

    from cosmos_rl.utils import util as cosmos_util

    original_setup_tokenizer = cosmos_util.setup_tokenizer

    def setup_tokenizer_with_policy_bundle(model_name_or_path: str) -> Any:
        bundle_tokenizer = setup_bundle_tokenizer(run_config)
        if bundle_tokenizer is not None:
            return bundle_tokenizer
        return original_setup_tokenizer(model_name_or_path)

    cosmos_util.setup_tokenizer = setup_tokenizer_with_policy_bundle

    try:
        from cosmos_rl.policy.trainer.llm_trainer import grpo_trainer
    except Exception:
        return
    grpo_trainer.setup_tokenizer = setup_tokenizer_with_policy_bundle


def _install_alpagym_rollout_teardown() -> None:
    """Close AlpaGym rollout resources before Cosmos destroys torch distributed."""
    from cosmos_rl.rollout.worker.llm_worker import LLMRolloutWorker
    from cosmos_rl.rollout.worker.rollout_control import (
        DisaggregatedRolloutControlWorker,
    )

    original_destroy_worker = LLMRolloutWorker.destroy_worker

    def join_shutdown_thread(worker: Any, thread_attr: str) -> None:
        """Join one Cosmos shutdown thread without blocking process teardown."""
        thread = getattr(worker, thread_attr)
        if thread is None:
            return
        thread.join(timeout=5.0)
        if thread.is_alive():
            logging.warning(
                "Cosmos rollout shutdown thread %s did not exit within 5 seconds",
                thread_attr,
            )
        setattr(worker, thread_attr, None)

    def handle_shutdown_with_bounded_joins(self: Any) -> None:
        """Stop Cosmos rollout background threads without unbounded joins."""
        if hasattr(self, "_shutdown_handled"):
            return
        self._shutdown_handled = True
        if not self.shutdown_signal.is_set():
            logging.info(
                "[Rollout] shutdown instruction of %s, setting shutdown signal",
                self.replica_name,
            )
            self.shutdown_signal.set()
        if not self.shutdown_mp_signal.is_set():
            self.shutdown_mp_signal.set()

        join_shutdown_thread(self, "background_thread")
        join_shutdown_thread(self, "teacher_interact_thread")
        if self.scheduler is not None:
            self.scheduler.stop(wait=False)
            self.scheduler = None
        join_shutdown_thread(self, "heartbeat_thread")
        self.unregister_from_controller()

    if not getattr(
        DisaggregatedRolloutControlWorker.handle_shutdown,
        "_alpagym_bounded_shutdown_wrap",
        False,
    ):
        handle_shutdown_with_bounded_joins._alpagym_bounded_shutdown_wrap = True  # type: ignore[attr-defined]
        DisaggregatedRolloutControlWorker.handle_shutdown = (
            handle_shutdown_with_bounded_joins
        )

    if getattr(original_destroy_worker, "_alpagym_teardown_wrap", False):
        return

    def destroy_worker_with_alpagym_teardown(self: Any) -> None:
        """Run AlpaGym cleanup before the upstream torch-distributed teardown."""
        try:
            rollout_worker = self.rollout_worker
            if rollout_worker is not None:
                rollout_worker.rollout.shutdown()
                rollout_worker.data_packer.close()
        finally:
            original_destroy_worker(self)

    destroy_worker_with_alpagym_teardown._alpagym_teardown_wrap = True  # type: ignore[attr-defined]
    LLMRolloutWorker.destroy_worker = destroy_worker_with_alpagym_teardown


def _enable_alpagym_async_rollout() -> None:
    """Declare the AlpaGym backend compatible with Cosmos's async scheduler."""
    from cosmos_rl.rollout.worker.rollout_control import (
        DisaggregatedRolloutControlWorker,
    )

    if "alpagym_rollout" not in DisaggregatedRolloutControlWorker.SUPPORT_ASYNC_BACKEND:
        DisaggregatedRolloutControlWorker.SUPPORT_ASYNC_BACKEND.append(
            "alpagym_rollout"
        )

    original_stream_generation_step = (
        DisaggregatedRolloutControlWorker.stream_generation_step
    )
    if getattr(
        original_stream_generation_step,
        "_alpagym_idempotent_prompt_end_wrap",
        False,
    ):
        return

    def stream_generation_step_with_idempotent_prompt_end(self: Any) -> None:
        """Keep drained async replicas alive while they await controller stop."""
        if self.state.prompt_consume_end():
            if self.should_report:
                self.send_end_signal()
            return
        original_stream_generation_step(self)

    stream_generation_step_with_idempotent_prompt_end._alpagym_idempotent_prompt_end_wrap = True  # type: ignore[attr-defined]
    DisaggregatedRolloutControlWorker.stream_generation_step = (
        stream_generation_step_with_idempotent_prompt_end
    )


# `@record` installs an excepthook that dumps uncaught exceptions (with full
# traceback) to `$TORCHELASTIC_ERROR_FILE`, which torchrun sets per-child in
# `cosmos_rl.launcher.utility.launch_processes` for Policy/Rollout workers.
# Without it, torchrun's `ChildFailedError` reports `error_file: <N/A>` and the
# real traceback only appears via the default excepthook in the per-worker log,
# making root-cause analysis harder. The decorator is a no-op when the env var
# is unset (e.g. when this entrypoint is invoked directly by the Controller),
# so it is safe to apply unconditionally. See
# https://pytorch.org/docs/stable/elastic/errors.html
@record
def main(argv: list[str] | None = None) -> None:
    """Launch the Cosmos worker entrypoint with the configured packer + reward.

    Cosmos passes ``--config <toml_path>`` to the entrypoint; its
    ``[custom].resolved_config_path`` points at the resolved host config so
    role-specific setup can run before Cosmos constructs registered components.
    The data packer owns this process's transport endpoint; the reward callback
    reads ``EpisodeOutput.reward.total`` off the in-memory completion.

    Args:
        argv: Optional command-line arguments. Cosmos passes its own arguments
            to this script; only ``--config`` is read here, the rest is
            ignored locally and re-parsed by Cosmos.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args, _ = parser.parse_known_args(argv)
    cosmos_config = tomllib.loads(Path(args.config).read_text(encoding="utf-8"))
    run_config = load_run_config(Path(cosmos_config["custom"]["resolved_config_path"]))
    _configure_logging(str(run_config.logging_level))

    cosmos_role = os.environ.get("COSMOS_ROLE")

    if cosmos_role == "Policy" and run_config.cosmos.mode == CosmosRLMode.colocated:
        install_colocated_resume_bootstrap_bridge()
        if not run_config.cosmos.train.resume.enabled:
            install_colocated_fresh_rollout_bridge()

    # The Controller owns no data plane, but on NCCL it starts the TCPStore
    # master that Policy/Rollout workers rendezvous through and installs the
    # buffer-clear cleanup publisher. Workers build their endpoint inside the packer.
    if cosmos_role == "Controller" and run_config.transport.kind == TransportKind.nccl:
        global _nccl_store_master
        install_cosmos_nccl_cleanup_publisher_opt_in()
        _nccl_store_master = start_nccl_store_master(run_config)
    elif cosmos_role == "Rollout":
        _install_alpagym_rollout_teardown()
        if run_config.cosmos.rollout.mode == "async":
            _enable_alpagym_async_rollout()

    policy_bundle = get_policy_bundle(run_config.policy.model.kind)
    _install_policy_bundle_runtime_hooks(policy_bundle, run_config)
    data_packer = policy_bundle.build_data_packer(run_config, cosmos_role)
    # Registered before launch_worker so close runs after the rollout backend's
    # own atexit shutdown (LIFO): the worker stops producing before the writer closes.
    atexit.register(data_packer.close)

    launch_worker(
        dataset=_build_dataset,
        data_packer=data_packer,
        reward_fns=[episode_reward_from_artifact],
    )


if __name__ == "__main__":
    main()
