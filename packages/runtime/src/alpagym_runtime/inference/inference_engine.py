# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-session batched dispatcher into an ``InferenceModel``.

Producers (per-session policy threads) call ``InferenceEngine.infer`` and
block on the returned ``concurrent.futures.Future``. A single worker thread
runs ``InferenceEngine.run_loop``, which blocks for the first queued request,
opportunistically drains additional requests up to ``max_batch_size``, runs
one batched forward pass, and resolves each future with its per-call
``ModelOutput``.

Forward-pass exceptions are unrecoverable: under streaming dispatch the
engine thread is long-lived and shared across overlapping rollouts, so
``run_loop`` logs the traceback and exits the process via ``os._exit(1)``
instead of attempting soft recovery. Cosmos-RL's controller detects the
dead rollout replica through heartbeat timeout and triggers mesh rebuild
on the surviving replicas.
"""

import logging
import os
import queue
import sys
import threading
from copy import deepcopy
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Final

import torch
from alpagym_host.config import SamplingParamsConfig

from alpagym_runtime.perf.instrument.scope import timed_scope
from alpagym_runtime.replay import ActionSelection, PolicyReplayData

from .types import (
    BatchedModelInput,
    BatchedModelOutput,
    InferenceModel,
    ModelInput,
    ModelOutput,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _PendingRequest:
    """One queued inference call: input plus the future to resolve."""

    model_input: ModelInput
    result: Future[ModelOutput]


class _Shutdown:
    """Sentinel posted by ``shutdown()`` to terminate ``run_loop``."""


_SHUTDOWN: Final[_Shutdown] = _Shutdown()


@dataclass(frozen=True, slots=True)
class InferenceModelLease:
    """One immutable rollout-model snapshot bound to a behavior version."""

    behavior_policy_version: int
    model: torch.nn.Module


class InferenceEngine:
    """Queue-based batched dispatcher into one `InferenceModel`."""

    def __init__(
        self,
        inference_model: InferenceModel,
        sampling: SamplingParamsConfig,
        return_trace_for_rl: bool,
        max_batch_size: int,
        *,
        require_session_model_leases: bool = False,
    ) -> None:
        """Wire the dispatcher with its model and batching settings."""
        self._inference_model = inference_model
        self._sampling_config = sampling
        self._return_trace_for_rl = return_trace_for_rl
        self._max_batch_size = max_batch_size
        self._require_session_model_leases = bool(require_session_model_leases)
        self._model_lock = threading.RLock()
        self._session_model_leases: dict[str, InferenceModelLease] = {}
        self._queue: queue.SimpleQueue[_PendingRequest | _Shutdown] = (
            queue.SimpleQueue()
        )

    def infer(self, model_input: ModelInput) -> Future[ModelOutput]:
        """Enqueue one `ModelInput` and return its result handle."""
        future: Future[ModelOutput] = Future()
        self._queue.put(_PendingRequest(model_input=model_input, result=future))
        return future

    def build_policy_replay_data(
        self,
        model_input: ModelInput,
        model_output: ModelOutput,
        action_selection: ActionSelection,
    ) -> PolicyReplayData:
        """Return the selected-action ``PolicyReplayData`` envelope for trainer-side scoring."""
        return self._inference_model.build_policy_replay_data(
            model_input=model_input,
            model_output=model_output,
            action_selection=action_selection,
        )

    def get_model(self) -> torch.nn.Module:
        """Return the rollout-serving model object."""
        with self._model_lock:
            return self._inference_model.get_model()

    @property
    def requires_session_model_leases(self) -> bool:
        """Whether every humanoid session must resolve an immutable model lease."""

        return self._require_session_model_leases

    def create_model_lease(self, behavior_policy_version: int) -> InferenceModelLease:
        """Clone the live model into a read-only snapshot for one rollout version.

        The disaggregated rollout backend calls this only after Cosmos has made
        ``current_weight_version`` live.  Later R2R writes target the live model,
        while policies holding this independent module keep scoring and sampling
        from the exact weights that opened their episode.  Large multimodal models
        may implement ``clone_for_inference_lease()`` to share immutable frozen
        storage while cloning only their mutable actor overlay; models without that
        hook retain the generic deep-copy behavior.
        """

        if (
            isinstance(behavior_policy_version, bool)
            or not isinstance(behavior_policy_version, int)
            or behavior_policy_version < 0
        ):
            raise ValueError("model lease requires a non-negative behavior version")
        with self._model_lock:
            live_model = self._inference_model.get_model()
            clone_for_lease = getattr(live_model, "clone_for_inference_lease", None)
            snapshot = (
                clone_for_lease() if callable(clone_for_lease) else deepcopy(live_model)
            )
        if not isinstance(snapshot, torch.nn.Module):
            raise TypeError(
                "model clone_for_inference_lease() must return a torch.nn.Module"
            )
        if snapshot is live_model:
            raise ValueError(
                "model clone_for_inference_lease() must return an independent module"
            )
        snapshot.eval()
        snapshot.requires_grad_(False)
        return InferenceModelLease(
            behavior_policy_version=behavior_policy_version,
            model=snapshot,
        )

    def register_session_model_lease(
        self,
        session_uuid: str,
        lease: InferenceModelLease,
    ) -> None:
        """Bind one session UUID to its already-snapshotted behavior model."""

        if not session_uuid:
            raise ValueError("model lease registration requires a session UUID")
        if not isinstance(lease, InferenceModelLease):
            raise TypeError("session model lease has an unexpected type")
        with self._model_lock:
            if session_uuid in self._session_model_leases:
                raise ValueError(f"session {session_uuid!r} already has a model lease")
            self._session_model_leases[session_uuid] = lease

    def release_session_model_lease(self, session_uuid: str) -> None:
        """Release the immutable model snapshot after a session closes or fails."""

        with self._model_lock:
            try:
                self._session_model_leases.pop(session_uuid)
            except KeyError as exc:
                raise ValueError(
                    f"session {session_uuid!r} has no model lease to release"
                ) from exc

    def get_model_for_session(self, session_uuid: str) -> torch.nn.Module:
        """Return a session snapshot, or the live model in colocated mode."""

        if not session_uuid:
            raise ValueError("session model lookup requires a session UUID")
        with self._model_lock:
            lease = self._session_model_leases.get(session_uuid)
            if lease is not None:
                return lease.model
            if self._require_session_model_leases:
                raise RuntimeError(
                    f"session {session_uuid!r} has no immutable model lease"
                )
            return self._inference_model.get_model()

    def set_model(self, model: torch.nn.Module) -> None:
        """Forward Cosmos weight-sync replacement into the rollout-serving model."""
        with self._model_lock:
            self._inference_model.set_model(model)

    def run_loop(self) -> None:
        """Drain the queue and dispatch batches until shutdown drains it."""
        while True:
            # --- Phase 1: collect a batch ---
            # Block for the first item; opportunistically drain the rest.
            first = self._queue.get()
            if isinstance(first, _Shutdown):
                return
            batch: list[_PendingRequest] = [first]
            saw_shutdown = False
            while len(batch) < self._max_batch_size:
                try:
                    item = self._queue.get_nowait()
                except queue.Empty:
                    break
                if isinstance(item, _Shutdown):
                    saw_shutdown = True
                    break
                batch.append(item)

            # --- Phase 2: dispatch through the model ---
            # Forward-pass exceptions kill the process; see module docstring.
            try:
                model_inputs = [req.model_input for req in batch]
                batched_model_input = BatchedModelInput.stack(model_inputs)
                with timed_scope(
                    "rollout/inference_forward",
                    category="compute_gpu_wall",
                    gpu_snapshot=True,
                ):
                    batched_model_output: BatchedModelOutput = (
                        self._inference_model.sample_trajectories_from_data(
                            batched_model_input,
                            self._sampling_config,
                            return_trace_for_rl=self._return_trace_for_rl,
                        )
                    )
                model_outputs = batched_model_output.unbind()
                if len(model_outputs) != len(model_inputs):
                    raise ValueError(
                        f"InferenceEngine: {len(model_outputs)} ModelOutput(s) returned for "
                        f"{len(model_inputs)} ModelInput(s)"
                    )
            except BaseException:
                logger.exception("InferenceEngine forward pass failed; killing process")
                from alpagym_runtime.perf.instrument.store import try_get_perf_store

                store = try_get_perf_store()
                if store is not None:
                    try:
                        store.write_atomic()
                    except Exception:
                        logger.exception(
                            "InferenceEngine failed to flush perf artifact"
                        )
                sys.stderr.flush()
                os._exit(1)

            # --- Phase 3: resolve futures and honour shutdown ---
            for req, model_output in zip(batch, model_outputs, strict=True):
                req.result.set_result(model_output)
            if saw_shutdown:
                return

    def shutdown(self) -> None:
        """Signal `run_loop` to return once the queue drains."""
        self._queue.put(_SHUTDOWN)
