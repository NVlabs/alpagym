# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402

"""Streaming AlpaSim rollout worker with per-rollout retry and queue-driven dispatch."""

import itertools
import logging
import queue
import threading
import uuid
from concurrent.futures import Future
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping

import torch
from alpagym_runtime.alpasim.grpc_import import ensure_alpasim_grpc_source

ensure_alpasim_grpc_source()
from alpagym_host.config import RewardConfig
from alpasim_grpc.v0.runtime_pb2_grpc import RuntimeServiceStub
from cosmos_rl.dispatcher.data.schema import RLPayload

from alpagym_runtime.alpasim.driver_server import EgodriverServer
from alpagym_runtime.alpasim.humanoid_policy_server import HumanoidPolicyServer
from alpagym_runtime.alpasim.proto_conversion import build_simulation_request_proto
from alpagym_runtime.perf.instrument.scope import measure_perf, timed_scope
from alpagym_runtime.rewards.compute import compute_reward
from alpagym_runtime.types import EpisodeMetrics, EpisodeOutput, PolicyOutput, RewardResult

logger = logging.getLogger(__name__)


@dataclass(eq=False)
class SharedPayloadState:
    """Per-payload accounting shared by reference across `n_target` sibling rollouts.

    Lock discipline: every field below is read/written under the owning worker's
    `_lock`. The single exception is `future.set_result(...)`, called outside the
    lock by the thread that flipped `future_resolved` from False to True.
    """

    payload: RLPayload
    n_target: int
    future: Future[list[EpisodeOutput]]
    retries_left: int
    collected: list[EpisodeOutput] = field(default_factory=list)
    permanently_failed: bool = False
    future_resolved: bool = False


@dataclass
class _RolloutJob:
    """One simulate(1) work item pulled from the rollout-job queue."""

    shared_payload_state: SharedPayloadState
    session_uuid: str
    scene_id: str
    attempts: int = 0


def _dense_metrics_from_rollout_return(rollout_return: object) -> dict[str, Any]:
    """Convert RuntimeService timestep metrics into an artifact-friendly mapping."""
    dense: dict[str, Any] = {}
    for metric in getattr(rollout_return, "timestep_metrics", []):
        dense[str(metric.name)] = {
            "timestamps_us": [int(value) for value in metric.timestamps_us],
            "values": [float(value) for value in metric.values],
            "valid": [bool(value) for value in metric.valid],
        }
    return dense


def _metric_values_by_name(rollout_return: object) -> dict[str, list[float]]:
    return {
        str(metric.name): [float(value) for value in metric.values]
        for metric in getattr(rollout_return, "timestep_metrics", [])
    }


def _payload_humanoid_meta(output: PolicyOutput) -> Mapping[str, Any]:
    replay_data = output.replay_data
    if replay_data is None:
        return {}
    meta = replay_data.payload.get("humanoid", {})
    return meta if isinstance(meta, Mapping) else {}


def _output_env_id(output: PolicyOutput) -> int:
    if output.model_extra and "humanoid_env_id" in output.model_extra:
        return int(output.model_extra["humanoid_env_id"])
    meta = _payload_humanoid_meta(output)
    return int(meta.get("env_id", 0))


def _output_step_index(output: PolicyOutput, fallback: int) -> int:
    if output.model_extra and "humanoid_step_index" in output.model_extra:
        return int(output.model_extra["humanoid_step_index"])
    meta = _payload_humanoid_meta(output)
    return int(meta.get("step_index", fallback))


def _output_value(output: PolicyOutput) -> float:
    if output.model_extra and "humanoid_value" in output.model_extra:
        return float(output.model_extra["humanoid_value"])
    meta = _payload_humanoid_meta(output)
    if "value" in meta:
        value = meta["value"]
        if isinstance(value, torch.Tensor):
            return float(value.reshape(()).item())
        return float(value)
    return 0.0


def _metric_value(
    metrics: Mapping[str, list[float]],
    name: str,
    step_index: int,
    default: float,
) -> float:
    values = metrics.get(name)
    if values is None or step_index < 0 or step_index >= len(values):
        return default
    return float(values[step_index])


def _attach_humanoid_transition_payloads(
    outputs: tuple[PolicyOutput, ...],
    rollout_return: object,
) -> tuple[PolicyOutput, ...]:
    """Attach PPO transition facts from AlpaSim metrics to humanoid replay rows."""
    metrics = _metric_values_by_name(rollout_return)
    next_step_by_env: dict[int, int] = {}
    patched_outputs: list[PolicyOutput] = []

    for output in outputs:
        replay_data = output.replay_data
        if replay_data is None:
            patched_outputs.append(output)
            continue

        env_id = _output_env_id(output)
        fallback_step_index = next_step_by_env.get(env_id, 0)
        step_index = _output_step_index(output, fallback=fallback_step_index)
        next_step_by_env[env_id] = max(fallback_step_index, step_index) + 1
        env_suffix = f"env{env_id}"

        reward = _metric_value(metrics, f"humanoid_reward_{env_suffix}", step_index, 0.0)
        terminated = bool(
            _metric_value(metrics, f"humanoid_terminated_{env_suffix}", step_index, 0.0)
        )
        truncated = bool(
            _metric_value(metrics, f"humanoid_truncated_{env_suffix}", step_index, 0.0)
        )

        payload = dict(replay_data.payload)
        transition = dict(payload.get("transition", {}))
        transition.setdefault("reward", reward)
        transition.setdefault("terminated", terminated)
        transition.setdefault("truncated", truncated)
        transition.setdefault("old_value", _output_value(output))
        # A value-only final-state query is not part of the service contract yet.
        # Use zero bootstrap for now; the policy/value at each sampled state is
        # still trained from the rollout values and realized rewards.
        transition.setdefault("bootstrap_value", 0.0)
        payload["transition"] = transition
        patched_outputs.append(
            replace(output, replay_data=replace(replay_data, payload=payload))
        )

    return tuple(patched_outputs)


class StreamingRolloutWorker:
    """Per-rollout dispatcher driving one AlpaSim runtime endpoint.

    `submit_payload(payload)` is idempotent by `prompt_idx`: a fresh call
    inserts a `SharedPayloadState` into `_active_payload_states` and
    enqueues `rollouts_per_payload` sibling `_RolloutJob`s; a repeat call
    returns the existing state. The state is dropped from the index when
    its future resolves. `max_concurrent_rollouts` worker threads consume
    jobs from a priority queue and run `simulate(1)` per job; collected
    episodes resolve the future once `n_target` siblings have landed. A
    failing job is re-enqueued with a fresh `session_uuid` and higher
    priority than fresh dispatch -- a retry runs before any payload that
    hasn't started yet, so a slow scrap-and-restart doesn't push tail
    latency past one extra `simulation_timeout_s` per failure. Retries
    continue until the payload's `max_scene_retries` budget is exhausted;
    on exhaustion the future resolves with `[]` and pending siblings are
    dropped.
    """

    def __init__(
        self,
        *,
        alpasim_runtime_stub: RuntimeServiceStub,
        driver_server: EgodriverServer | None,
        simulation_timeout_s: float,
        reward_config: RewardConfig,
        max_concurrent_rollouts: int,
        rollouts_per_payload: int,
        scene_id_resolver: Callable[[RLPayload], str],
        max_scene_retries: int = 3,
        humanoid_policy_server: HumanoidPolicyServer | None = None,
        simulation_domain: str = "av",
    ) -> None:
        """Wire the worker and start `max_concurrent_rollouts` simulate-pool threads."""
        if max_concurrent_rollouts < 1:
            raise ValueError("max_concurrent_rollouts must be at least 1")
        if rollouts_per_payload < 1:
            raise ValueError("rollouts_per_payload must be at least 1")
        if max_scene_retries < 0:
            raise ValueError("max_scene_retries must be non-negative")
        self._alpasim_runtime_stub = alpasim_runtime_stub
        self._simulation_domain = str(simulation_domain)
        self._driver_server = driver_server
        self._humanoid_policy_server = humanoid_policy_server
        if self._simulation_domain == "humanoid":
            if humanoid_policy_server is None:
                raise ValueError("humanoid simulation requires humanoid_policy_server")
            self._policy_endpoint = humanoid_policy_server.topology_endpoint
        elif self._simulation_domain == "av":
            if driver_server is None:
                raise ValueError("av simulation requires driver_server")
            self._policy_endpoint = driver_server.topology_endpoint
        else:
            raise ValueError(f"unsupported simulation_domain={simulation_domain!r}")
        self._simulation_timeout_s = simulation_timeout_s
        self._reward_config = reward_config
        self._rollouts_per_payload = rollouts_per_payload
        self._scene_id_resolver = scene_id_resolver
        self._max_scene_retries = max_scene_retries

        self._lock = threading.Lock()
        # Inserted on first submit per prompt_idx, removed on future resolution;
        # values() also defines the shutdown unresolved set. Cosmos-RL's
        # training and validation requests live in disjoint `prompt_idx`
        # spaces, so a validation payload never collides with an in-flight
        # training state -- it always takes the fresh-dispatch path.
        self._active_payload_states: dict[int, SharedPayloadState] = {}
        # Priority queue keyed by (priority, seq, rollout_job). Retries use
        # priority 0 so they preempt fresh dispatch (priority 1); shutdown
        # sentinels use priority 2 (only reached after the queue is drained).
        # `seq` is a monotonic tiebreak so equal-priority items pop FIFO.
        self._rollout_job_queue: queue.PriorityQueue[tuple[int, int, _RolloutJob | None]] = (
            queue.PriorityQueue()
        )
        self._rollout_job_seq = itertools.count()
        self._closed = False
        self._rollout_workers = [
            threading.Thread(
                target=self._rollout_worker_loop,
                name=f"alpagym-sim-{i}",
                daemon=True,
            )
            for i in range(max_concurrent_rollouts)
        ]
        for rollout_worker in self._rollout_workers:
            rollout_worker.start()

    # ---------- public dispatch surface ----------

    def submit_payload(self, payload: RLPayload) -> SharedPayloadState:
        """Return the running state for `payload`; dispatch at most once per `prompt_idx`.

        A repeat call with the same `prompt_idx` returns the state from
        the first call until that payload's future resolves; after
        resolution the next call dispatches afresh. Cosmos-RL's prefetch
        hook and `rollout_generation` both call this with the same
        payload, and the second call piggy-backs on the first via the
        live-state index. Training and validation payloads come from
        disjoint `prompt_idx` spaces, so they never collide.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError("StreamingRolloutWorker is shut down")

            # Return the existing state if the payload has already been submitted.
            if payload.prompt_idx in self._active_payload_states:
                return self._active_payload_states[payload.prompt_idx]

            # Otherwise, create a new state and enqueue the rollout jobs.
            scene_id = self._scene_id_resolver(payload)
            payload_state = SharedPayloadState(
                payload=payload,
                n_target=self._rollouts_per_payload,
                future=Future(),
                retries_left=self._max_scene_retries,
            )
            self._active_payload_states[payload.prompt_idx] = payload_state
            for _ in range(self._rollouts_per_payload):
                self._rollout_job_queue.put(
                    (
                        1,  # fresh-dispatch priority
                        next(self._rollout_job_seq),
                        _RolloutJob(
                            shared_payload_state=payload_state,
                            session_uuid=uuid.uuid4().hex,
                            scene_id=scene_id,
                        ),
                    )
                )
            return payload_state

    def shutdown(self) -> None:
        """Stop accepting payloads, resolve every unresolved future with `[]`, dismiss workers.

        Workers in mid-`simulate()` finish their RPC and exit on the next
        queue read; daemon threads do not block process exit.
        """
        with self._lock:
            self._closed = True
            unresolved = [
                payload_state
                for payload_state in self._active_payload_states.values()
                if not payload_state.future_resolved
            ]
            for payload_state in unresolved:
                payload_state.future_resolved = True
                payload_state.permanently_failed = True
            self._active_payload_states.clear()
        while True:
            try:
                self._rollout_job_queue.get_nowait()
            except queue.Empty:
                break
        for _ in self._rollout_workers:
            # priority 2 = shutdown sentinel; the lowest-priority.
            self._rollout_job_queue.put((2, next(self._rollout_job_seq), None))
        for payload_state in unresolved:
            payload_state.future.set_result([])

    # ---------- internal ----------

    def _rollout_worker_loop(self) -> None:
        """Pull jobs until a `None` sentinel; skip siblings of permanently-failed payloads."""
        while True:
            _, _, rollout_job = self._rollout_job_queue.get()
            if rollout_job is None:
                return
            if rollout_job.shared_payload_state.permanently_failed:
                continue
            self._run_rollout(rollout_job)

    @measure_perf("rollout/episode", category="orchestration", cpu_snapshot=True)
    def _run_rollout(self, rollout_job: _RolloutJob) -> None:
        """Run one simulate(1) end-to-end and finalize on success or failure."""
        try:
            with timed_scope("rollout/sim_request_build", category="orchestration"):
                request = self._build_simulation_request(rollout_job)
            with timed_scope("rollout/sim_step_rpc", category="external_rpc"):
                sim_return = self._alpasim_runtime_stub.simulate(
                    request,
                    timeout=self._simulation_timeout_s,
                )
            rollout_return = sim_return.rollout_returns[0]
            if not rollout_return.success:
                self._on_rollout_failed(
                    rollout_job,
                    RuntimeError(rollout_return.error or "AlpaSim rollout failed"),
                )
                return
            if self._simulation_domain == "humanoid":
                episode = self._build_humanoid_episode(rollout_job, rollout_return)
            else:
                episode = self._build_av_episode(rollout_job, rollout_return)
            self._on_rollout_succeeded(rollout_job, episode)
        except Exception as exc:
            self._on_rollout_failed(rollout_job, exc)

    def _build_simulation_request(self, rollout_job: _RolloutJob):
        """Build the RuntimeService request for this worker's simulation domain."""
        endpoint = self._policy_endpoint
        if self._simulation_domain == "humanoid":
            return build_simulation_request_proto(
                scene_ids=(rollout_job.scene_id,),
                n_generation=1,
                humanoid_policy_host=endpoint.host,
                humanoid_policy_port=int(endpoint.port),
                n_concurrent_per_humanoid_policy=1,
                session_uuid=rollout_job.session_uuid,
            )
        return build_simulation_request_proto(
            scene_ids=(rollout_job.scene_id,),
            n_generation=1,
            driver_host=endpoint.host,
            driver_port=int(endpoint.port),
            n_concurrent_per_driver=1,
            session_uuid=rollout_job.session_uuid,
        )

    def _build_av_episode(self, rollout_job: _RolloutJob, rollout_return: object) -> EpisodeOutput:
        """Build an EpisodeOutput for the existing AV egodriver path."""
        if self._driver_server is None:
            raise RuntimeError("AV rollout completed without a driver server")
        record = self._driver_server.servicer.pop_session_record(rollout_job.session_uuid)
        aggregated = dict(rollout_return.aggregated_metrics)
        base = EpisodeOutput(
            scene_id=rollout_job.scene_id,
            session_uuid=rollout_job.session_uuid,
            num_steps=len(record.outputs),
            policy_outputs=record.outputs,
            executed_ego_trajectory=record.executed_ego_trajectory,
            route_waypoints=(),
            metrics=EpisodeMetrics(aggregated=aggregated, dense={}) if aggregated else None,
            reward=None,
        )
        with timed_scope("rollout/reward_compute", category="compute_cpu", cpu_snapshot=True):
            reward = compute_reward(base, record.ground_truth, self._reward_config)
        return replace(base, reward=reward)

    def _build_humanoid_episode(
        self,
        rollout_job: _RolloutJob,
        rollout_return: object,
    ) -> EpisodeOutput:
        """Build an EpisodeOutput for humanoid PPO replay."""
        if self._humanoid_policy_server is None:
            raise RuntimeError("Humanoid rollout completed without a policy server")
        record = self._humanoid_policy_server.servicer.pop_session_record(
            rollout_job.session_uuid
        )
        aggregated = dict(rollout_return.aggregated_metrics)
        dense = _dense_metrics_from_rollout_return(rollout_return)
        outputs = _attach_humanoid_transition_payloads(record.outputs, rollout_return)
        reward_total = float(aggregated.get("humanoid_total_return", 0.0))
        return EpisodeOutput(
            scene_id=rollout_job.scene_id,
            session_uuid=rollout_job.session_uuid,
            num_steps=len(outputs),
            policy_outputs=outputs,
            metrics=EpisodeMetrics(aggregated=aggregated, dense=dense),
            reward=RewardResult(total=reward_total, report_metrics=aggregated),
        )

    def _on_rollout_succeeded(self, rollout_job: _RolloutJob, episode: EpisodeOutput) -> None:
        """Append the episode; resolve the future when `n_target` siblings land."""
        payload_state = rollout_job.shared_payload_state
        should_resolve = False
        result_payload: list[EpisodeOutput] = []
        with self._lock:
            if not payload_state.permanently_failed:
                payload_state.collected.append(episode)
                if (
                    len(payload_state.collected) >= payload_state.n_target
                    and not payload_state.future_resolved
                ):
                    payload_state.future_resolved = True
                    should_resolve = True
                    result_payload = list(payload_state.collected[: payload_state.n_target])
                    self._active_payload_states.pop(payload_state.payload.prompt_idx, None)
        if should_resolve:
            payload_state.future.set_result(result_payload)

    def _on_rollout_failed(self, rollout_job: _RolloutJob, exc: BaseException) -> None:
        """Retry the payload until the per-payload budget is exhausted; otherwise drain + drop."""
        payload_state = rollout_job.shared_payload_state
        should_drop = False
        with self._lock:
            if payload_state.permanently_failed:
                return
            payload_state.retries_left -= 1
            if payload_state.retries_left >= 0:
                self._rollout_job_queue.put(
                    (
                        0,  # retry priority: jumps ahead of fresh dispatch
                        next(self._rollout_job_seq),
                        _RolloutJob(
                            shared_payload_state=payload_state,
                            session_uuid=uuid.uuid4().hex,
                            scene_id=rollout_job.scene_id,
                            attempts=rollout_job.attempts + 1,
                        ),
                    )
                )
                logger.warning(
                    "Retrying payload uuid=%s scene=%s attempt=%d retries_left=%d: %s",
                    rollout_job.session_uuid,
                    rollout_job.scene_id,
                    rollout_job.attempts + 1,
                    payload_state.retries_left,
                    exc,
                )
            else:
                payload_state.permanently_failed = True
                if not payload_state.future_resolved:
                    payload_state.future_resolved = True
                    should_drop = True
                    self._active_payload_states.pop(payload_state.payload.prompt_idx, None)
                logger.error(
                    "Dropping payload after exhausted retries: scene=%s last_uuid=%s: %s",
                    rollout_job.scene_id,
                    rollout_job.session_uuid,
                    exc,
                )
        if should_drop:
            payload_state.future.set_result([])
