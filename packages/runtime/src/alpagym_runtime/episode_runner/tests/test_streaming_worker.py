# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the streaming AlpaSim rollout worker.

These contracts pin dispatch-state-machine behaviors of
`StreamingRolloutWorker`. Each test uses a `_StubWorker` subclass that
replaces `_run_rollout` with a deterministic per-uuid outcome map, so
no AlpaSim runtime, gRPC channel, disk I/O, or reward computation runs.
"""

import threading
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from alpagym_runtime.alpasim.tests.test_proto_conversion import install_alpasim_grpc_stubs

install_alpasim_grpc_stubs()

from alpagym_runtime.episode_runner.streaming_worker import (  # noqa: E402
    SharedPayloadState,
    StreamingRolloutWorker,
    _RolloutJob,
)
from alpagym_runtime.replay import ActionSelection, PolicyReplayData  # noqa: E402
from alpagym_runtime.types import EpisodeOutput, PolicyOutput, Trajectory  # noqa: E402


def _payload(prompt_idx: int) -> SimpleNamespace:
    """Build a duck-typed cosmos-rl payload that carries `prompt_idx`."""
    return SimpleNamespace(prompt_idx=prompt_idx)


def _resolve_scene(payload: SimpleNamespace) -> str:
    """Map prompt_idx -> scene id used as the per-payload scene identifier."""
    return f"scene-{payload.prompt_idx}"


def _make_episode(scene_id: str, session_uuid: str) -> EpisodeOutput:
    """Build a minimal in-memory `EpisodeOutput` for synchronous finalize calls."""
    return EpisodeOutput(
        scene_id=scene_id,
        session_uuid=session_uuid,
        num_steps=0,
        policy_outputs=(),
        executed_ego_trajectory=Trajectory(poses=()),
    )


class _StubWorker(StreamingRolloutWorker):
    """Worker subclass that replaces `_run_rollout` with a per-uuid outcome map.

    Each job's `session_uuid` is looked up in `outcomes_by_uuid`; if missing,
    the job's `scene_id` is consulted in `outcomes_by_scene`. The "success"
    branch finalizes with a fresh `EpisodeOutput`; "fail" finalizes with
    `RuntimeError`. Retries trigger fresh uuid lookups against the same maps.
    """

    def __init__(
        self,
        *,
        tmp_path: Path,
        outcomes_by_scene: dict[str, str],
        outcomes_by_uuid: dict[str, str] | None = None,
        max_concurrent_rollouts: int = 2,
        rollouts_per_payload: int = 1,
        max_scene_retries: int = 3,
    ) -> None:
        """Wire the worker with deterministic outcomes; tracks concurrency observed."""
        self._tmp_path = tmp_path
        self._outcomes_by_scene = outcomes_by_scene
        self._outcomes_by_uuid: dict[str, str] = dict(outcomes_by_uuid or {})
        self._call_log: list[str] = []  # list of session_uuid in dispatch order
        self._in_flight = 0
        self._max_in_flight = 0
        self._in_flight_lock = threading.Lock()
        self._gate = threading.Event()
        self._gate.set()
        super().__init__(
            alpasim_runtime_stub=SimpleNamespace(),
            driver_server=SimpleNamespace(
                topology_endpoint=SimpleNamespace(host="localhost", port=0),
            ),
            simulation_timeout_s=10.0,
            reward_config=SimpleNamespace(),
            max_concurrent_rollouts=max_concurrent_rollouts,
            rollouts_per_payload=rollouts_per_payload,
            scene_id_resolver=_resolve_scene,
            max_scene_retries=max_scene_retries,
        )

    def _run_rollout(self, rollout_job: _RolloutJob) -> None:
        """Skip real gRPC/disk work and finalize per the outcome map."""
        with self._in_flight_lock:
            self._in_flight += 1
            self._max_in_flight = max(self._max_in_flight, self._in_flight)
            self._call_log.append(rollout_job.session_uuid)
        try:
            self._gate.wait(timeout=5.0)
            outcome = self._outcomes_by_uuid.get(rollout_job.session_uuid)
            if outcome is None:
                outcome = self._outcomes_by_scene.get(rollout_job.scene_id, "success")
            if outcome == "success":
                self._on_rollout_succeeded(
                    rollout_job,
                    _make_episode(rollout_job.scene_id, rollout_job.session_uuid),
                )
            elif outcome == "fail":
                self._on_rollout_failed(
                    rollout_job,
                    RuntimeError(f"forced failure {rollout_job.session_uuid}"),
                )
            else:
                raise AssertionError(f"unknown outcome {outcome!r}")
        finally:
            with self._in_flight_lock:
                self._in_flight -= 1


def _drain_pool(worker: StreamingRolloutWorker) -> None:
    """Wait for all simulate-pool worker threads to finish before assertions."""
    worker.shutdown()
    for rollout_worker in worker._rollout_workers:
        rollout_worker.join(timeout=5.0)



def test_humanoid_worker_request_and_transition_payloads(tmp_path: Path) -> None:
    """Humanoid rollouts use the humanoid endpoint and carry PPO transition facts."""

    class _Servicer:
        def __init__(self) -> None:
            self.records: dict[str, object] = {}

        def pop_session_record(self, session_uuid: str) -> object:
            return self.records.pop(session_uuid)

    class _HumanoidPolicyServer:
        def __init__(self) -> None:
            self.topology_endpoint = SimpleNamespace(host="policy-host", port=5057)
            self.servicer = _Servicer()

    del tmp_path
    humanoid_policy_server = _HumanoidPolicyServer()
    worker = StreamingRolloutWorker(
        alpasim_runtime_stub=SimpleNamespace(),
        driver_server=None,
        humanoid_policy_server=humanoid_policy_server,
        simulation_domain="humanoid",
        simulation_timeout_s=10.0,
        reward_config=SimpleNamespace(),
        max_concurrent_rollouts=1,
        rollouts_per_payload=1,
        scene_id_resolver=_resolve_scene,
        scenario_id_resolver=lambda scene_id: "ascend",
    )
    try:
        rollout_job = _RolloutJob(
            shared_payload_state=SharedPayloadState(
                payload=_payload(0),
                n_target=1,
                future=Future(),
                retries_left=0,
                behavior_policy_version=7,
            ),
            session_uuid="humanoid-session",
            scene_id="stairs-scene",
        )
        request = worker._build_simulation_request(rollout_job)
        assert len(request.available_drivers) == 0
        assert request.available_humanoid_policies[0].ip == "policy-host"
        assert request.available_humanoid_policies[0].port == 5057
        assert list(request.rollout_specs[0].session_uuids) == ["humanoid-session"]
        assert request.rollout_specs[0].scene_id == "stairs-scene"
        assert request.rollout_specs[0].scenario_id == "ascend"

        replay_data = PolicyReplayData(
            replay_schema_version=1,
            payload_schema="test.g1",
            payload_schema_version=1,
            model_family="g1",
            action_selection=ActionSelection(set_ix=0, sample_ix=0),
            old_logprob=torch.tensor(-0.25),
            payload={"observation": {"x": torch.zeros(1)}, "action": torch.zeros(23)},
        )
        humanoid_policy_server.servicer.records["humanoid-session"] = SimpleNamespace(
            outputs=(
                PolicyOutput(
                    chosen_xyz=torch.zeros(1, 23),
                    chosen_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
                    chosen_dt_us=torch.zeros(1, dtype=torch.int64),
                    chosen_logprob=torch.tensor(-0.25),
                    replay_data=replay_data,
                    model_extra={
                        "humanoid_env_id": 0,
                        "humanoid_episode_id": 11,
                        "humanoid_step_index": 0,
                        "humanoid_timestamp_us": 100_000,
                        "humanoid_value": 0.75,
                    },
                ),
            ),
            final_bootstrap_values={},
            behavior_policy_version=7,
        )
        rollout_return = SimpleNamespace(
            aggregated_metrics={
                "humanoid_total_return": 2.0,
                "humanoid_return_env0": 2.0,
                "humanoid_episode_length_env0": 1.0,
            },
            behavior_policy_version="7",
            timestep_metrics=[
                SimpleNamespace(
                    name="humanoid_reward_env0",
                    timestamps_us=[120_000],
                    values=[2.0],
                    valid=[True],
                ),
                SimpleNamespace(
                    name="humanoid_terminated_env0",
                    timestamps_us=[120_000],
                    values=[1.0],
                    valid=[True],
                ),
                SimpleNamespace(
                    name="humanoid_truncated_env0",
                    timestamps_us=[120_000],
                    values=[0.0],
                    valid=[True],
                ),
            ],
        )

        episode = worker._build_humanoid_episode(rollout_job, rollout_return)

        assert episode.reward is not None
        assert episode.reward.total == pytest.approx(2.0)
        assert episode.metrics is not None
        assert episode.metrics.dense["humanoid_reward_env0"]["values"] == [2.0]
        transition = episode.policy_outputs[0].replay_data.payload["transition"]
        assert transition == {
            "env_id": 0,
            "episode_id": 11,
            "step_index": 0,
            "timestamp_us": 120_000,
            "reward": 2.0,
            "terminated": True,
            "truncated": False,
            "old_value": 0.75,
            "bootstrap_value": 0.0,
            "behavior_policy_version": 7,
        }
    finally:
        worker.shutdown()

# ---------- 1. Slot budget respected ----------


def test_slot_budget_caps_in_flight_simulate_calls(tmp_path: Path) -> None:
    """At most `max_concurrent_rollouts` simulate calls run in flight."""
    worker = _StubWorker(
        tmp_path=tmp_path,
        outcomes_by_scene={f"scene-{i}": "success" for i in range(5)},
        max_concurrent_rollouts=2,
        rollouts_per_payload=1,
    )
    worker._gate.clear()
    try:
        payload_states = [worker.submit_payload(_payload(i)) for i in range(5)]
        # Give the pool a moment to saturate.
        for _ in range(50):
            with worker._in_flight_lock:
                if worker._in_flight >= 2:
                    break
            threading.Event().wait(0.01)
        assert worker._max_in_flight == 2
        worker._gate.set()
        for payload_state in payload_states:
            assert payload_state.future.result(timeout=5.0)[0].scene_id == _resolve_scene(
                payload_state.payload
            )
        assert worker._max_in_flight == 2
    finally:
        worker._gate.set()
        _drain_pool(worker)


# ---------- 2. Per-payload future resolution ----------


def test_future_resolves_when_n_target_siblings_succeed(tmp_path: Path) -> None:
    """Future fires with exactly `rollouts_per_payload` artifacts when all siblings succeed."""
    worker = _StubWorker(
        tmp_path=tmp_path,
        outcomes_by_scene={"scene-0": "success"},
        max_concurrent_rollouts=4,
        rollouts_per_payload=3,
    )
    try:
        payload_state = worker.submit_payload(_payload(0))
        artifacts = payload_state.future.result(timeout=5.0)
        assert len(artifacts) == 3
        assert {a.scene_id for a in artifacts} == {"scene-0"}
        # Each sibling had its own uuid.
        assert len({a.session_uuid for a in artifacts}) == 3
    finally:
        _drain_pool(worker)


# ---------- 3. Idempotent submit returns the same state ----------


def test_duplicate_submit_returns_same_state(tmp_path: Path) -> None:
    """A repeat `submit_payload(p)` with the same `prompt_idx` returns the same state."""
    worker = _StubWorker(
        tmp_path=tmp_path,
        outcomes_by_scene={f"scene-{i}": "success" for i in range(4)},
        max_concurrent_rollouts=4,
        rollouts_per_payload=1,
    )
    worker._gate.clear()
    try:
        first = [worker.submit_payload(_payload(i)) for i in range(4)]
        # Second submit returns the same state object; no fresh dispatch.
        second = [worker.submit_payload(_payload(i)) for i in range(4)]
        for first_state, second_state in zip(first, second, strict=True):
            assert second_state is first_state
        worker._gate.set()
        for payload_state in second:
            payload_state.future.result(timeout=5.0)
        # Exactly one dispatch per prompt_idx, not two.
        assert len(worker._call_log) == 4
    finally:
        worker._gate.set()
        _drain_pool(worker)


# ---------- 4. Bootstrap path (cache miss) ----------


def test_submit_payload_dispatches_inline_on_cache_miss(tmp_path: Path) -> None:
    """First submit with a previously-unseen `prompt_idx` dispatches immediately."""
    worker = _StubWorker(
        tmp_path=tmp_path,
        outcomes_by_scene={"scene-0": "success", "scene-1": "success"},
        max_concurrent_rollouts=2,
        rollouts_per_payload=1,
    )
    try:
        payload_states = [worker.submit_payload(_payload(0)), worker.submit_payload(_payload(1))]
        for payload_state in payload_states:
            artifacts = payload_state.future.result(timeout=5.0)
            assert len(artifacts) == 1
    finally:
        _drain_pool(worker)


# ---------- 5. Distinct prompt_idx dispatches independently ----------


def test_distinct_prompt_idx_dispatches_independently(tmp_path: Path) -> None:
    """Submits with different `prompt_idx`s produce distinct state objects."""
    worker = _StubWorker(
        tmp_path=tmp_path,
        outcomes_by_scene={f"scene-{i}": "success" for i in range(3)},
        max_concurrent_rollouts=3,
        rollouts_per_payload=1,
    )
    try:
        first = worker.submit_payload(_payload(0))
        second = worker.submit_payload(_payload(2))
        # Different prompt_idx -> different state objects, both running.
        assert second is not first
        # Repeat submit on the original prompt_idx still returns the original state.
        repeat = worker.submit_payload(_payload(0))
        assert repeat is first
        for payload_state in (first, second):
            payload_state.future.result(timeout=5.0)
    finally:
        _drain_pool(worker)


# ---------- 6. Retry with fresh uuid ----------


def test_failed_job_retries_with_fresh_uuid_and_decrements_budget(tmp_path: Path) -> None:
    """A failing job is re-enqueued with a new session_uuid and the budget decrements."""
    # First uuid fails; the retry (any other uuid) succeeds via scene-level default.
    worker = _StubWorker(
        tmp_path=tmp_path,
        outcomes_by_scene={"scene-0": "success"},
        max_concurrent_rollouts=1,
        rollouts_per_payload=1,
        max_scene_retries=3,
    )
    worker._gate.clear()
    try:
        payload_state = worker.submit_payload(_payload(0))
        # Wait until the first uuid enters `_run_rollout` (and blocks on the gate).
        for _ in range(200):
            if len(worker._call_log) >= 1:
                break
            threading.Event().wait(0.01)
        assert len(worker._call_log) == 1
        first_uuid = worker._call_log[0]
        worker._outcomes_by_uuid[first_uuid] = "fail"
        worker._gate.set()
        artifacts = payload_state.future.result(timeout=5.0)
        assert len(artifacts) == 1
        assert artifacts[0].session_uuid != first_uuid  # retry minted a fresh uuid
        assert payload_state.retries_left == 2  # one retry consumed
    finally:
        worker._gate.set()
        _drain_pool(worker)


# ---------- 7. Retry exhaustion drops the payload ----------


def test_retry_exhaustion_resolves_future_with_empty_list(tmp_path: Path) -> None:
    """Once `retries_left < 0` the future resolves empty and the payload is dropped."""
    worker = _StubWorker(
        tmp_path=tmp_path,
        outcomes_by_scene={"scene-0": "fail"},
        max_concurrent_rollouts=1,
        rollouts_per_payload=1,
        max_scene_retries=2,
    )
    try:
        payload_state = worker.submit_payload(_payload(0))
        result = payload_state.future.result(timeout=5.0)
        assert result == []
        assert payload_state.permanently_failed is True
        # initial attempt + 2 retries = 3 dispatched uuids; the 3rd failure exhausts.
        assert len(worker._call_log) == 3
    finally:
        _drain_pool(worker)


def test_retry_exhaustion_drops_pending_siblings(tmp_path: Path) -> None:
    """A permanently-failed payload drops its still-pending siblings."""
    worker = _StubWorker(
        tmp_path=tmp_path,
        outcomes_by_scene={"scene-0": "fail"},
        max_concurrent_rollouts=1,  # so only 1 sibling can be in flight at a time
        rollouts_per_payload=3,  # 3 siblings
        max_scene_retries=0,
    )
    try:
        payload_state = worker.submit_payload(_payload(0))
        result = payload_state.future.result(timeout=5.0)
        assert result == []
        # Only the first sibling reaches _run_rollout; the other two are dropped.
        assert len(worker._call_log) == 1
    finally:
        _drain_pool(worker)


# ---------- 8. Independent retry quotas across duplicate-scene payloads ----------


def test_duplicate_scene_payloads_have_independent_retry_quotas(tmp_path: Path) -> None:
    """Two payloads sharing a scene_id each get the full `max_scene_retries` quota."""

    # Both payloads resolve to "scene-0"; one fails consistently, the other succeeds.
    def shared_resolver(payload: SimpleNamespace) -> str:
        del payload
        return "scene-0"

    worker = _StubWorker(
        tmp_path=tmp_path,
        outcomes_by_scene={"scene-0": "success"},
        max_concurrent_rollouts=2,
        rollouts_per_payload=1,
        max_scene_retries=2,
    )
    worker._scene_id_resolver = shared_resolver  # both payloads -> "scene-0"
    try:
        worker._gate.clear()  # hold workers so we can inspect dispatched uuids
        payload_states = [
            worker.submit_payload(_payload(0)),
            worker.submit_payload(_payload(1)),
        ]
        # Wait until both initial uuids are dispatched before tagging outcomes.
        for _ in range(200):
            if len(worker._call_log) >= 2:
                break
            threading.Event().wait(0.01)
        assert len(worker._call_log) == 2
        first_uuid = worker._call_log[0]
        worker._outcomes_by_uuid[first_uuid] = "fail"  # payload A's first attempt fails
        worker._gate.set()
        for payload_state in payload_states:
            artifacts = payload_state.future.result(timeout=5.0)
            assert len(artifacts) == 1
        # The dropped sibling A still completes via retry. Payload B never touches A's quota.
        retries_consumed = [s for s in payload_states if s.retries_left < 2]
        assert len(retries_consumed) == 1
        untouched = [s for s in payload_states if s.retries_left == 2]
        assert len(untouched) == 1
    finally:
        worker._gate.set()
        _drain_pool(worker)


# ---------- 9. Re-submit after resolution dispatches a fresh rollout ----------


def test_resubmit_after_resolution_dispatches_fresh_rollout(tmp_path: Path) -> None:
    """A repeat `submit_payload(p)` after the previous future resolved runs a new rollout."""
    worker = _StubWorker(
        tmp_path=tmp_path,
        outcomes_by_scene={"scene-0": "success", "scene-1": "fail"},
        max_concurrent_rollouts=1,
        rollouts_per_payload=1,
        max_scene_retries=0,
    )
    try:
        # After a successful resolution, re-submitting the same prompt_idx
        # returns a distinct state and dispatches another simulate(1).
        first_success = worker.submit_payload(_payload(0))
        first_success.future.result(timeout=5.0)
        assert len(worker._call_log) == 1
        second_success = worker.submit_payload(_payload(0))
        assert second_success is not first_success
        second_success.future.result(timeout=5.0)
        assert len(worker._call_log) == 2

        # Same contract after a permanent-failure resolution.
        first_fail = worker.submit_payload(_payload(1))
        assert first_fail.future.result(timeout=5.0) == []
        calls_after_first_fail = len(worker._call_log)
        second_fail = worker.submit_payload(_payload(1))
        assert second_fail is not first_fail
        assert second_fail.future.result(timeout=5.0) == []
        assert len(worker._call_log) > calls_after_first_fail
    finally:
        _drain_pool(worker)


# ---------- shutdown ----------


def test_shutdown_rejects_subsequent_submits(tmp_path: Path) -> None:
    """`shutdown` aborts the pool and refuses new dispatch."""
    worker = _StubWorker(
        tmp_path=tmp_path,
        outcomes_by_scene={"scene-0": "success"},
        max_concurrent_rollouts=2,
        rollouts_per_payload=1,
    )
    worker.shutdown()
    with pytest.raises(RuntimeError, match="shut down"):
        worker.submit_payload(_payload(0))


# ---------- internal dataclasses ----------


def test_shared_payload_state_carries_retry_budget(tmp_path: Path) -> None:
    """Initial `SharedPayloadState` retries_left matches the worker's `max_scene_retries`."""
    del tmp_path
    payload_state = SharedPayloadState(
        payload=_payload(0),
        n_target=2,
        future=Future(),
        retries_left=3,
    )
    assert payload_state.retries_left == 3
    assert payload_state.collected == []
    assert payload_state.permanently_failed is False
    assert payload_state.future_resolved is False
