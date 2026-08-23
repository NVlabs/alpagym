# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the streaming AlpaSim rollout worker.

These contracts pin dispatch-state-machine behaviors of
`StreamingRolloutWorker`. Each test uses a `_StubWorker` subclass that
replaces `_run_rollout` with a deterministic per-uuid outcome map, so
no AlpaSim runtime, gRPC channel, disk I/O, or reward computation runs.
"""

import hashlib
import logging
import threading
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from alpagym_runtime.alpasim.tests.test_proto_conversion import (
    install_alpasim_grpc_stubs,
)

install_alpasim_grpc_stubs()

from alpagym_runtime.episode_runner.streaming_worker import (  # noqa: E402
    SharedPayloadState,
    StreamingRolloutWorker,
    _RolloutJob,
)
from alpagym_runtime.inference.inference_engine import InferenceModelLease  # noqa: E402
from alpagym_runtime.replay import ActionSelection, PolicyReplayData  # noqa: E402
from alpagym_runtime.types import EpisodeOutput, PolicyOutput, Trajectory  # noqa: E402


def _payload(prompt_idx: int) -> SimpleNamespace:
    """Build a duck-typed cosmos-rl payload that carries `prompt_idx`."""
    return SimpleNamespace(prompt_idx=prompt_idx)


def _resolve_scene(payload: SimpleNamespace) -> str:
    """Map prompt_idx -> scene id used as the per-payload scene identifier."""
    return f"scene-{payload.prompt_idx}"


def _make_episode(
    scene_id: str,
    session_uuid: str,
    rollout_seed: int | None = None,
) -> EpisodeOutput:
    """Build a minimal in-memory `EpisodeOutput` for synchronous finalize calls."""
    return EpisodeOutput(
        scene_id=scene_id,
        session_uuid=session_uuid,
        num_steps=0,
        policy_outputs=(),
        rollout_seed=rollout_seed,
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
        simulation_domain: str = "av",
        rollout_seed_base: int | None = None,
    ) -> None:
        """Wire the worker with deterministic outcomes; tracks concurrency observed."""
        self._tmp_path = tmp_path
        self._outcomes_by_scene = outcomes_by_scene
        self._outcomes_by_uuid: dict[str, str] = dict(outcomes_by_uuid or {})
        self._call_log: list[str] = []  # list of session_uuid in dispatch order
        self._random_seed_log: list[int | None] = []
        self._lease_reservations: list[tuple[str, int, InferenceModelLease | None]] = []
        self._discarded_sessions: list[str] = []
        self._in_flight = 0
        self._max_in_flight = 0
        self._in_flight_lock = threading.Lock()
        self._gate = threading.Event()
        self._gate.set()
        humanoid_policy_server = None
        driver_server = SimpleNamespace(
            topology_endpoint=SimpleNamespace(host="localhost", port=0),
        )
        if simulation_domain == "humanoid":

            class _Servicer:
                def reserve_session(
                    self,
                    session_uuid: str,
                    behavior_policy_version: int,
                    model_lease: InferenceModelLease | None = None,
                ) -> None:
                    self_reservation = (
                        session_uuid,
                        behavior_policy_version,
                        model_lease,
                    )
                    self_outer._lease_reservations.append(self_reservation)

                def discard_session(self, session_uuid: str) -> None:
                    self_outer._discarded_sessions.append(session_uuid)

            self_outer = self
            humanoid_policy_server = SimpleNamespace(
                topology_endpoint=SimpleNamespace(host="localhost", port=0),
                servicer=_Servicer(),
            )
            driver_server = None
        super().__init__(
            alpasim_runtime_stub=SimpleNamespace(),
            driver_server=driver_server,
            humanoid_policy_server=humanoid_policy_server,
            simulation_domain=simulation_domain,
            simulation_timeout_s=10.0,
            reward_config=SimpleNamespace(),
            max_concurrent_rollouts=max_concurrent_rollouts,
            rollouts_per_payload=rollouts_per_payload,
            scene_id_resolver=_resolve_scene,
            scenario_id_resolver=lambda scene_id: "ascend",
            max_scene_retries=max_scene_retries,
            rollout_seed_base=rollout_seed_base,
        )

    def _run_rollout(self, rollout_job: _RolloutJob) -> None:
        """Skip real gRPC/disk work and finalize per the outcome map."""
        with self._in_flight_lock:
            self._in_flight += 1
            self._max_in_flight = max(self._max_in_flight, self._in_flight)
            self._call_log.append(rollout_job.session_uuid)
            self._random_seed_log.append(rollout_job.random_seed)
        try:
            self._gate.wait(timeout=5.0)
            outcome = self._outcomes_by_uuid.get(rollout_job.session_uuid)
            if outcome is None:
                outcome = self._outcomes_by_scene.get(rollout_job.scene_id, "success")
            if outcome == "success":
                self._on_rollout_succeeded(
                    rollout_job,
                    _make_episode(
                        rollout_job.scene_id,
                        rollout_job.session_uuid,
                        rollout_job.random_seed,
                    ),
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
        expected_seed = (
            int.from_bytes(
                hashlib.sha256(b"alpagym-humanoid-v1:humanoid-session").digest()[:8],
                "big",
            )
            or 1
        )
        assert request.rollout_specs[0].random_seed == expected_seed
        assert request.rollout_specs[0].expected_behavior_policy_version == "7"

        rollout_job.random_seed = 17
        fixed_seed_request = worker._build_simulation_request(rollout_job)
        assert fixed_seed_request.rollout_specs[0].random_seed == 17

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
        assert episode.rollout_seed == 17
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
            assert payload_state.future.result(timeout=5.0)[
                0
            ].scene_id == _resolve_scene(payload_state.payload)
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
        payload_states = [
            worker.submit_payload(_payload(0)),
            worker.submit_payload(_payload(1)),
        ]
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


def test_failed_job_retries_with_fresh_uuid_and_decrements_budget(
    tmp_path: Path,
) -> None:
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


def test_fixed_seed_panel_is_sequential_across_siblings_and_episodes(
    tmp_path: Path,
) -> None:
    """Fresh humanoid jobs consume ``base + ordinal`` in creation order."""
    worker = _StubWorker(
        tmp_path=tmp_path,
        outcomes_by_scene={"scene-0": "success", "scene-1": "success"},
        max_concurrent_rollouts=1,
        rollouts_per_payload=3,
        simulation_domain="humanoid",
        rollout_seed_base=100,
    )
    try:
        first = worker.submit_payload(_payload(0), behavior_policy_version=7)
        assert len(first.future.result(timeout=5.0)) == 3
        second = worker.submit_payload(_payload(1), behavior_policy_version=7)
        assert len(second.future.result(timeout=5.0)) == 3

        assert worker._random_seed_log == [100, 101, 102, 103, 104, 105]
        assert [episode.rollout_seed for episode in first.collected] == [100, 101, 102]
        assert [episode.rollout_seed for episode in second.collected] == [103, 104, 105]
    finally:
        _drain_pool(worker)


def test_fixed_seed_retry_reuses_seed_while_minting_fresh_uuid(tmp_path: Path) -> None:
    """Retry changes UUID but retains seed, behavior version, and model lease."""
    worker = _StubWorker(
        tmp_path=tmp_path,
        outcomes_by_scene={"scene-0": "success"},
        max_concurrent_rollouts=1,
        max_scene_retries=1,
        simulation_domain="humanoid",
        rollout_seed_base=42,
    )
    lease = InferenceModelLease(
        behavior_policy_version=3,
        model=torch.nn.Linear(1, 1),
    )
    worker._gate.clear()
    try:
        payload_state = worker.submit_payload(
            _payload(0),
            behavior_policy_version=3,
            model_lease=lease,
        )
        for _ in range(200):
            if worker._call_log:
                break
            threading.Event().wait(0.01)
        assert len(worker._call_log) == 1
        first_uuid = worker._call_log[0]
        worker._outcomes_by_uuid[first_uuid] = "fail"
        worker._gate.set()

        artifacts = payload_state.future.result(timeout=5.0)
        assert len(artifacts) == 1
        assert artifacts[0].session_uuid != first_uuid
        assert worker._random_seed_log == [42, 42]
        assert len(worker._lease_reservations) == 2
        assert {version for _, version, _ in worker._lease_reservations} == {3}
        assert all(
            reserved_lease is lease
            for _, _, reserved_lease in worker._lease_reservations
        )
    finally:
        worker._gate.set()
        _drain_pool(worker)


def test_unset_seed_base_keeps_legacy_uuid_hash_selection(tmp_path: Path) -> None:
    """Training-default jobs leave proto conversion to derive the UUID hash seed."""
    worker = _StubWorker(
        tmp_path=tmp_path,
        outcomes_by_scene={"scene-0": "success"},
        max_concurrent_rollouts=1,
        simulation_domain="humanoid",
        rollout_seed_base=None,
    )
    try:
        payload_state = worker.submit_payload(_payload(0), behavior_policy_version=0)
        assert len(payload_state.future.result(timeout=5.0)) == 1
        assert worker._random_seed_log == [None]
    finally:
        _drain_pool(worker)


def test_fixed_seed_panel_rejects_job_ordinal_overflow(tmp_path: Path) -> None:
    """A panel cannot silently wrap past the uint64 ABI boundary."""
    worker = _StubWorker(
        tmp_path=tmp_path,
        outcomes_by_scene={"scene-0": "success"},
        max_concurrent_rollouts=1,
        rollouts_per_payload=2,
        simulation_domain="humanoid",
        rollout_seed_base=(1 << 64) - 1,
    )
    try:
        with pytest.raises(OverflowError, match="exceeds uint64"):
            worker.submit_payload(_payload(0), behavior_policy_version=0)
    finally:
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


def test_simulate_failure_retries_when_failed_policy_close_raises(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Policy-close diagnostics cannot replace a simulate failure or lose its retry."""

    class _RuntimeStub:
        def __init__(self) -> None:
            self.calls = 0

        def simulate(self, request, timeout):
            del request, timeout
            self.calls += 1
            return SimpleNamespace(
                rollout_returns=[
                    SimpleNamespace(success=False, error="simulate primary")
                ]
            )

    class _Policy:
        def close(self) -> None:
            raise RuntimeError("policy close failed")

    class _Servicer:
        def __init__(self) -> None:
            self.sessions: dict[str, _Policy] = {}
            self.records: dict[str, object] = {}

        def reserve_session(
            self,
            session_uuid: str,
            behavior_policy_version: int,
            model_lease: InferenceModelLease | None = None,
        ) -> None:
            del behavior_policy_version, model_lease
            self.sessions[session_uuid] = _Policy()

        def discard_session(self, session_uuid: str) -> None:
            policy = self.sessions.pop(session_uuid, None)
            self.records.pop(session_uuid, None)
            if policy is not None:
                policy.close()

    runtime_stub = _RuntimeStub()
    servicer = _Servicer()
    policy_server = SimpleNamespace(
        topology_endpoint=SimpleNamespace(host="localhost", port=0),
        servicer=servicer,
    )
    worker = StreamingRolloutWorker(
        alpasim_runtime_stub=runtime_stub,
        driver_server=None,
        humanoid_policy_server=policy_server,
        simulation_domain="humanoid",
        simulation_timeout_s=10.0,
        reward_config=SimpleNamespace(),
        max_concurrent_rollouts=1,
        rollouts_per_payload=1,
        scene_id_resolver=_resolve_scene,
        scenario_id_resolver=lambda scene_id: "ascend",
        max_scene_retries=1,
    )
    try:
        with caplog.at_level(logging.WARNING):
            payload_state = worker.submit_payload(
                _payload(0), behavior_policy_version=7
            )
            assert payload_state.future.result(timeout=5.0) == []

        assert runtime_stub.calls == 2
        assert payload_state.retries_left == -1
        assert payload_state.permanently_failed is True
        assert servicer.sessions == {}
        assert servicer.records == {}
        assert "simulate primary" in caplog.text
        assert "policy close failed" in caplog.text
    finally:
        _drain_pool(worker)


def test_retry_reservation_failure_drops_payload_and_cleans_partial_uuid(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A partially allocated retry UUID cannot strand the payload future."""

    class _RuntimeStub:
        def __init__(self) -> None:
            self.calls = 0

        def simulate(self, request, timeout):
            del request, timeout
            self.calls += 1
            return SimpleNamespace(
                rollout_returns=[
                    SimpleNamespace(success=False, error="simulate primary")
                ]
            )

    class _Servicer:
        def __init__(self) -> None:
            self.reserve_calls: list[str] = []
            self.sessions: dict[str, object] = {}
            self.records: dict[str, object] = {}

        def reserve_session(
            self,
            session_uuid: str,
            behavior_policy_version: int,
            model_lease: InferenceModelLease | None = None,
        ) -> None:
            del behavior_policy_version, model_lease
            self.reserve_calls.append(session_uuid)
            self.sessions[session_uuid] = object()
            self.records[session_uuid] = object()
            if len(self.reserve_calls) == 2:
                raise RuntimeError("retry reserve failed after partial allocation")

        def discard_session(self, session_uuid: str) -> None:
            self.sessions.pop(session_uuid, None)
            self.records.pop(session_uuid, None)

    runtime_stub = _RuntimeStub()
    servicer = _Servicer()
    worker = StreamingRolloutWorker(
        alpasim_runtime_stub=runtime_stub,
        driver_server=None,
        humanoid_policy_server=SimpleNamespace(
            topology_endpoint=SimpleNamespace(host="localhost", port=0),
            servicer=servicer,
        ),
        simulation_domain="humanoid",
        simulation_timeout_s=10.0,
        reward_config=SimpleNamespace(),
        max_concurrent_rollouts=1,
        rollouts_per_payload=1,
        scene_id_resolver=_resolve_scene,
        scenario_id_resolver=lambda scene_id: "ascend",
        max_scene_retries=1,
    )
    try:
        with caplog.at_level(logging.WARNING):
            payload_state = worker.submit_payload(
                _payload(0), behavior_policy_version=7
            )
            assert payload_state.future.result(timeout=5.0) == []

        assert runtime_stub.calls == 1
        assert len(servicer.reserve_calls) == 2
        assert len(set(servicer.reserve_calls)) == 2
        assert servicer.sessions == {}
        assert servicer.records == {}
        assert payload_state.permanently_failed is True
        assert payload_state.future_resolved is True
        assert 0 not in worker._active_payload_states
        assert worker._rollout_workers[0].is_alive()
        assert "simulate primary" in caplog.text
        assert "retry reserve failed after partial allocation" in caplog.text
    finally:
        _drain_pool(worker)


def test_retry_reservation_failure_cannot_poison_concurrent_success() -> None:
    """A sibling success remains final while a failed retry reservation drains."""

    retry_reserve_started = threading.Event()
    release_retry_reserve = threading.Event()

    class _RuntimeStub:
        def simulate(self, request, timeout):
            del request, timeout
            return SimpleNamespace(
                rollout_returns=[
                    SimpleNamespace(success=False, error="simulate primary")
                ]
            )

    class _Servicer:
        def __init__(self) -> None:
            self.reserve_calls: list[str] = []
            self.sessions: set[str] = set()

        def reserve_session(
            self,
            session_uuid: str,
            behavior_policy_version: int,
            model_lease: InferenceModelLease | None = None,
        ) -> None:
            del behavior_policy_version, model_lease
            self.reserve_calls.append(session_uuid)
            self.sessions.add(session_uuid)
            if len(self.reserve_calls) == 2:
                retry_reserve_started.set()
                assert release_retry_reserve.wait(timeout=5.0)
                raise RuntimeError("retry reserve lost race with sibling success")

        def discard_session(self, session_uuid: str) -> None:
            self.sessions.discard(session_uuid)

    servicer = _Servicer()
    worker = StreamingRolloutWorker(
        alpasim_runtime_stub=_RuntimeStub(),
        driver_server=None,
        humanoid_policy_server=SimpleNamespace(
            topology_endpoint=SimpleNamespace(host="localhost", port=0),
            servicer=servicer,
        ),
        simulation_domain="humanoid",
        simulation_timeout_s=10.0,
        reward_config=SimpleNamespace(),
        max_concurrent_rollouts=1,
        rollouts_per_payload=1,
        scene_id_resolver=_resolve_scene,
        scenario_id_resolver=lambda scene_id: "ascend",
        max_scene_retries=1,
    )
    try:
        payload_state = worker.submit_payload(_payload(0), behavior_policy_version=7)
        assert retry_reserve_started.wait(timeout=5.0)
        accepted = _make_episode("scene-0", "concurrent-sibling")
        with worker._lock:
            payload_state.collected.append(accepted)
            payload_state.future_resolved = True
            worker._active_payload_states.pop(0, None)
        payload_state.future.set_result([accepted])
        release_retry_reserve.set()

        assert payload_state.future.result(timeout=5.0) == [accepted]
        for _ in range(200):
            if len(servicer.reserve_calls) == 2 and not servicer.sessions:
                break
            threading.Event().wait(0.01)
        assert payload_state.permanently_failed is False
        assert payload_state.future_resolved is True
        assert servicer.sessions == set()
        assert worker._rollout_workers[0].is_alive()
    finally:
        release_retry_reserve.set()
        _drain_pool(worker)


def test_retry_enqueue_failure_drops_payload_and_cleans_reserved_uuid(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A retry queue failure invalidates its reservation and resolves empty."""

    class _RuntimeStub:
        def __init__(self) -> None:
            self.calls = 0

        def simulate(self, request, timeout):
            del request, timeout
            self.calls += 1
            return SimpleNamespace(
                rollout_returns=[
                    SimpleNamespace(success=False, error="simulate primary")
                ]
            )

    class _Servicer:
        def __init__(self) -> None:
            self.reserve_calls: list[str] = []
            self.sessions: dict[str, object] = {}

        def reserve_session(
            self,
            session_uuid: str,
            behavior_policy_version: int,
            model_lease: InferenceModelLease | None = None,
        ) -> None:
            del behavior_policy_version, model_lease
            self.reserve_calls.append(session_uuid)
            self.sessions[session_uuid] = object()

        def discard_session(self, session_uuid: str) -> None:
            self.sessions.pop(session_uuid, None)

    runtime_stub = _RuntimeStub()
    servicer = _Servicer()
    worker = StreamingRolloutWorker(
        alpasim_runtime_stub=runtime_stub,
        driver_server=None,
        humanoid_policy_server=SimpleNamespace(
            topology_endpoint=SimpleNamespace(host="localhost", port=0),
            servicer=servicer,
        ),
        simulation_domain="humanoid",
        simulation_timeout_s=10.0,
        reward_config=SimpleNamespace(),
        max_concurrent_rollouts=1,
        rollouts_per_payload=1,
        scene_id_resolver=_resolve_scene,
        scenario_id_resolver=lambda scene_id: "ascend",
        max_scene_retries=1,
    )
    original_put = worker._rollout_job_queue.put

    def _fail_retry_enqueue(item, *args, **kwargs) -> None:
        if item[0] == 0:
            raise RuntimeError("retry enqueue failed")
        original_put(item, *args, **kwargs)

    monkeypatch.setattr(worker._rollout_job_queue, "put", _fail_retry_enqueue)
    try:
        with caplog.at_level(logging.WARNING):
            payload_state = worker.submit_payload(
                _payload(0), behavior_policy_version=7
            )
            assert payload_state.future.result(timeout=5.0) == []

        assert runtime_stub.calls == 1
        assert len(servicer.reserve_calls) == 2
        assert len(set(servicer.reserve_calls)) == 2
        assert servicer.sessions == {}
        assert payload_state.permanently_failed is True
        assert payload_state.future_resolved is True
        assert 0 not in worker._active_payload_states
        assert worker._rollout_workers[0].is_alive()
        assert "simulate primary" in caplog.text
        assert "retry enqueue failed" in caplog.text
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


def test_retry_exhaustion_discards_skipped_humanoid_sibling_reservations(
    tmp_path: Path,
) -> None:
    """Skipped siblings release reservations after one sibling permanently fails."""
    worker = _StubWorker(
        tmp_path=tmp_path,
        outcomes_by_scene={"scene-0": "fail"},
        max_concurrent_rollouts=1,
        rollouts_per_payload=3,
        max_scene_retries=0,
        simulation_domain="humanoid",
    )
    try:
        payload_state = worker.submit_payload(_payload(0), behavior_policy_version=4)
        assert payload_state.future.result(timeout=5.0) == []
        for _ in range(200):
            if len(worker._discarded_sessions) == 3:
                break
            threading.Event().wait(0.01)

        reserved = {session_uuid for session_uuid, _, _ in worker._lease_reservations}
        assert len(worker._call_log) == 1
        assert set(worker._discarded_sessions) == reserved
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


@pytest.mark.parametrize("running_outcome", ["success", "fail"])
def test_shutdown_discards_only_queued_humanoid_sessions(
    tmp_path: Path,
    running_outcome: str,
) -> None:
    """Shutdown drains queued reservations without touching an in-flight session."""
    worker = _StubWorker(
        tmp_path=tmp_path,
        outcomes_by_scene={"scene-0": running_outcome},
        max_concurrent_rollouts=1,
        rollouts_per_payload=3,
        max_scene_retries=0,
        simulation_domain="humanoid",
    )
    worker._gate.clear()
    payload_state = worker.submit_payload(_payload(0), behavior_policy_version=5)
    for _ in range(200):
        if worker._call_log:
            break
        threading.Event().wait(0.01)
    assert len(worker._call_log) == 1
    running_session = worker._call_log[0]
    reserved = {session_uuid for session_uuid, _, _ in worker._lease_reservations}
    queued_sessions = reserved - {running_session}

    worker.shutdown()

    assert payload_state.future.result(timeout=5.0) == []
    assert set(worker._discarded_sessions) == queued_sessions
    assert running_session not in worker._discarded_sessions

    worker._gate.set()
    for rollout_worker in worker._rollout_workers:
        rollout_worker.join(timeout=5.0)

    if running_outcome == "fail":
        assert set(worker._discarded_sessions) == reserved
    else:
        assert set(worker._discarded_sessions) == queued_sessions


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
