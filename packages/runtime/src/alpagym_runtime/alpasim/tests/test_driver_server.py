# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from alpagym_runtime.alpasim.tests.test_proto_conversion import (
    install_alpasim_grpc_stubs,
)

install_alpasim_grpc_stubs()

import pytest  # noqa: E402
import torch  # noqa: E402
from alpagym_runtime.alpasim.driver_server import (  # noqa: E402
    EgodriverGrpcServicer,
    EgodriverServer,
    SessionRecord,
    _Session,
)
from alpagym_runtime.alpasim.humanoid_policy_server import (  # noqa: E402
    MOTION_REFERENCE_JOINT_NAMES,
    HumanoidMotionReference,
    HumanoidMotionReferenceFrame,
    HumanoidPolicyGrpcServicer,
    HumanoidPolicyStepOutput,
    ZeroHumanoidPolicy,
)
from alpagym_runtime.replay import ActionSelection, PolicyReplayData  # noqa: E402
from alpagym_runtime.types import (  # noqa: E402
    EgoPose,
    PolicyInput,
    PolicyOutput,
    Pose,
    RolloutCalibration,
    RouteWaypoint,
    Trajectory,
    Vec3,
)
from alpasim_grpc.v0.common_pb2 import Trajectory as ProtoTrajectory  # noqa: E402
from alpasim_grpc.v0.egodriver_pb2 import (  # noqa: E402
    GroundTruth as ProtoGroundTruth,
    RolloutCameraImage,
    Route,
)


class _StubPolicy:
    """Minimal Policy implementation that records inputs for simulator-layer tests."""

    def __init__(
        self,
        session_uuid: str,
        calibration: RolloutCalibration,
        random_seed: int,
    ) -> None:
        """Store the construction args and prepare recording state."""
        self.session_uuid = session_uuid
        self.calibration = calibration
        self.random_seed = random_seed
        self.received_inputs: list[PolicyInput] = []
        self.closed = False

    def step(self, policy_input: PolicyInput) -> PolicyOutput:
        """Record the input and return a fixed policy-owned trajectory."""
        self.received_inputs.append(policy_input)
        return PolicyOutput(
            chosen_xyz=torch.tensor(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                dtype=torch.float32,
            ),
            chosen_quat=torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
            ),
            chosen_dt_us=torch.tensor([0, 100, 200], dtype=torch.int64),
        )

    def close(self) -> None:
        """Mark the policy closed for assertions."""
        self.closed = True


def _make_rollout_spec(camera_logical_ids: tuple[str, ...] = ("front",)) -> Any:
    """Build a duck-typed rollout_spec with one or more available cameras."""
    cameras = [
        SimpleNamespace(
            logical_id=logical_id,
            intrinsics=SimpleNamespace(
                opencv_pinhole_param=SimpleNamespace(
                    focal_length_x=1.0,
                    focal_length_y=1.0,
                    principal_point_x=0.5,
                    principal_point_y=0.5,
                ),
            ),
            rig_to_camera=SimpleNamespace(
                vec=SimpleNamespace(x=0.0, y=0.0, z=0.0),
                quat=SimpleNamespace(w=1.0, x=0.0, y=0.0, z=0.0),
            ),
        )
        for logical_id in camera_logical_ids
    ]
    return SimpleNamespace(
        vehicle=SimpleNamespace(available_cameras=cameras),
    )


def _make_factory() -> tuple[list[_StubPolicy], Any]:
    """Return a recording-list and a factory that appends each created policy."""
    created: list[_StubPolicy] = []

    def factory(
        session_uuid: str,
        calibration: RolloutCalibration,
        random_seed: int,
    ) -> _StubPolicy:
        """Instantiate one `_StubPolicy` per session and record it."""
        policy = _StubPolicy(session_uuid, calibration, random_seed)
        created.append(policy)
        return policy

    return created, factory


def _proto_trajectory(*timestamps_us: int) -> ProtoTrajectory:
    """Build a proto trajectory with identity poses at the given timestamps."""
    return ProtoTrajectory(
        poses=[
            SimpleNamespace(
                timestamp_us=timestamp_us,
                pose=SimpleNamespace(
                    vec=SimpleNamespace(x=float(timestamp_us), y=0.0, z=0.0),
                    quat=SimpleNamespace(w=1.0, x=0.0, y=0.0, z=0.0),
                ),
            )
            for timestamp_us in timestamps_us
        ]
    )


def test_egodriver_server_binds_port_and_publishes_topology_endpoint() -> None:
    """Constructor binds a real port and surfaces it through topology_endpoint."""
    _, factory = _make_factory()
    server = EgodriverServer(
        name="driver-0",
        max_concurrent_rollouts=2,
        policy_factory=factory,
    )

    try:
        endpoint = server.topology_endpoint
        assert server.port > 0
        assert endpoint.id == "driver-0"
        assert endpoint.host == "localhost"
        assert endpoint.port == server.port
        assert endpoint.capacity == 2
    finally:
        server.stop()


def test_egodriver_server_publishes_configured_host_on_real_port() -> None:
    """Distributed workers publish the reachable host while binding a real port."""
    _, factory = _make_factory()
    server = EgodriverServer(
        name="driver-remote",
        max_concurrent_rollouts=1,
        policy_factory=factory,
        publish_host="worker-1.example",
    )

    try:
        endpoint = server.topology_endpoint
        assert server.port > 0
        assert endpoint.id == "driver-remote"
        assert endpoint.host == "worker-1.example"
        assert endpoint.port == server.port
        assert endpoint.capacity == 1
    finally:
        server.stop()


def test_start_session_invokes_policy_factory_with_calibrated_context() -> None:
    """Factory receives `(session_uuid, calibration, random_seed)` from the request."""
    created, factory = _make_factory()
    servicer = EgodriverGrpcServicer(policy_factory=factory)

    request = SimpleNamespace(
        session_uuid="session-1",
        random_seed=123,
        rollout_spec=_make_rollout_spec(camera_logical_ids=("front_wide", "rear")),
    )
    servicer.start_session(request, context=None)

    assert len(created) == 1
    policy = created[0]
    assert policy.session_uuid == "session-1"
    assert policy.random_seed == 123
    assert tuple(camera.logical_id for camera in policy.calibration) == (
        "front_wide",
        "rear",
    )
    stored = servicer._sessions["session-1"]
    assert isinstance(stored, _Session)
    assert stored.policy is policy


def test_drive_pipeline_passes_buffered_observations_into_policy_step() -> None:
    """Submit RPCs populate the tick buffer; drive snapshots them into PolicyInput."""
    created, factory = _make_factory()
    server = EgodriverServer(
        name="driver-0",
        max_concurrent_rollouts=1,
        policy_factory=factory,
    )
    servicer = server._servicer
    servicer.start_session(
        SimpleNamespace(
            session_uuid="session-1",
            random_seed=0,
            rollout_spec=_make_rollout_spec(),
        ),
        context=None,
    )
    camera_image = RolloutCameraImage.CameraImage(
        logical_id="front",
        image_bytes=b"jpeg-bytes",
        frame_end_us=99,
    )
    servicer.submit_image_observation(
        SimpleNamespace(session_uuid="session-1", camera_image=camera_image),
        context=None,
    )
    servicer.submit_egomotion_observation(
        SimpleNamespace(session_uuid="session-1", trajectory=_proto_trajectory(80)),
        context=None,
    )
    servicer.submit_egomotion_observation(
        SimpleNamespace(session_uuid="session-1", trajectory=_proto_trajectory(90)),
        context=None,
    )
    servicer.submit_route(
        SimpleNamespace(
            session_uuid="session-1",
            route=Route(
                timestamp_us=85,
                waypoints=[SimpleNamespace(x=1.0, y=2.0, z=0.0)],
            ),
        ),
        context=None,
    )

    response = servicer.drive(
        SimpleNamespace(
            session_uuid="session-1", time_now_us=1_000, time_query_us=2_000
        ),
        context=None,
    )

    policy = created[0]
    assert len(policy.received_inputs) == 1
    captured = policy.received_inputs[0]
    assert captured.step_index == 0
    assert captured.time_now_us == 1_000
    assert captured.time_query_us == 2_000
    assert len(captured.camera_images) == 1
    assert captured.camera_images[0].logical_id == "front"
    assert captured.camera_images[0].image_bytes == b"jpeg-bytes"
    assert [pose.timestamp_us for pose in captured.ego_trajectory.poses] == [80, 90]
    timestamps = [pose.timestamp_us for pose in response.trajectory.poses]
    assert timestamps == [1_000, 1_100, 1_200]
    assert servicer._sessions["session-1"].step_index == 1
    assert servicer._sessions["session-1"].tick_buffer.camera_images == []
    assert servicer._sessions["session-1"].tick_buffer.ego_trajectory is None
    server.stop()


def test_close_session_drops_session_and_freezes_record_for_streaming_worker() -> None:
    """Closing the session drops it and freezes its record for `pop_session_record`."""
    created, factory = _make_factory()
    server = EgodriverServer(
        name="driver-0",
        max_concurrent_rollouts=1,
        policy_factory=factory,
    )
    try:
        server._servicer.start_session(
            SimpleNamespace(
                session_uuid="session-1",
                random_seed=0,
                rollout_spec=_make_rollout_spec(),
            ),
            context=None,
        )
        server._servicer.close_session(
            SimpleNamespace(session_uuid="session-1"),
            context=None,
        )

        assert "session-1" not in server._servicer._sessions
        assert created[0].closed is True
        assert server._servicer.pop_session_record("session-1") is not None
    finally:
        server.stop()


def test_close_session_raises_keyerror_for_unknown_session_uuid() -> None:
    """`close_session` fails fast when AlpaSim closes a session that never started."""
    _, factory = _make_factory()
    server = EgodriverServer(
        name="driver-0",
        max_concurrent_rollouts=1,
        policy_factory=factory,
    )
    try:
        with pytest.raises(KeyError):
            server._servicer.close_session(
                SimpleNamespace(session_uuid="never-started"),
                context=None,
            )
    finally:
        server.stop()


def test_submit_recording_ground_truth_stores_on_session_record() -> None:
    """`submit_recording_ground_truth` writes `_Session.ground_truth`; `close_session` reads it."""
    _, factory = _make_factory()
    server = EgodriverServer(
        name="driver-0",
        max_concurrent_rollouts=1,
        policy_factory=factory,
    )
    servicer = server._servicer
    try:
        servicer.start_session(
            SimpleNamespace(
                session_uuid="session-1",
                random_seed=0,
                rollout_spec=_make_rollout_spec(),
            ),
            context=None,
        )
        gt_proto = ProtoGroundTruth(
            trajectory=ProtoTrajectory(
                poses=[
                    SimpleNamespace(
                        timestamp_us=42,
                        pose=SimpleNamespace(
                            vec=SimpleNamespace(x=5.0, y=6.0, z=7.0),
                            quat=SimpleNamespace(w=1.0, x=0.0, y=0.0, z=0.0),
                        ),
                    )
                ]
            ),
            timestamp_us=12345,
        )

        servicer.submit_recording_ground_truth(
            SimpleNamespace(session_uuid="session-1", ground_truth=gt_proto),
            context=None,
        )

        stored = servicer._sessions["session-1"].ground_truth
        assert stored is not None
        assert stored.timestamp_us == 12345
        assert stored.ego_trajectory.poses[0].pose.vec.z == 7.0

        servicer.close_session(SimpleNamespace(session_uuid="session-1"), context=None)
        record = servicer.pop_session_record("session-1")
        assert record.ground_truth is stored
    finally:
        server.stop()


def test_concurrent_sessions_keep_per_session_state_isolated() -> None:
    """Two interleaved sessions remain isolated through drive() callbacks and close_session."""
    created, factory = _make_factory()
    server = EgodriverServer(
        name="driver-0",
        max_concurrent_rollouts=2,
        policy_factory=factory,
    )
    servicer = server._servicer
    try:
        for session_uuid in ("session-1", "session-2"):
            servicer.start_session(
                SimpleNamespace(
                    session_uuid=session_uuid,
                    random_seed=0,
                    rollout_spec=_make_rollout_spec(),
                ),
                context=None,
            )

        # Interleaved observation submissions: each session's tick buffer should
        # only see its own egomotion entries.
        servicer.submit_egomotion_observation(
            SimpleNamespace(session_uuid="session-1", trajectory=_proto_trajectory(80)),
            context=None,
        )
        servicer.submit_egomotion_observation(
            SimpleNamespace(session_uuid="session-2", trajectory=_proto_trajectory(90)),
            context=None,
        )
        for session_uuid, timestamp_us, x in (
            ("session-1", 81, 1.0),
            ("session-2", 91, 2.0),
        ):
            servicer.submit_route(
                SimpleNamespace(
                    session_uuid=session_uuid,
                    route=Route(
                        timestamp_us=timestamp_us,
                        waypoints=[SimpleNamespace(x=x, y=0.0, z=0.0)],
                    ),
                ),
                context=None,
            )

        # Both drive ticks succeed independently and bump only their own step_index.
        servicer.drive(
            SimpleNamespace(
                session_uuid="session-1", time_now_us=1_000, time_query_us=2_000
            ),
            context=None,
        )
        servicer.drive(
            SimpleNamespace(
                session_uuid="session-2", time_now_us=3_000, time_query_us=4_000
            ),
            context=None,
        )
        assert servicer._sessions["session-1"].step_index == 1
        assert servicer._sessions["session-2"].step_index == 1
        assert created[0].received_inputs[0].route_timestamp_us == 81
        assert created[0].received_inputs[0].route_waypoints[0].x == 1.0
        assert created[1].received_inputs[0].route_timestamp_us == 91
        assert created[1].received_inputs[0].route_waypoints[0].x == 2.0

        # Close in mixed order; both records remain pop-able by uuid.
        servicer.close_session(SimpleNamespace(session_uuid="session-2"), context=None)
        servicer.close_session(SimpleNamespace(session_uuid="session-1"), context=None)
        record_1 = servicer.pop_session_record("session-1")
        record_2 = servicer.pop_session_record("session-2")
        assert len(record_1.outputs) == 1
        assert len(record_2.outputs) == 1
    finally:
        server.stop()


def _ego_pose(timestamp_us: int, x: float = 0.0, y: float = 0.0) -> EgoPose:
    """Build an `EgoPose` at `timestamp_us` with the given XY position."""
    return EgoPose(timestamp_us=timestamp_us, pose=Pose(vec=Vec3(x=x, y=y, z=0.0)))


def _policy_input(ego_poses: tuple[EgoPose, ...] = ()) -> PolicyInput:
    """Build a tiny `PolicyInput` with the given executed ego poses."""
    return PolicyInput(
        step_index=0,
        time_now_us=0,
        time_query_us=0,
        camera_images=(),
        ego_trajectory=Trajectory(poses=ego_poses),
        route_waypoints=(RouteWaypoint(x=0.0, y=0.0),),
        route_timestamp_us=0,
        calibration=(),
    )


def _policy_output(value: float) -> PolicyOutput:
    """Build a tiny `PolicyOutput` whose tensors carry `value` for identification."""
    return PolicyOutput(
        chosen_xyz=torch.tensor(
            [[0.0, 0.0, 0.0], [value, value, value]], dtype=torch.float32
        ),
        chosen_quat=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=torch.float32
        ),
        chosen_dt_us=torch.tensor([0, 100], dtype=torch.int64),
    )


def test_humanoid_policy_server_saves_camera_images_from_policy_options(
    tmp_path: Path,
) -> None:
    servicer = HumanoidPolicyGrpcServicer(
        policy_factory=lambda session_uuid, request: ZeroHumanoidPolicy(
            int(request.action_size)
        )
    )
    servicer.reserve_session("session/1", behavior_policy_version=7)
    servicer.start_session(
        SimpleNamespace(
            session_uuid="session/1",
            action_size=1,
            observation_schema="test.v1",
            observation_terms=[SimpleNamespace(name="test", size=1)],
            policy_options={"save_camera_dir": str(tmp_path)},
        ),
        context=None,
    )

    response = servicer.act(
        SimpleNamespace(
            session_uuid="session/1",
            observation=SimpleNamespace(
                env_states=[
                    SimpleNamespace(
                        env_id=0,
                        reset_id=0,
                        timestamp_us=20_000,
                        qpos=[],
                        qvel=[],
                        observation=[0.0],
                        observation_schema="test.v1",
                        named_observations=[
                            SimpleNamespace(name="test", values=[0.0], shape=[1])
                        ],
                        scalars={},
                    )
                ],
                camera_images=[
                    SimpleNamespace(
                        frame_start_us=20_000,
                        frame_end_us=40_000,
                        env_id=0,
                        logical_id="front/camera",
                        image_bytes=b"\xff\xd8fake-jpeg",
                    )
                ],
            ),
            bootstrap_only=False,
            bootstrap_env_ids=[],
        ),
        context=None,
    )

    assert len(response.actions) == 1
    assert response.behavior_policy_version == "7"
    saved = list((tmp_path / "session_1").glob("*.jpg"))
    assert [path.name for path in saved] == [
        "frame_000000020000_env000_00_front_camera.jpg"
    ]
    assert saved[0].read_bytes() == b"\xff\xd8fake-jpeg"


class _MotionPlannerPolicy:
    """Deterministic reference planner used to exercise the full RPC lifecycle."""

    def step(self, policy_inputs, *, sample_actions: bool = True):
        outputs = []
        for item in policy_inputs:
            if not sample_actions:
                outputs.append(
                    HumanoidPolicyStepOutput(
                        env_id=item.env_id,
                        value=(torch.tensor(3.0) if item.bootstrap_requested else None),
                    )
                )
                continue
            digest = ("a" if item.decision_id == 0 else "b") * 64
            root = item.qpos[:3].clone()
            root[2] += 0.125
            frames = tuple(
                HumanoidMotionReferenceFrame(
                    timestamp_us=item.timestamp_us + index * 20_000,
                    joint_position=item.qpos[7:].clone(),
                    joint_velocity=item.qvel[6:].clone(),
                    root_position=root.clone(),
                    root_quaternion_wxyz=item.qpos[3:7].clone(),
                )
                for index in range(50)
            )
            reference = HumanoidMotionReference(
                reference_id=100 + item.decision_id,
                source_decision_id=item.decision_id,
                frames=frames,
                reference_sha256=digest,
                root_z_alignment_offset_m=0.125,
            )
            replay = PolicyReplayData(
                replay_schema_version=1,
                payload_schema="motion.test.v1",
                payload_schema_version=1,
                model_family="motion-test",
                action_selection=ActionSelection(0, 0),
                old_logprob=torch.tensor(0.0),
                payload={"trace": torch.zeros(1)},
            )
            outputs.append(
                HumanoidPolicyStepOutput(
                    env_id=item.env_id,
                    motion_reference=reference,
                    logprob=torch.tensor(0.0),
                    value=torch.tensor(1.0),
                    replay_data=replay,
                )
            )
        return tuple(outputs)

    def close(self) -> None:
        return None


def _motion_state(timestamp_us: int) -> SimpleNamespace:
    qpos = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0] + [0.0] * 29
    qvel = [0.0] * 35
    return SimpleNamespace(
        env_id=0,
        reset_id=9,
        timestamp_us=timestamp_us,
        qpos=qpos,
        qvel=qvel,
        observation=[0.0, 0.0],
        observation_schema="videomimic_motion_planner_state.v1",
        named_observations=[
            SimpleNamespace(
                name="navigation_position_xy",
                values=[0.0, 0.0],
                shape=[2],
            )
        ],
        scalars={},
    )


def _motion_feedback(
    *,
    source_decision_id: int,
    reference_id: int,
    digest: str,
    start_timestamp_us: int,
    first_control_step: int,
) -> SimpleNamespace:
    ticks = []
    for index in range(5):
        timestamp_us = start_timestamp_us + index * 20_000
        ticks.append(
            SimpleNamespace(
                control_tick_offset=index + 1,
                reference_action_index=index,
                state=_motion_state(timestamp_us),
                active_reference_id=reference_id,
                active_reference_sha256=digest,
                applied_reference_sha256=digest,
                root_z_alignment_offset_m=0.125,
                reward=0.1,
                terminated=False,
                truncated=False,
                metrics={},
                control_episode_step=first_control_step + index,
            )
        )
    return SimpleNamespace(
        env_id=0,
        source_decision_id=source_decision_id,
        ticks=ticks,
    )


def _motion_act_request(
    *,
    decision_id: int,
    timestamp_us: int,
    request_kind: int,
    feedback_traces: list[SimpleNamespace],
    bootstrap_env_ids: list[int] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        session_uuid="motion-session",
        request_kind=request_kind,
        bootstrap_only=False,
        bootstrap_env_ids=list(bootstrap_env_ids or []),
        observation=SimpleNamespace(
            decision_id=decision_id,
            env_states=[_motion_state(timestamp_us)],
            camera_images=[],
            feedback_traces=feedback_traces,
        ),
    )


def test_motion_reference_policy_rpc_joins_feedback_and_finalizes_without_extra_plan() -> (
    None
):
    servicer = HumanoidPolicyGrpcServicer(
        policy_factory=lambda session_uuid, request: _MotionPlannerPolicy()
    )
    servicer.reserve_session("motion-session", behavior_policy_version=11)
    servicer.start_session(
        SimpleNamespace(
            session_uuid="motion-session",
            random_seed=7,
            action_size=0,
            execution_mode=2,
            joint_names=list(MOTION_REFERENCE_JOINT_NAMES),
            observation_schema="videomimic_motion_planner_state.v1",
            action_schema="g1_motion_reference_29d_50hz_h50.v1",
            observation_terms=[SimpleNamespace(name="navigation_position_xy", size=2)],
            reference_spec=SimpleNamespace(
                schema="g1_motion_reference_29d_50hz_h50.v1",
                joint_names=list(MOTION_REFERENCE_JOINT_NAMES),
                frame_count=50,
                sample_period_us=20_000,
                control_ticks_per_policy_step=5,
            ),
            policy_options={},
            attempt_id="attempt",
            scene_id="hq_stairs",
            scenario_id="ascend",
        ),
        context=None,
    )

    initial = servicer.act(
        _motion_act_request(
            decision_id=0,
            timestamp_us=0,
            request_kind=2,
            feedback_traces=[],
        ),
        context=None,
    )
    assert len(initial.plan_updates) == 1
    assert len(initial.actions) == 0
    first = initial.plan_updates[0]

    replanned = servicer.act(
        _motion_act_request(
            decision_id=1,
            timestamp_us=100_000,
            request_kind=3,
            feedback_traces=[
                _motion_feedback(
                    source_decision_id=0,
                    reference_id=first.reference_id,
                    digest=first.reference_sha256,
                    start_timestamp_us=20_000,
                    first_control_step=1,
                )
            ],
        ),
        context=None,
    )
    assert len(replanned.plan_updates) == 1
    second = replanned.plan_updates[0]

    finalized = servicer.act(
        _motion_act_request(
            decision_id=2,
            timestamp_us=200_000,
            request_kind=4,
            feedback_traces=[
                _motion_feedback(
                    source_decision_id=1,
                    reference_id=second.reference_id,
                    digest=second.reference_sha256,
                    start_timestamp_us=120_000,
                    first_control_step=6,
                )
            ],
            bootstrap_env_ids=[0],
        ),
        context=None,
    )
    assert len(finalized.plan_updates) == 0
    assert finalized.value_estimates[0].value == pytest.approx(3.0)

    servicer.close_session(
        SimpleNamespace(session_uuid="motion-session"),
        context=None,
    )
    record = servicer.pop_session_record("motion-session")
    assert len(record.outputs) == 2
    assert record.final_bootstrap_values == {0: 3.0}
    assert "feedback_trace" in record.outputs[0].replay_data.payload
    assert "feedback_trace" in record.outputs[1].replay_data.payload
    assert record.outputs[1].replay_data.payload["outer_truncated"] is True


def test_session_record_step_appends_outputs_and_dedupes_executed_poses() -> None:
    """`record_step` keeps outputs in call order and drops already-recorded poses across ticks."""
    session = _Session(calibration=(), policy=_StubPolicy("session-a", (), 0))
    out_a, out_b, out_c = _policy_output(1.0), _policy_output(2.0), _policy_output(3.0)

    session.record_step(
        _policy_input(ego_poses=(_ego_pose(10, x=1.0), _ego_pose(20, x=2.0))), out_a
    )
    session.record_step(
        _policy_input(
            ego_poses=(_ego_pose(10, x=1.0), _ego_pose(20, x=2.0), _ego_pose(30, x=3.0))
        ),
        out_b,
    )
    session.record_step(
        _policy_input(ego_poses=(_ego_pose(50, x=5.0), _ego_pose(40, x=4.0))),
        out_c,
    )

    assert session.outputs == [out_a, out_b, out_c]
    assert session.executed_poses == [
        _ego_pose(10, x=1.0),
        _ego_pose(20, x=2.0),
        _ego_pose(30, x=3.0),
        _ego_pose(40, x=4.0),
        _ego_pose(50, x=5.0),
    ]


def test_session_record_step_skips_empty_ego_trajectories() -> None:
    """Ticks with no ego trajectory contribute no pose but still append the output."""
    session = _Session(calibration=(), policy=_StubPolicy("session-a", (), 0))
    session.record_step(_policy_input(ego_poses=()), _policy_output(1.0))
    session.record_step(
        _policy_input(ego_poses=(_ego_pose(10, x=1.0),)), _policy_output(2.0)
    )
    session.record_step(_policy_input(ego_poses=()), _policy_output(3.0))

    assert len(session.outputs) == 3
    assert session.executed_poses == [_ego_pose(10, x=1.0)]


def test_close_session_freezes_session_record_for_runner_to_drain() -> None:
    """`close_session` freezes outputs/executed_poses/GT for the runner to drain."""
    _, factory = _make_factory()
    server = EgodriverServer(
        name="driver-0",
        max_concurrent_rollouts=1,
        policy_factory=factory,
    )
    servicer = server._servicer
    try:
        servicer.start_session(
            SimpleNamespace(
                session_uuid="session-1",
                random_seed=0,
                rollout_spec=_make_rollout_spec(),
            ),
            context=None,
        )
        session = servicer._sessions["session-1"]
        out_a, out_b = _policy_output(1.0), _policy_output(2.0)
        session.record_step(_policy_input(ego_poses=(_ego_pose(10, x=1.0),)), out_a)
        session.record_step(_policy_input(ego_poses=(_ego_pose(20, x=2.0),)), out_b)

        servicer.close_session(SimpleNamespace(session_uuid="session-1"), context=None)

        record = servicer.pop_session_record("session-1")
        assert record == SessionRecord(
            outputs=(out_a, out_b),
            executed_ego_trajectory=Trajectory(
                poses=(_ego_pose(10, x=1.0), _ego_pose(20, x=2.0))
            ),
            ground_truth=None,
        )
        with pytest.raises(KeyError):
            servicer.pop_session_record("session-1")
    finally:
        server.stop()
