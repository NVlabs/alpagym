# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import io
import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from alpagym_runtime.alpasim.tests.test_proto_conversion import (
    install_alpasim_grpc_stubs,
)

install_alpasim_grpc_stubs()

import pytest  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from alpagym_runtime.alpasim.driver_server import (  # noqa: E402
    EgodriverGrpcServicer,
    EgodriverServer,
    SessionRecord,
    _Session,
)
from alpagym_runtime.alpasim.humanoid_policy_server import (  # noqa: E402
    HUMANOID_FEEDBACK_STATE_CONTRACT_SCHEMA,
    HUMANOID_VISUAL_INPUT_MANIFEST_SCHEMA,
    MOTION_REFERENCE_JOINT_NAMES,
    HumanoidCameraFrameIdentity,
    HumanoidMotionReference,
    HumanoidMotionReferenceFrame,
    HumanoidReferenceDecodeContext,
    HumanoidPolicyCameraContract,
    HumanoidCameraFrame,
    HumanoidPolicyGrpcServicer,
    HumanoidPolicyInput,
    HumanoidPolicyStepOutput,
    ZeroHumanoidPolicy,
    _camera_frames_by_env,
    _camera_frame_identity,
    _encoded_image_size,
    _policy_camera_contract,
    _recorded_policy_output,
    _route_camera_frames,
    _save_camera_images,
    _validate_recorded_visual_input_manifest,
    _validate_motion_reference_session,
    humanoid_model_input_tensor_sha256,
    humanoid_visual_input_manifest_sha256,
    _plan_update_from_output,
)
from alpagym_runtime.inference.inference_engine import InferenceModelLease  # noqa: E402
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
from alpasim_grpc.v0.humanoid_contracts import (  # noqa: E402
    HUMANOID_FULL_ROTATION_LOCAL_XY_DECODE_CONTEXT_SCHEMA,
    HUMANOID_RENDER_STATE_SCHEMA,
    HumanoidRenderState,
    humanoid_render_receipt_v2_sha256,
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


class _RecordingZeroHumanoidPolicy(ZeroHumanoidPolicy):
    """No-op humanoid policy that retains each typed policy input call."""

    def __init__(self, action_size: int) -> None:
        """Initialize the zero policy and its call log."""
        super().__init__(action_size)
        self.calls = []

    def step(self, policy_inputs, *, sample_actions: bool = True):
        """Record immutable lane inputs before returning zero actions."""
        self.calls.append((policy_inputs, sample_actions))
        return super().step(policy_inputs, sample_actions=sample_actions)


def test_humanoid_policy_server_saves_camera_images_from_policy_options(
    tmp_path: Path,
) -> None:
    policy = _RecordingZeroHumanoidPolicy(action_size=1)
    servicer = HumanoidPolicyGrpcServicer(
        policy_factory=lambda session_uuid, request: policy
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
                timestamp_us=20_000,
                decision_id=0,
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
                        frame_start_us=10_000,
                        frame_end_us=20_000,
                        env_id=0,
                        logical_id="front/camera",
                        image_bytes=b"\xff\xd8fake-jpeg",
                        render_timestamp_us=0,
                        observation_decision_id=0,
                        render_qpos=[],
                        render_state_sha256="",
                        camera_contract_sha256="",
                        image_sha256="",
                        render_receipt_sha256="",
                        scene_fingerprint="",
                        model_signature_sha256="",
                        camera_to_world_sha256="",
                        renderer_binding_sha256="",
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
    image_sha256 = hashlib.sha256(b"\xff\xd8fake-jpeg").hexdigest()
    assert [path.name for path in saved] == [
        "frame_000000010000_decision_000000000000_env000_00_front_camera_"
        f"render_none_image_{image_sha256}.jpg"
    ]
    assert saved[0].read_bytes() == b"\xff\xd8fake-jpeg"
    camera_frame = policy.calls[0][0][0].camera_frames[0]
    assert isinstance(camera_frame, HumanoidCameraFrame)
    assert camera_frame == HumanoidCameraFrame(
        env_id=0,
        frame_start_us=10_000,
        frame_end_us=20_000,
        logical_id="front/camera",
        image_bytes=b"\xff\xd8fake-jpeg",
        render_timestamp_us=0,
        observation_decision_id=0,
        render_qpos=(),
        render_state_sha256="",
        camera_contract_sha256="",
        image_sha256="",
        render_receipt_sha256="",
        scene_fingerprint="",
        model_signature_sha256="",
        camera_to_world_sha256="",
        renderer_binding_sha256="",
    )
    with pytest.raises(FrozenInstanceError):
        setattr(camera_frame, "logical_id", "rewritten")
    _save_camera_images(
        [replace(camera_frame, image_bytes=b"\xff\xd8other-jpeg")],
        tmp_path / "session_1",
    )
    assert len(list((tmp_path / "session_1").glob("*.jpg"))) == 2

    servicer.close_session(
        SimpleNamespace(session_uuid="session/1"),
        context=None,
    )
    record = servicer.pop_session_record("session/1")
    camera_metadata = record.outputs[0].model_extra["humanoid_camera_frames"]
    assert camera_metadata == [
        {
            "env_id": 0,
            "frame_start_us": 10_000,
            "frame_end_us": 20_000,
            "logical_id": "front/camera",
            "byte_length": len(b"\xff\xd8fake-jpeg"),
            "sha256": hashlib.sha256(b"\xff\xd8fake-jpeg").hexdigest(),
            "render_timestamp_us": 0,
            "observation_decision_id": 0,
            "render_state_sha256": "",
            "camera_contract_sha256": "",
            "image_sha256": "",
            "render_receipt_sha256": "",
            "scene_fingerprint": "",
            "model_signature_sha256": "",
            "camera_to_world_sha256": "",
            "renderer_binding_sha256": "",
        }
    ]
    json.dumps(record.outputs[0].model_extra)
    assert record.outputs[0].replay_data is None


def _camera_image(
    *,
    env_id: int = 0,
    frame_start_us: int = 0,
    frame_end_us: int = 10_000,
    logical_id: str = "d455",
    image_bytes: bytes = b"rgb",
    render_timestamp_us: int = 0,
    observation_decision_id: int = 0,
    render_qpos: list[float] | None = None,
    render_state_sha256: str = "",
    camera_contract_sha256: str = "",
    image_sha256: str = "",
    render_receipt_sha256: str = "",
    scene_fingerprint: str = "",
    model_signature_sha256: str = "",
    camera_to_world_sha256: str = "",
    renderer_binding_sha256: str = "",
) -> SimpleNamespace:
    """Build a proto-like humanoid camera packet for server tests."""
    return SimpleNamespace(
        env_id=env_id,
        frame_start_us=frame_start_us,
        frame_end_us=frame_end_us,
        logical_id=logical_id,
        image_bytes=image_bytes,
        render_timestamp_us=render_timestamp_us,
        observation_decision_id=observation_decision_id,
        render_qpos=list(render_qpos or []),
        render_state_sha256=render_state_sha256,
        camera_contract_sha256=camera_contract_sha256,
        image_sha256=image_sha256,
        render_receipt_sha256=render_receipt_sha256,
        scene_fingerprint=scene_fingerprint,
        model_signature_sha256=model_signature_sha256,
        camera_to_world_sha256=camera_to_world_sha256,
        renderer_binding_sha256=renderer_binding_sha256,
    )


def test_humanoid_camera_frames_are_grouped_by_exact_env_lane() -> None:
    grouped = _camera_frames_by_env(
        states=[
            SimpleNamespace(env_id=3, timestamp_us=20_000),
            SimpleNamespace(env_id=7, timestamp_us=20_000),
        ],
        camera_images=[
            _camera_image(
                env_id=7,
                frame_start_us=10_000,
                frame_end_us=20_000,
                logical_id="rear",
                image_bytes=b"rear",
            ),
            _camera_image(env_id=3, logical_id="front", image_bytes=b"front"),
            _camera_image(
                env_id=7,
                frame_start_us=5_000,
                frame_end_us=20_000,
                logical_id="front",
                image_bytes=b"front-late",
            ),
            _camera_image(env_id=7, logical_id="front", image_bytes=b"front-old"),
            _camera_image(
                env_id=7,
                frame_start_us=0,
                frame_end_us=20_000,
                logical_id="front",
                image_bytes=b"front-early",
            ),
        ],
        observation_timestamp_us=20_000,
        observation_decision_id=0,
        joint_names=(),
        policy_camera_contract=None,
    )

    assert tuple(grouped) == (3, 7)
    assert [frame.logical_id for frame in grouped[3]] == ["front"]
    assert [
        (frame.frame_end_us, frame.logical_id, frame.frame_start_us)
        for frame in grouped[7]
    ] == [
        (10_000, "front", 0),
        (20_000, "front", 0),
        (20_000, "front", 5_000),
        (20_000, "rear", 10_000),
    ]
    assert all(frame.env_id == 3 for frame in grouped[3])
    assert all(frame.env_id == 7 for frame in grouped[7])


def test_generic_camera_transport_accepts_legacy_base_packet() -> None:
    legacy_image = SimpleNamespace(
        env_id=0,
        frame_start_us=10_000,
        frame_end_us=20_000,
        logical_id="legacy-camera",
        image_bytes=b"legacy-encoded-bytes",
    )

    frame = _camera_frames_by_env(
        states=[SimpleNamespace(env_id=0, timestamp_us=20_000)],
        camera_images=[legacy_image],
        observation_timestamp_us=20_000,
        observation_decision_id=3,
        joint_names=(),
        policy_camera_contract=None,
    )[0][0]

    assert frame.render_qpos == ()
    assert frame.render_state_sha256 == ""
    assert frame.camera_contract_sha256 == ""
    assert frame.image_sha256 == ""
    assert frame.render_receipt_sha256 == ""
    assert frame.scene_fingerprint == ""
    assert frame.model_signature_sha256 == ""
    assert frame.camera_to_world_sha256 == ""
    assert frame.renderer_binding_sha256 == ""


@pytest.mark.parametrize(
    ("camera_image", "error"),
    (
        (_camera_image(env_id=1), "unknown env_id=1"),
        (
            _camera_image(frame_start_us=20_000, frame_end_us=10_000),
            "non-negative interval",
        ),
        (
            _camera_image(frame_start_us=10_000, frame_end_us=20_000),
            "must not exceed its env state timestamp",
        ),
        (_camera_image(logical_id="  "), "logical_id must be non-empty"),
        (_camera_image(image_bytes=b""), "image_bytes must be non-empty"),
    ),
)
def test_humanoid_camera_frames_reject_invalid_packets(
    camera_image: SimpleNamespace,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        _camera_frames_by_env(
            states=[SimpleNamespace(env_id=0, timestamp_us=10_000)],
            camera_images=[camera_image],
            observation_timestamp_us=10_000,
            observation_decision_id=0,
            joint_names=(),
            policy_camera_contract=None,
        )


def test_humanoid_camera_frames_reject_duplicate_lane_identity() -> None:
    duplicate = _camera_image()
    with pytest.raises(ValueError, match="duplicate camera frame"):
        _camera_frames_by_env(
            states=[SimpleNamespace(env_id=0, timestamp_us=10_000)],
            camera_images=[duplicate, duplicate],
            observation_timestamp_us=10_000,
            observation_decision_id=0,
            joint_names=(),
            policy_camera_contract=None,
        )


def _strict_camera_fixture() -> tuple[
    SimpleNamespace,
    SimpleNamespace,
    HumanoidPolicyCameraContract,
    tuple[str, ...],
]:
    """Build one valid image-paired G1 camera packet and its session contract."""
    joint_names = tuple(f"joint_{index}" for index in range(29))
    render_qpos = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0] + [
        float(index) / 10.0 for index in range(29)
    ]
    contract = HumanoidPolicyCameraContract(
        schema="humanoid_policy_camera_rgb_qpos.v1",
        logical_id="d455",
        width=224,
        height=140,
        image_format="png",
        max_frame_age_us=0,
        contract_sha256="c" * 64,
    )
    state = SimpleNamespace(env_id=0, timestamp_us=20_000, qpos=render_qpos)
    image_buffer = io.BytesIO()
    Image.new("RGB", (224, 140)).save(image_buffer, format="PNG")
    image_bytes = image_buffer.getvalue()
    image_sha256 = hashlib.sha256(image_bytes).hexdigest()
    renderer_evidence = {
        "scene_fingerprint": "1" * 64,
        "model_signature_sha256": "2" * 64,
        "camera_to_world_sha256": "3" * 64,
        "renderer_binding_sha256": "4" * 64,
    }
    render_receipt_sha256 = humanoid_render_receipt_v2_sha256(
        render_state_sha256="e" * 64,
        camera_contract_sha256=contract.contract_sha256,
        image_sha256=image_sha256,
        image_format="png",
        width=224,
        height=140,
        **renderer_evidence,
    )
    image = _camera_image(
        frame_start_us=20_000,
        frame_end_us=20_000,
        image_bytes=image_bytes,
        render_timestamp_us=20_000,
        observation_decision_id=7,
        render_qpos=render_qpos,
        render_state_sha256="e" * 64,
        camera_contract_sha256=contract.contract_sha256,
        image_sha256=image_sha256,
        render_receipt_sha256=render_receipt_sha256,
        **renderer_evidence,
    )
    return state, image, contract, joint_names


def test_strict_policy_camera_exposes_image_paired_joint_state() -> None:
    state, image, contract, joint_names = _strict_camera_fixture()
    frames = _camera_frames_by_env(
        states=[state],
        camera_images=[image],
        observation_timestamp_us=20_000,
        observation_decision_id=7,
        joint_names=joint_names,
        policy_camera_contract=contract,
    )

    frame = frames[0][0]
    assert frame.policy_joint_position == tuple(state.qpos[7:])
    assert len(frame.policy_joint_position) == 29
    assert frame.render_state_sha256 == image.render_state_sha256
    assert frame.camera_contract_sha256 == contract.contract_sha256
    assert frame.image_sha256 == image.image_sha256
    assert frame.render_receipt_sha256 == image.render_receipt_sha256
    assert frame.scene_fingerprint == image.scene_fingerprint
    assert frame.model_signature_sha256 == image.model_signature_sha256
    assert frame.camera_to_world_sha256 == image.camera_to_world_sha256
    assert frame.renderer_binding_sha256 == image.renderer_binding_sha256
    recorded = _recorded_policy_output(
        step_index=0,
        policy_input=HumanoidPolicyInput(
            session_uuid="strict-camera",
            episode_id=1,
            step_index=0,
            timestamp_us=20_000,
            env_id=0,
            qpos=torch.tensor(state.qpos),
            qvel=torch.zeros(35),
            observation=torch.zeros(1),
            scalars={},
            camera_frames=(frame,),
            decision_id=7,
        ),
        output=HumanoidPolicyStepOutput(env_id=0, action=torch.zeros(1)),
        action_values=torch.zeros(1),
        motion_reference=None,
    )
    camera_metadata = recorded.model_extra["humanoid_camera_frames"][0]
    assert "render_qpos" not in camera_metadata
    assert "image_bytes" not in camera_metadata
    assert camera_metadata["render_state_sha256"] == image.render_state_sha256
    assert camera_metadata["image_sha256"] == image.image_sha256
    assert camera_metadata["render_receipt_sha256"] == image.render_receipt_sha256
    assert camera_metadata["scene_fingerprint"] == image.scene_fingerprint
    assert camera_metadata["model_signature_sha256"] == image.model_signature_sha256
    assert camera_metadata["camera_to_world_sha256"] == image.camera_to_world_sha256
    assert camera_metadata["renderer_binding_sha256"] == image.renderer_binding_sha256
    json.dumps(recorded.model_extra)


def _strict_visual_manifest_fixture() -> tuple[
    dict[str, object],
    dict[str, object],
    HumanoidPolicyInput,
]:
    """Build one internally consistent visual replay fixture."""

    state, image, contract, joint_names = _strict_camera_fixture()
    frame = _camera_frames_by_env(
        states=[state],
        camera_images=[image],
        observation_timestamp_us=20_000,
        observation_decision_id=7,
        joint_names=joint_names,
        policy_camera_contract=contract,
    )[0][0]
    policy_input = HumanoidPolicyInput(
        session_uuid="strict-camera",
        episode_id=1,
        step_index=0,
        timestamp_us=20_000,
        env_id=0,
        qpos=torch.tensor(state.qpos),
        qvel=torch.zeros(35),
        observation=torch.zeros(1),
        scalars={},
        camera_frames=(frame,),
        decision_id=7,
    )
    pixels = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    instruction_sha256 = "a" * 64
    manifest: dict[str, object] = {
        "schema": HUMANOID_VISUAL_INPUT_MANIFEST_SCHEMA,
        "session_uuid": policy_input.session_uuid,
        "episode_id": policy_input.episode_id,
        "step_index": policy_input.step_index,
        "timestamp_us": policy_input.timestamp_us,
        "env_id": policy_input.env_id,
        "decision_id": policy_input.decision_id,
        "camera_logical_id": frame.logical_id,
        "image_format": contract.image_format,
        "preprocess_profile": "test-native-preprocess.v1",
        "instruction_sha256": instruction_sha256,
        "source_frames": [
            {
                "role": "current",
                "env_id": frame.env_id,
                "frame_start_us": frame.frame_start_us,
                "frame_end_us": frame.frame_end_us,
                "logical_id": frame.logical_id,
                "byte_length": len(frame.image_bytes),
                "render_timestamp_us": frame.render_timestamp_us,
                "observation_decision_id": frame.observation_decision_id,
                "render_state_sha256": frame.render_state_sha256,
                "camera_contract_sha256": frame.camera_contract_sha256,
                "image_sha256": frame.image_sha256,
                "render_receipt_sha256": frame.render_receipt_sha256,
                "scene_fingerprint": frame.scene_fingerprint,
                "model_signature_sha256": frame.model_signature_sha256,
                "camera_to_world_sha256": frame.camera_to_world_sha256,
                "renderer_binding_sha256": frame.renderer_binding_sha256,
            }
        ],
        "pixel_values": {
            "dtype": str(pixels.dtype),
            "shape": list(pixels.shape),
            "sha256": humanoid_model_input_tensor_sha256(pixels),
        },
    }
    payload: dict[str, object] = {
        "instruction_sha256": instruction_sha256,
        "pixel_values": pixels,
        "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.int64),
        "selected_history_indices": torch.empty(0, dtype=torch.int64),
        "visual_input_manifest_sha256": (
            humanoid_visual_input_manifest_sha256(manifest)
        ),
    }
    return {"humanoid_visual_input_manifest": manifest}, payload, policy_input


def _strict_visual_manifest_history_fixture() -> tuple[
    dict[str, object],
    dict[str, object],
    HumanoidPolicyInput,
]:
    """Add one provenance-bound history frame with a valid v2 receipt."""

    model_extra, payload, policy_input = _strict_visual_manifest_fixture()
    manifest = model_extra["humanoid_visual_input_manifest"]
    current = manifest["source_frames"][0]
    with Image.open(io.BytesIO(policy_input.camera_frames[0].image_bytes)) as image:
        width, height = (int(value) for value in image.size)
    history: dict[str, object] = {
        "role": "history",
        "env_id": current["env_id"],
        "frame_start_us": 0,
        "frame_end_us": 0,
        "logical_id": current["logical_id"],
        "byte_length": 123,
        "render_timestamp_us": 0,
        "observation_decision_id": 6,
        "render_state_sha256": "7" * 64,
        "camera_contract_sha256": current["camera_contract_sha256"],
        "image_sha256": "8" * 64,
        "render_receipt_sha256": "",
        "scene_fingerprint": current["scene_fingerprint"],
        "model_signature_sha256": current["model_signature_sha256"],
        "camera_to_world_sha256": "9" * 64,
        "renderer_binding_sha256": "a" * 64,
    }
    history["render_receipt_sha256"] = humanoid_render_receipt_v2_sha256(
        render_state_sha256=history["render_state_sha256"],
        camera_contract_sha256=history["camera_contract_sha256"],
        image_sha256=history["image_sha256"],
        image_format=manifest["image_format"],
        width=width,
        height=height,
        scene_fingerprint=history["scene_fingerprint"],
        model_signature_sha256=history["model_signature_sha256"],
        camera_to_world_sha256=history["camera_to_world_sha256"],
        renderer_binding_sha256=history["renderer_binding_sha256"],
    )
    current["role"] = "current"
    manifest["source_frames"] = [history, current]
    payload["image_grid_thw"] = torch.tensor([[1, 2, 2], [1, 2, 2]], dtype=torch.int64)
    payload["selected_history_indices"] = torch.tensor([0], dtype=torch.int64)
    payload["visual_input_manifest_sha256"] = humanoid_visual_input_manifest_sha256(
        manifest
    )
    return model_extra, payload, policy_input


def _strict_visual_history_identity(
    model_extra: dict[str, object],
) -> HumanoidCameraFrameIdentity:
    """Materialize the prior server-routed identity used by history tests."""

    manifest = model_extra["humanoid_visual_input_manifest"]
    history = manifest["source_frames"][0]
    return HumanoidCameraFrameIdentity(
        env_id=history["env_id"],
        frame_start_us=history["frame_start_us"],
        frame_end_us=history["frame_end_us"],
        logical_id=history["logical_id"],
        byte_length=history["byte_length"],
        sha256=history["image_sha256"],
        render_timestamp_us=history["render_timestamp_us"],
        observation_decision_id=history["observation_decision_id"],
        render_state_sha256=history["render_state_sha256"],
        camera_contract_sha256=history["camera_contract_sha256"],
        image_sha256=history["image_sha256"],
        render_receipt_sha256=history["render_receipt_sha256"],
        scene_fingerprint=history["scene_fingerprint"],
        model_signature_sha256=history["model_signature_sha256"],
        camera_to_world_sha256=history["camera_to_world_sha256"],
        renderer_binding_sha256=history["renderer_binding_sha256"],
    )


def test_visual_input_manifest_records_exact_source_and_pixel_tensor() -> None:
    model_extra, payload, policy_input = _strict_visual_manifest_fixture()
    replay = PolicyReplayData(
        replay_schema_version=1,
        payload_schema="test.visual-replay.v1",
        payload_schema_version=1,
        model_family="test",
        action_selection=ActionSelection(set_ix=0, sample_ix=0),
        old_logprob=None,
        payload=payload,
    )

    recorded = _recorded_policy_output(
        step_index=0,
        policy_input=policy_input,
        output=HumanoidPolicyStepOutput(
            env_id=0,
            action=torch.zeros(1),
            replay_data=replay,
            model_extra=model_extra,
        ),
        action_values=torch.zeros(1),
        motion_reference=None,
    )

    assert recorded.replay_data is not None
    assert recorded.model_extra is not None
    manifest = recorded.model_extra["humanoid_visual_input_manifest"]
    assert manifest["source_frames"][0]["renderer_binding_sha256"] == "4" * 64
    assert json.loads(json.dumps(manifest, sort_keys=True)) == manifest
    assert recorded.replay_data.payload["visual_input_manifest_sha256"] == (
        humanoid_visual_input_manifest_sha256(manifest)
    )


def test_visual_input_manifest_accepts_provenance_bound_history_receipt() -> None:
    """Trusted producer/local transport history passes when all evidence is paired."""

    model_extra, payload, policy_input = _strict_visual_manifest_history_fixture()
    prior_identity = _strict_visual_history_identity(model_extra)

    _validate_recorded_visual_input_manifest(
        model_extra=model_extra,
        payload=payload,
        policy_input=policy_input,
        step_index=0,
        prior_current_frame_identities=(prior_identity,),
    )


@pytest.mark.parametrize(
    "field_name",
    ("camera_to_world_sha256", "renderer_binding_sha256"),
)
def test_visual_input_manifest_rejects_mix_and_match_history_receipt(
    field_name: str,
) -> None:
    """A rehashed manifest cannot hide stale/mix-and-match history evidence."""

    model_extra, payload, policy_input = _strict_visual_manifest_history_fixture()
    manifest = model_extra["humanoid_visual_input_manifest"]
    prior_identity = _strict_visual_history_identity(model_extra)
    history = manifest["source_frames"][0]
    history[field_name] = "b" * 64
    payload["visual_input_manifest_sha256"] = humanoid_visual_input_manifest_sha256(
        manifest
    )

    with pytest.raises(ValueError, match="source frame render receipt is invalid"):
        _validate_recorded_visual_input_manifest(
            model_extra=model_extra,
            payload=payload,
            policy_input=policy_input,
            step_index=0,
            prior_current_frame_identities=(prior_identity,),
        )


def test_visual_input_manifest_rejects_invented_history_with_valid_hashes() -> None:
    """Valid receipts and a rehashed manifest do not replace server provenance."""

    model_extra, payload, policy_input = _strict_visual_manifest_history_fixture()

    with pytest.raises(ValueError, match="previously routed current frame"):
        _validate_recorded_visual_input_manifest(
            model_extra=model_extra,
            payload=payload,
            policy_input=policy_input,
            step_index=0,
        )


def _visual_manifest_entry_from_identity(
    identity: HumanoidCameraFrameIdentity,
    *,
    role: str,
) -> dict[str, object]:
    return {
        "role": role,
        "env_id": identity.env_id,
        "frame_start_us": identity.frame_start_us,
        "frame_end_us": identity.frame_end_us,
        "logical_id": identity.logical_id,
        "byte_length": identity.byte_length,
        "render_timestamp_us": identity.render_timestamp_us,
        "observation_decision_id": identity.observation_decision_id,
        "render_state_sha256": identity.render_state_sha256,
        "camera_contract_sha256": identity.camera_contract_sha256,
        "image_sha256": identity.image_sha256,
        "render_receipt_sha256": identity.render_receipt_sha256,
        "scene_fingerprint": identity.scene_fingerprint,
        "model_signature_sha256": identity.model_signature_sha256,
        "camera_to_world_sha256": identity.camera_to_world_sha256,
        "renderer_binding_sha256": identity.renderer_binding_sha256,
    }


class _VisualHistoryLedgerPolicy:
    """Policy fixture that reuses the prior routed current as visual history."""

    def __init__(self, *, invent_history: bool = False) -> None:
        self._invent_history = invent_history
        self._prior_by_env: dict[int, HumanoidCameraFrameIdentity] = {}

    def step(self, policy_inputs, *, sample_actions: bool = True):
        del sample_actions
        outputs = []
        for policy_input in policy_inputs:
            current_identity = _camera_frame_identity(policy_input.camera_frames[0])
            source_frames = []
            prior = self._prior_by_env.get(policy_input.env_id)
            if prior is not None:
                history = _visual_manifest_entry_from_identity(prior, role="history")
                if self._invent_history:
                    history["camera_to_world_sha256"] = "b" * 64
                    history["render_receipt_sha256"] = (
                        humanoid_render_receipt_v2_sha256(
                            render_state_sha256=history["render_state_sha256"],
                            camera_contract_sha256=history["camera_contract_sha256"],
                            image_sha256=history["image_sha256"],
                            image_format="png",
                            width=224,
                            height=140,
                            scene_fingerprint=history["scene_fingerprint"],
                            model_signature_sha256=history["model_signature_sha256"],
                            camera_to_world_sha256=history["camera_to_world_sha256"],
                            renderer_binding_sha256=history["renderer_binding_sha256"],
                        )
                    )
                source_frames.append(history)
            source_frames.append(
                _visual_manifest_entry_from_identity(current_identity, role="current")
            )
            pixels = torch.arange(12, dtype=torch.float32).reshape(3, 4)
            instruction_sha256 = "a" * 64
            manifest: dict[str, object] = {
                "schema": HUMANOID_VISUAL_INPUT_MANIFEST_SCHEMA,
                "session_uuid": policy_input.session_uuid,
                "episode_id": policy_input.episode_id,
                "step_index": policy_input.step_index,
                "timestamp_us": policy_input.timestamp_us,
                "env_id": policy_input.env_id,
                "decision_id": policy_input.decision_id,
                "camera_logical_id": current_identity.logical_id,
                "image_format": "png",
                "preprocess_profile": "test-native-preprocess.v1",
                "instruction_sha256": instruction_sha256,
                "source_frames": source_frames,
                "pixel_values": {
                    "dtype": str(pixels.dtype),
                    "shape": list(pixels.shape),
                    "sha256": humanoid_model_input_tensor_sha256(pixels),
                },
            }
            history_count = len(source_frames) - 1
            payload = {
                "instruction_sha256": instruction_sha256,
                "pixel_values": pixels,
                "image_grid_thw": torch.tensor(
                    [[1, 2, 2]] * len(source_frames), dtype=torch.int64
                ),
                "selected_history_indices": torch.arange(
                    history_count, dtype=torch.int64
                ),
                "visual_input_manifest_sha256": (
                    humanoid_visual_input_manifest_sha256(manifest)
                ),
            }
            replay = PolicyReplayData(
                replay_schema_version=1,
                payload_schema="test.visual-replay.v1",
                payload_schema_version=1,
                model_family="test",
                action_selection=ActionSelection(set_ix=0, sample_ix=0),
                old_logprob=None,
                payload=payload,
            )
            outputs.append(
                HumanoidPolicyStepOutput(
                    env_id=policy_input.env_id,
                    action=torch.zeros(1),
                    replay_data=replay,
                    model_extra={"humanoid_visual_input_manifest": manifest},
                )
            )
            self._prior_by_env[policy_input.env_id] = current_identity
        return tuple(outputs)

    def close(self) -> None:
        return None


def _visual_ledger_step(
    *,
    timestamp_us: int,
    decision_id: int,
    contract: HumanoidPolicyCameraContract,
    joint_names: tuple[str, ...],
) -> tuple[SimpleNamespace, SimpleNamespace]:
    qpos = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0] + [
        float(index) / 10.0 for index in range(29)
    ]
    state = SimpleNamespace(
        env_id=0,
        reset_id=1,
        timestamp_us=timestamp_us,
        qpos=qpos,
        qvel=[0.0] * 35,
        observation=[0.0],
        observation_schema="test.v1",
        named_observations=[SimpleNamespace(name="test", values=[0.0], shape=[1])],
        scalars={},
    )
    image_buffer = io.BytesIO()
    Image.new("RGB", (contract.width, contract.height)).save(image_buffer, format="PNG")
    image_bytes = image_buffer.getvalue()
    image_sha256 = hashlib.sha256(image_bytes).hexdigest()
    render_state_sha256 = HumanoidRenderState(
        schema=HUMANOID_RENDER_STATE_SCHEMA,
        env_id=0,
        timestamp_us=timestamp_us,
        observation_decision_id=decision_id,
        camera_logical_id=contract.logical_id,
        joint_names=joint_names,
        qpos=qpos,
        camera_contract_sha256=contract.contract_sha256,
    ).canonical_sha256()
    renderer_evidence = {
        "scene_fingerprint": "1" * 64,
        "model_signature_sha256": "2" * 64,
        "camera_to_world_sha256": f"{decision_id + 3:064x}",
        "renderer_binding_sha256": f"{decision_id + 4:064x}",
    }
    receipt = humanoid_render_receipt_v2_sha256(
        render_state_sha256=render_state_sha256,
        camera_contract_sha256=contract.contract_sha256,
        image_sha256=image_sha256,
        image_format=contract.image_format,
        width=contract.width,
        height=contract.height,
        **renderer_evidence,
    )
    image = _camera_image(
        frame_start_us=timestamp_us,
        frame_end_us=timestamp_us,
        logical_id=contract.logical_id,
        image_bytes=image_bytes,
        render_timestamp_us=timestamp_us,
        observation_decision_id=decision_id,
        render_qpos=qpos,
        render_state_sha256=render_state_sha256,
        camera_contract_sha256=contract.contract_sha256,
        image_sha256=image_sha256,
        render_receipt_sha256=receipt,
        **renderer_evidence,
    )
    return state, image


def _visual_ledger_request(
    *,
    timestamp_us: int,
    decision_id: int,
    contract: HumanoidPolicyCameraContract,
    joint_names: tuple[str, ...],
) -> SimpleNamespace:
    state, image = _visual_ledger_step(
        timestamp_us=timestamp_us,
        decision_id=decision_id,
        contract=contract,
        joint_names=joint_names,
    )
    return SimpleNamespace(
        session_uuid="visual-ledger",
        bootstrap_only=False,
        bootstrap_env_ids=[],
        observation=SimpleNamespace(
            timestamp_us=timestamp_us,
            decision_id=decision_id,
            env_states=[state],
            camera_images=[image],
            feedback_traces=[],
        ),
    )


def _visual_ledger_servicer(
    policy: _VisualHistoryLedgerPolicy,
) -> tuple[
    HumanoidPolicyGrpcServicer,
    HumanoidPolicyCameraContract,
    tuple[str, ...],
]:
    _, _, contract, joint_names = _strict_camera_fixture()
    servicer = HumanoidPolicyGrpcServicer(
        policy_factory=lambda session_uuid, request: policy,
        require_policy_camera=True,
    )
    servicer.reserve_session("visual-ledger", behavior_policy_version=1)
    servicer.start_session(
        SimpleNamespace(
            session_uuid="visual-ledger",
            action_size=1,
            joint_names=list(joint_names),
            observation_schema="test.v1",
            observation_terms=[SimpleNamespace(name="test", size=1)],
            policy_camera_spec=SimpleNamespace(
                schema=contract.schema,
                logical_id=contract.logical_id,
                width=contract.width,
                height=contract.height,
                image_format=contract.image_format,
                max_frame_age_us=contract.max_frame_age_us,
                contract_sha256=contract.contract_sha256,
            ),
            policy_options={},
        ),
        context=None,
    )
    return servicer, contract, joint_names


def test_server_camera_ledger_accepts_actual_prior_current_as_history() -> None:
    servicer, contract, joint_names = _visual_ledger_servicer(
        _VisualHistoryLedgerPolicy()
    )

    for timestamp_us, decision_id in ((20_000, 1), (40_000, 2)):
        response = servicer.act(
            _visual_ledger_request(
                timestamp_us=timestamp_us,
                decision_id=decision_id,
                contract=contract,
                joint_names=joint_names,
            ),
            context=None,
        )
        assert len(response.actions) == 1

    servicer.close_session(
        SimpleNamespace(session_uuid="visual-ledger"),
        context=None,
    )
    assert len(servicer.pop_session_record("visual-ledger").outputs) == 2


def test_server_camera_ledger_rejects_invented_valid_receipt_history() -> None:
    servicer, contract, joint_names = _visual_ledger_servicer(
        _VisualHistoryLedgerPolicy(invent_history=True)
    )
    servicer.act(
        _visual_ledger_request(
            timestamp_us=20_000,
            decision_id=1,
            contract=contract,
            joint_names=joint_names,
        ),
        context=None,
    )

    with pytest.raises(ValueError, match="previously routed current frame"):
        servicer.act(
            _visual_ledger_request(
                timestamp_us=40_000,
                decision_id=2,
                contract=contract,
                joint_names=joint_names,
            ),
            context=None,
        )
    servicer.discard_session("visual-ledger")


@pytest.mark.parametrize(
    "tamper",
    (
        "missing_manifest",
        "renderer_evidence",
        "instruction",
        "pixel_bytes",
        "source_count",
        "manifest_digest",
    ),
)
def test_visual_input_manifest_tampering_fails_closed(tamper: str) -> None:
    model_extra, payload, policy_input = _strict_visual_manifest_fixture()
    manifest = model_extra["humanoid_visual_input_manifest"]
    if tamper == "missing_manifest":
        del model_extra["humanoid_visual_input_manifest"]
        del payload["visual_input_manifest_sha256"]
    elif tamper == "renderer_evidence":
        manifest["source_frames"][0]["renderer_binding_sha256"] = "b" * 64
        payload["visual_input_manifest_sha256"] = humanoid_visual_input_manifest_sha256(
            manifest
        )
    elif tamper == "instruction":
        manifest["instruction_sha256"] = "b" * 64
        payload["visual_input_manifest_sha256"] = humanoid_visual_input_manifest_sha256(
            manifest
        )
    elif tamper == "pixel_bytes":
        payload["pixel_values"][0, 0] = -1.0
    elif tamper == "source_count":
        payload["image_grid_thw"] = torch.tensor(
            [[1, 2, 2], [1, 2, 2]], dtype=torch.int64
        )
    elif tamper == "manifest_digest":
        payload["visual_input_manifest_sha256"] = "0" * 64
    else:
        raise AssertionError(tamper)

    with pytest.raises(ValueError):
        _validate_recorded_visual_input_manifest(
            model_extra=model_extra,
            payload=payload,
            policy_input=policy_input,
            step_index=0,
        )


def test_model_input_tensor_digest_binds_dtype_shape_and_exact_bytes() -> None:
    tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    expected = humanoid_model_input_tensor_sha256(tensor)

    assert humanoid_model_input_tensor_sha256(tensor.clone()) == expected
    assert humanoid_model_input_tensor_sha256(tensor.reshape(2, 6)) != expected
    assert humanoid_model_input_tensor_sha256(tensor.to(torch.float64)) != expected
    mutated = tensor.clone()
    mutated[0, 0] = -1.0
    assert humanoid_model_input_tensor_sha256(mutated) != expected


def test_strict_policy_camera_filters_auxiliary_frames_but_audits_and_saves(
    tmp_path: Path,
) -> None:
    state, policy_image, contract, joint_names = _strict_camera_fixture()
    auxiliary_image = _camera_image(
        frame_start_us=10_000,
        frame_end_us=20_000,
        logical_id="rear-debug",
        image_bytes=b"auxiliary-camera-bytes",
    )

    routing = _route_camera_frames(
        states=[state],
        camera_images=[auxiliary_image, policy_image],
        observation_timestamp_us=20_000,
        observation_decision_id=7,
        joint_names=joint_names,
        policy_camera_contract=contract,
    )

    assert [frame.logical_id for frame in routing.policy_frames_by_env[0]] == ["d455"]
    assert {frame.logical_id for frame in routing.all_frames_by_env[0]} == {
        "d455",
        "rear-debug",
    }
    identities = tuple(
        _camera_frame_identity(frame) for frame in routing.all_frames_by_env[0]
    )
    assert all(isinstance(item, HumanoidCameraFrameIdentity) for item in identities)
    recorded = _recorded_policy_output(
        step_index=0,
        policy_input=HumanoidPolicyInput(
            session_uuid="strict-camera-with-auxiliary",
            episode_id=1,
            step_index=0,
            timestamp_us=20_000,
            env_id=0,
            qpos=torch.tensor(state.qpos),
            qvel=torch.zeros(35),
            observation=torch.zeros(1),
            scalars={},
            camera_frames=routing.policy_frames_by_env[0],
            camera_frame_identities=identities,
            decision_id=7,
        ),
        output=HumanoidPolicyStepOutput(env_id=0, action=torch.zeros(1)),
        action_values=torch.zeros(1),
        motion_reference=None,
    )
    metadata = recorded.model_extra["humanoid_camera_frames"]
    assert {item["logical_id"] for item in metadata} == {"d455", "rear-debug"}
    assert all(
        "image_bytes" not in item and "render_qpos" not in item for item in metadata
    )

    _save_camera_images(routing.all_frames_by_env[0], tmp_path)
    assert len(tuple(tmp_path.iterdir())) == 2


def test_policy_camera_session_spec_is_parsed_without_planner_coupling() -> None:
    _, _, expected, _ = _strict_camera_fixture()
    request = SimpleNamespace(
        policy_camera_spec=SimpleNamespace(
            schema=expected.schema,
            logical_id=expected.logical_id,
            width=expected.width,
            height=expected.height,
            image_format=expected.image_format,
            max_frame_age_us=expected.max_frame_age_us,
            contract_sha256=expected.contract_sha256,
        )
    )

    assert _policy_camera_contract(request) == expected


def test_required_policy_camera_server_rejects_missing_session_contract() -> None:
    factory_called = False

    def policy_factory(session_uuid, request):
        nonlocal factory_called
        factory_called = True
        return ZeroHumanoidPolicy(int(request.action_size))

    servicer = HumanoidPolicyGrpcServicer(
        policy_factory=policy_factory,
        require_policy_camera=True,
    )
    servicer.reserve_session("strict-camera", behavior_policy_version=1)

    with pytest.raises(ValueError, match="non-empty policy_camera_spec"):
        servicer.start_session(
            SimpleNamespace(
                session_uuid="strict-camera",
                action_size=1,
                observation_schema="test.v1",
                observation_terms=[SimpleNamespace(name="test", size=1)],
                policy_options={},
            ),
            context=None,
        )

    assert not factory_called


def test_strict_policy_camera_accepts_declared_jpeg_raster() -> None:
    state, image, contract, joint_names = _strict_camera_fixture()
    image_buffer = io.BytesIO()
    Image.new("RGB", (224, 140)).save(image_buffer, format="JPEG")
    image.image_bytes = image_buffer.getvalue()
    image.image_sha256 = hashlib.sha256(image.image_bytes).hexdigest()
    image.render_receipt_sha256 = humanoid_render_receipt_v2_sha256(
        render_state_sha256=image.render_state_sha256,
        camera_contract_sha256=image.camera_contract_sha256,
        image_sha256=image.image_sha256,
        image_format="jpeg",
        width=224,
        height=140,
        scene_fingerprint=image.scene_fingerprint,
        model_signature_sha256=image.model_signature_sha256,
        camera_to_world_sha256=image.camera_to_world_sha256,
        renderer_binding_sha256=image.renderer_binding_sha256,
    )
    frames = _camera_frames_by_env(
        states=[state],
        camera_images=[image],
        observation_timestamp_us=20_000,
        observation_decision_id=7,
        joint_names=joint_names,
        policy_camera_contract=replace(contract, image_format="jpeg"),
    )

    assert frames[0][0].image_bytes == image.image_bytes


def test_policy_camera_rejects_non_rgb_png_and_jpeg_headers() -> None:
    rgba_buffer = io.BytesIO()
    Image.new("RGBA", (224, 140)).save(rgba_buffer, format="PNG")
    grayscale_buffer = io.BytesIO()
    Image.new("L", (224, 140)).save(grayscale_buffer, format="JPEG")

    with pytest.raises(ValueError, match="HWC uint8 RGB"):
        _encoded_image_size(rgba_buffer.getvalue(), "png")
    with pytest.raises(ValueError, match="HWC uint8 RGB"):
        _encoded_image_size(grayscale_buffer.getvalue(), "jpeg")
    with pytest.raises(ValueError, match="fully decoded"):
        _encoded_image_size(rgba_buffer.getvalue()[:40], "png")


def test_strict_policy_camera_rejects_raster_size_mismatch() -> None:
    state, image, contract, joint_names = _strict_camera_fixture()
    with pytest.raises(ValueError, match="encoded raster"):
        _camera_frames_by_env(
            states=[state],
            camera_images=[image],
            observation_timestamp_us=20_000,
            observation_decision_id=7,
            joint_names=joint_names,
            policy_camera_contract=replace(contract, width=225),
        )


def test_strict_policy_camera_rejects_lowercase_renderer_binding_tamper() -> None:
    state, image, contract, joint_names = _strict_camera_fixture()
    image.renderer_binding_sha256 = "deadbeef" * 8

    with pytest.raises(ValueError, match="renderer-evidence receipt is invalid"):
        _camera_frames_by_env(
            states=[state],
            camera_images=[image],
            observation_timestamp_us=20_000,
            observation_decision_id=7,
            joint_names=joint_names,
            policy_camera_contract=contract,
        )


@pytest.mark.parametrize(
    ("field", "replacement", "error"),
    (
        ("logical_id", "other", "exactly one current frame"),
        ("frame_start_us", 19_999, "zero-shutter"),
        ("render_timestamp_us", 19_999, "render_timestamp_us"),
        ("observation_decision_id", 6, "observation_decision_id"),
        ("render_qpos", [0.0] * 35, "36-D render_qpos"),
        ("render_qpos", [0.0] * 36, "does not match its captured state"),
        ("render_state_sha256", "d" * 64, "render-state receipt"),
        ("camera_contract_sha256", "d" * 64, "contract identity"),
        ("image_sha256", "d" * 64, "encoded-image receipt"),
        ("render_receipt_sha256", "d" * 64, "renderer-evidence receipt"),
        ("scene_fingerprint", "A" * 64, "scene_fingerprint"),
        ("model_signature_sha256", "", "model_signature_sha256"),
        ("camera_to_world_sha256", "f" * 63, "camera_to_world_sha256"),
        ("renderer_binding_sha256", "g" * 64, "renderer_binding_sha256"),
        ("image_bytes", b"not-an-image", "fully decoded"),
    ),
)
def test_strict_policy_camera_rejects_mismatched_receipts(
    field: str,
    replacement: object,
    error: str,
) -> None:
    state, image, contract, joint_names = _strict_camera_fixture()
    invalid_image = SimpleNamespace(**{**vars(image), field: replacement})
    with pytest.raises(ValueError, match=error):
        _camera_frames_by_env(
            states=[state],
            camera_images=[invalid_image],
            observation_timestamp_us=20_000,
            observation_decision_id=7,
            joint_names=joint_names,
            policy_camera_contract=contract,
        )


def test_strict_policy_camera_requires_one_current_frame_per_lane() -> None:
    state, _, contract, joint_names = _strict_camera_fixture()
    with pytest.raises(ValueError, match="exactly one current frame"):
        _camera_frames_by_env(
            states=[state],
            camera_images=[],
            observation_timestamp_us=20_000,
            observation_decision_id=7,
            joint_names=joint_names,
            policy_camera_contract=contract,
        )


class _MotionReferencePolicy:
    """Deterministic reference planner used to exercise the full RPC lifecycle."""

    def __init__(self, frame_count: int = 50, decode_context_schema: str = "") -> None:
        self.frame_count = frame_count
        self.decode_context_schema = decode_context_schema
        self.calls = []
        self.close_calls = 0

    def step(self, policy_inputs, *, sample_actions: bool = True):
        self.calls.append((policy_inputs, sample_actions))
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
                for index in range(self.frame_count)
            )
            decode_context = None
            if self.decode_context_schema:
                local_xy = torch.zeros((self.frame_count, 2), dtype=torch.float32)
                local_xy[:, 0] = (
                    torch.arange(self.frame_count, dtype=torch.float32) * 0.01
                )
                decode_context = HumanoidReferenceDecodeContext(
                    schema=self.decode_context_schema,
                    chunk_base_quaternion_wxyz=item.qpos[3:7].clone(),
                    local_xy_from_frame_zero=local_xy,
                )
            reference = HumanoidMotionReference(
                reference_id=100 + item.decision_id,
                source_decision_id=item.decision_id,
                frames=frames,
                root_z_alignment_offset_m=0.125,
                decode_context=decode_context,
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
        self.close_calls += 1


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
        observation_schema="humanoid_motion_reference_navigation_xy.v1",
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
    duration: int = 25,
    applied_digest: str | None = None,
    terminated: bool = False,
    truncated: bool = False,
) -> SimpleNamespace:
    ticks = []
    for index in range(duration):
        timestamp_us = start_timestamp_us + index * 20_000
        ticks.append(
            SimpleNamespace(
                control_tick_offset=index + 1,
                reference_action_index=index,
                state=_motion_state(timestamp_us),
                active_reference_id=reference_id,
                active_reference_sha256=digest,
                applied_reference_sha256=applied_digest or digest,
                root_z_alignment_offset_m=0.125,
                reward=0.1,
                terminated=terminated and index == duration - 1,
                truncated=truncated and index == duration - 1,
                metrics={},
                control_episode_step=first_control_step + index,
            )
        )
    return SimpleNamespace(
        env_id=0,
        source_decision_id=source_decision_id,
        ticks=ticks,
    )


def _async_motion_feedback(
    *,
    source_decision_id: int,
    predecessor_reference_id: int,
    predecessor_digest: str,
    source_reference_id: int,
    source_digest: str,
    predecessor_ticks: int,
    source_ticks: int,
    source_timestamp_us: int,
    first_control_step: int,
    terminated: bool = False,
) -> SimpleNamespace:
    """Build one delayed-install interval with exact active-plan provenance."""
    ticks = []
    duration = predecessor_ticks + source_ticks
    for tick_index in range(duration):
        predecessor = tick_index < predecessor_ticks
        ticks.append(
            SimpleNamespace(
                control_tick_offset=tick_index + 1,
                reference_action_index=(
                    min(25 + tick_index, 49) if predecessor else tick_index
                ),
                state=_motion_state(source_timestamp_us + (tick_index + 1) * 20_000),
                active_reference_id=(
                    predecessor_reference_id if predecessor else source_reference_id
                ),
                active_reference_sha256=(
                    predecessor_digest if predecessor else source_digest
                ),
                applied_reference_sha256=("c" if predecessor else "d") * 64,
                root_z_alignment_offset_m=0.125,
                reward=0.1,
                terminated=terminated and tick_index == duration - 1,
                truncated=False,
                metrics={},
                control_episode_step=first_control_step + tick_index,
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
    camera_images: list[SimpleNamespace] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        session_uuid="motion-session",
        request_kind=request_kind,
        bootstrap_only=False,
        bootstrap_env_ids=list(bootstrap_env_ids or []),
        observation=SimpleNamespace(
            timestamp_us=timestamp_us,
            decision_id=decision_id,
            env_states=[_motion_state(timestamp_us)],
            camera_images=list(camera_images or []),
            feedback_traces=feedback_traces,
        ),
    )


def test_abort_discards_incomplete_motion_replay_and_is_idempotent() -> None:
    planner = _MotionReferencePolicy()
    servicer = HumanoidPolicyGrpcServicer(
        policy_factory=lambda session_uuid, request: planner
    )
    servicer.reserve_session("motion-session", behavior_policy_version=11)
    servicer.start_session(
        SimpleNamespace(
            session_uuid="motion-session",
            random_seed=7,
            action_size=0,
            execution_mode=2,
            joint_names=list(MOTION_REFERENCE_JOINT_NAMES),
            observation_schema="humanoid_motion_reference_navigation_xy.v1",
            action_schema="g1_motion_reference_29d_50hz_h50.v1",
            observation_terms=[SimpleNamespace(name="navigation_position_xy", size=2)],
            reference_spec=SimpleNamespace(
                schema="g1_motion_reference_29d_50hz_h50.v1",
                joint_names=list(MOTION_REFERENCE_JOINT_NAMES),
                frame_count=50,
                sample_period_us=20_000,
                control_ticks_per_policy_step=25,
                decode_context_schema="",
            ),
            policy_options={},
            attempt_id="attempt",
            scene_id="hq_stairs",
            scenario_id="ascend",
        ),
        context=None,
    )
    servicer.act(
        _motion_act_request(
            decision_id=0,
            timestamp_us=0,
            request_kind=2,
            feedback_traces=[],
        ),
        context=None,
    )

    with pytest.raises(
        ValueError, match="before every plan received physical feedback"
    ):
        servicer.close_session(
            SimpleNamespace(session_uuid="motion-session"), context=None
        )
    with pytest.raises(KeyError):
        servicer.pop_session_record("motion-session")

    abort = SimpleNamespace(session_uuid="motion-session")
    servicer.abort_session(abort, context=None)
    servicer.abort_session(abort, context=None)

    assert "motion-session" not in servicer._sessions
    assert planner.close_calls == 1
    with pytest.raises(KeyError):
        servicer.pop_session_record("motion-session")


def test_motion_policy_receives_camera_frames_at_initial_replan_and_finalize() -> None:
    planner = _MotionReferencePolicy(frame_count=50)
    servicer = HumanoidPolicyGrpcServicer(
        policy_factory=lambda session_uuid, request: planner
    )
    servicer.reserve_session("motion-session", behavior_policy_version=11)
    servicer.start_session(
        SimpleNamespace(
            session_uuid="motion-session",
            random_seed=7,
            action_size=0,
            execution_mode=2,
            joint_names=list(MOTION_REFERENCE_JOINT_NAMES),
            observation_schema="humanoid_motion_reference_navigation_xy.v1",
            action_schema="g1_motion_reference_29d_50hz_h50.v1",
            observation_terms=[SimpleNamespace(name="navigation_position_xy", size=2)],
            reference_spec=SimpleNamespace(
                schema="g1_motion_reference_29d_50hz_h50.v1",
                joint_names=list(MOTION_REFERENCE_JOINT_NAMES),
                frame_count=50,
                sample_period_us=20_000,
                control_ticks_per_policy_step=25,
                decode_context_schema="",
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
            camera_images=[
                _camera_image(
                    frame_start_us=0,
                    frame_end_us=0,
                    image_bytes=b"initial-rgb",
                )
            ],
        ),
        context=None,
    )
    first_reference = initial.plan_updates[0]
    replan = servicer.act(
        _motion_act_request(
            decision_id=1,
            timestamp_us=500_000,
            request_kind=3,
            feedback_traces=[
                _motion_feedback(
                    source_decision_id=0,
                    reference_id=first_reference.reference_id,
                    digest=first_reference.reference_sha256,
                    start_timestamp_us=20_000,
                    first_control_step=1,
                )
            ],
            camera_images=[
                _camera_image(
                    frame_start_us=490_000,
                    frame_end_us=500_000,
                    image_bytes=b"replan-rgb",
                )
            ],
        ),
        context=None,
    )
    second_reference = replan.plan_updates[0]
    finalized = servicer.act(
        _motion_act_request(
            decision_id=2,
            timestamp_us=560_000,
            request_kind=4,
            feedback_traces=[
                _motion_feedback(
                    source_decision_id=1,
                    reference_id=second_reference.reference_id,
                    digest=second_reference.reference_sha256,
                    start_timestamp_us=520_000,
                    first_control_step=26,
                    duration=3,
                    truncated=True,
                )
            ],
            bootstrap_env_ids=[0],
            camera_images=[
                _camera_image(
                    frame_start_us=550_000,
                    frame_end_us=560_000,
                    image_bytes=b"final-rgb",
                )
            ],
        ),
        context=None,
    )

    assert [sample_actions for _, sample_actions in planner.calls] == [
        True,
        True,
        False,
    ]
    assert [
        call_inputs[0].camera_frames[0].image_bytes for call_inputs, _ in planner.calls
    ] == [b"initial-rgb", b"replan-rgb", b"final-rgb"]
    assert planner.calls[0][0][0].step_index == 0
    assert planner.calls[1][0][0].step_index == 1
    assert planner.calls[2][0][0].step_index == 2
    assert planner.calls[2][0][0].bootstrap_requested
    assert {
        int(item.env_id): float(item.value) for item in finalized.value_estimates
    } == {0: 3.0}

    servicer.close_session(
        SimpleNamespace(session_uuid="motion-session"),
        context=None,
    )
    record = servicer.pop_session_record("motion-session")
    assert len(record.outputs) == 2
    assert record.final_bootstrap_values == {0: 3.0}
    assert [
        output.model_extra["humanoid_camera_frames"][0]["sha256"]
        for output in record.outputs
    ] == [
        hashlib.sha256(b"initial-rgb").hexdigest(),
        hashlib.sha256(b"replan-rgb").hexdigest(),
    ]
    json.dumps([output.model_extra for output in record.outputs])
    assert all(
        "humanoid_camera_frames" not in output.replay_data.payload
        for output in record.outputs
    )


@pytest.mark.parametrize(
    (
        "terminal",
        "bootstrap_env_ids",
        "expected_bootstrap",
        "predecessor_ticks",
        "source_ticks",
    ),
    ((True, [], {}, 12, 25), (False, [0], {0: 3.0}, 37, 0)),
)
def test_motion_policy_records_37_tick_predecessor_to_source_interval(
    terminal: bool,
    bootstrap_env_ids: list[int],
    expected_bootstrap: dict[int, float],
    predecessor_ticks: int,
    source_ticks: int,
) -> None:
    """One policy row owns all ticks elapsed until the following policy sample."""
    servicer = HumanoidPolicyGrpcServicer(
        policy_factory=lambda session_uuid, request: _MotionReferencePolicy(
            frame_count=50
        )
    )
    servicer.reserve_session("motion-session", behavior_policy_version=11)
    servicer.start_session(
        SimpleNamespace(
            session_uuid="motion-session",
            random_seed=7,
            action_size=0,
            execution_mode=2,
            joint_names=list(MOTION_REFERENCE_JOINT_NAMES),
            observation_schema="humanoid_motion_reference_navigation_xy.v1",
            action_schema="g1_motion_reference_29d_50hz_h50.v1",
            observation_terms=[SimpleNamespace(name="navigation_position_xy", size=2)],
            reference_spec=SimpleNamespace(
                schema="g1_motion_reference_29d_50hz_h50.v1",
                joint_names=list(MOTION_REFERENCE_JOINT_NAMES),
                frame_count=50,
                sample_period_us=20_000,
                control_ticks_per_policy_step=25,
                decode_context_schema="",
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
    first = initial.plan_updates[0]
    replanned = servicer.act(
        _motion_act_request(
            decision_id=1,
            timestamp_us=500_000,
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
    second = replanned.plan_updates[0]
    servicer.act(
        _motion_act_request(
            decision_id=2,
            timestamp_us=1_240_000,
            request_kind=4,
            feedback_traces=[
                _async_motion_feedback(
                    source_decision_id=1,
                    predecessor_reference_id=first.reference_id,
                    predecessor_digest=first.reference_sha256,
                    source_reference_id=second.reference_id,
                    source_digest=second.reference_sha256,
                    predecessor_ticks=predecessor_ticks,
                    source_ticks=source_ticks,
                    source_timestamp_us=500_000,
                    first_control_step=26,
                    terminated=terminal,
                )
            ],
            bootstrap_env_ids=bootstrap_env_ids,
        ),
        context=None,
    )
    servicer.close_session(
        SimpleNamespace(session_uuid="motion-session"),
        context=None,
    )

    record = servicer.pop_session_record("motion-session")
    assert record.final_bootstrap_values == expected_bootstrap
    trace = record.outputs[1].replay_data.payload["feedback_trace"]
    assert trace["state_contract_schema"] == HUMANOID_FEEDBACK_STATE_CONTRACT_SCHEMA
    assert tuple(trace["joint_names"]) == MOTION_REFERENCE_JOINT_NAMES
    expected_joint_hash = hashlib.sha256(
        json.dumps(
            list(MOTION_REFERENCE_JOINT_NAMES),
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert trace["ordered_joint_names_sha256"] == expected_joint_hash
    assert len(trace["ticks"]) == 37
    assert [tick["active_reference_id"] for tick in trace["ticks"]] == [
        first.reference_id
    ] * predecessor_ticks + [second.reference_id] * source_ticks
    assert bool(record.outputs[1].replay_data.payload.get("outer_truncated")) is bool(
        bootstrap_env_ids
    )


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


@pytest.mark.parametrize("control_ticks", (25,))
def test_h50_motion_reference_abi_accepts_native_replan_trigger(
    control_ticks: int,
) -> None:
    """The generic H50 wire keeps its one-second buffer at the native trigger."""
    request = SimpleNamespace(
        action_schema="g1_motion_reference_29d_50hz_h50.v1",
        joint_names=list(MOTION_REFERENCE_JOINT_NAMES),
        reference_spec=SimpleNamespace(
            schema="g1_motion_reference_29d_50hz_h50.v1",
            joint_names=list(MOTION_REFERENCE_JOINT_NAMES),
            frame_count=50,
            sample_period_us=20_000,
            control_ticks_per_policy_step=control_ticks,
            decode_context_schema="",
        ),
    )

    joint_names, frame_count, sample_period_us, control_ticks, decode_context_schema = (
        _validate_motion_reference_session(request)
    )

    assert joint_names == MOTION_REFERENCE_JOINT_NAMES
    assert (frame_count, sample_period_us, control_ticks) == (
        50,
        20_000,
        request.reference_spec.control_ticks_per_policy_step,
    )
    assert decode_context_schema == ""

    request.reference_spec.decode_context_schema = (
        HUMANOID_FULL_ROTATION_LOCAL_XY_DECODE_CONTEXT_SCHEMA
    )
    assert _validate_motion_reference_session(request)[-1] == (
        HUMANOID_FULL_ROTATION_LOCAL_XY_DECODE_CONTEXT_SCHEMA
    )
    request.reference_spec.decode_context_schema = "unknown/v1"
    with pytest.raises(ValueError, match="unsupported decode_context_schema"):
        _validate_motion_reference_session(request)

    request.reference_spec.decode_context_schema = ""
    request.reference_spec.schema = "g1_motion_reference_29d_50hz_h70.v1"
    with pytest.raises(ValueError, match="requires the one-second H50"):
        _validate_motion_reference_session(request)


def _motion_reference_wire_fixture(
    *,
    negotiated_schema: str,
    decode_context: HumanoidReferenceDecodeContext | None,
) -> tuple[HumanoidPolicyStepOutput, object, object]:
    """Build one source plan, source observation, and negotiated session."""
    policy_input = SimpleNamespace(
        env_id=0,
        decision_id=7,
        timestamp_us=400_000,
        qpos=torch.tensor(
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0] + [0.0] * 29,
            dtype=torch.float32,
        ),
        qvel=torch.zeros(35, dtype=torch.float32),
        bootstrap_requested=False,
    )
    output = _MotionReferencePolicy().step((policy_input,))[0]
    assert output.motion_reference is not None
    output = replace(
        output,
        motion_reference=replace(
            output.motion_reference,
            decode_context=decode_context,
        ),
    )
    session = SimpleNamespace(
        reference_joint_names=MOTION_REFERENCE_JOINT_NAMES,
        reference_frame_count=50,
        reference_sample_period_us=20_000,
        control_ticks_per_policy_step=25,
        reference_decode_context_schema=negotiated_schema,
    )
    return output, policy_input, session


def _valid_reference_decode_context() -> HumanoidReferenceDecodeContext:
    """Return a nontrivial context anchored to the fixture source state."""
    local_xy = torch.zeros((50, 2), dtype=torch.float32)
    local_xy[:, 0] = torch.arange(50, dtype=torch.float32) * 0.01
    local_xy[:, 1] = torch.arange(50, dtype=torch.float32) * -0.02
    return HumanoidReferenceDecodeContext(
        schema=HUMANOID_FULL_ROTATION_LOCAL_XY_DECODE_CONTEXT_SCHEMA,
        chunk_base_quaternion_wxyz=torch.tensor(
            [1.0, 0.0, 0.0, 0.0], dtype=torch.float32
        ),
        local_xy_from_frame_zero=local_xy,
    )


def test_plan_update_wires_negotiated_decode_context_and_uses_v2_hash() -> None:
    context = _valid_reference_decode_context()
    output, policy_input, session = _motion_reference_wire_fixture(
        negotiated_schema=HUMANOID_FULL_ROTATION_LOCAL_XY_DECODE_CONTEXT_SCHEMA,
        decode_context=context,
    )

    update = _plan_update_from_output(
        output=output,
        policy_input=policy_input,
        session=session,
    )

    assert update.HasField("decode_context")
    assert update.decode_context.schema == context.schema
    assert (
        update.decode_context.chunk_base_quaternion_wxyz.w,
        update.decode_context.chunk_base_quaternion_wxyz.x,
        update.decode_context.chunk_base_quaternion_wxyz.y,
        update.decode_context.chunk_base_quaternion_wxyz.z,
    ) == (1.0, 0.0, 0.0, 0.0)
    assert update.decode_context.local_xy_from_frame_zero == pytest.approx(
        context.local_xy_from_frame_zero.reshape(-1).tolist()
    )
    assert update.reference_sha256 == "2" * 64


def test_plan_update_without_decode_context_retains_v1_hash() -> None:
    output, policy_input, session = _motion_reference_wire_fixture(
        negotiated_schema="",
        decode_context=None,
    )

    update = _plan_update_from_output(
        output=output,
        policy_input=policy_input,
        session=session,
    )

    assert not update.HasField("decode_context")
    assert update.reference_sha256 == "1" * 64


def test_motion_policy_rpc_records_the_context_bearing_wire_digest() -> None:
    planner = _MotionReferencePolicy(
        decode_context_schema=(HUMANOID_FULL_ROTATION_LOCAL_XY_DECODE_CONTEXT_SCHEMA)
    )
    servicer = HumanoidPolicyGrpcServicer(
        policy_factory=lambda session_uuid, request: planner
    )
    servicer.reserve_session("motion-session", behavior_policy_version=11)
    servicer.start_session(
        SimpleNamespace(
            session_uuid="motion-session",
            action_size=0,
            execution_mode=2,
            joint_names=list(MOTION_REFERENCE_JOINT_NAMES),
            observation_schema="humanoid_motion_reference_navigation_xy.v1",
            action_schema="g1_motion_reference_29d_50hz_h50.v1",
            observation_terms=[SimpleNamespace(name="navigation_position_xy", size=2)],
            reference_spec=SimpleNamespace(
                schema="g1_motion_reference_29d_50hz_h50.v1",
                joint_names=list(MOTION_REFERENCE_JOINT_NAMES),
                frame_count=50,
                sample_period_us=20_000,
                control_ticks_per_policy_step=25,
                decode_context_schema=(
                    HUMANOID_FULL_ROTATION_LOCAL_XY_DECODE_CONTEXT_SCHEMA
                ),
            ),
            policy_options={},
        ),
        context=None,
    )

    response = servicer.act(
        _motion_act_request(
            decision_id=0,
            timestamp_us=0,
            request_kind=2,
            feedback_traces=[],
        ),
        context=None,
    )

    update = response.plan_updates[0]
    recorded = servicer._sessions["motion-session"].outputs[0]
    assert update.HasField("decode_context")
    assert update.reference_sha256 == "2" * 64
    assert recorded.replay_data.payload["reference_sha256"] == (update.reference_sha256)
    servicer.discard_session("motion-session")


@pytest.mark.parametrize(
    ("negotiated_schema", "context", "error"),
    (
        (
            HUMANOID_FULL_ROTATION_LOCAL_XY_DECODE_CONTEXT_SCHEMA,
            None,
            "missing its negotiated decode context",
        ),
        (
            "",
            _valid_reference_decode_context(),
            "was not negotiated",
        ),
        (
            HUMANOID_FULL_ROTATION_LOCAL_XY_DECODE_CONTEXT_SCHEMA,
            replace(_valid_reference_decode_context(), schema="wrong/v1"),
            "schema does not match",
        ),
        (
            HUMANOID_FULL_ROTATION_LOCAL_XY_DECODE_CONTEXT_SCHEMA,
            replace(
                _valid_reference_decode_context(),
                chunk_base_quaternion_wxyz=torch.tensor(
                    [0.0, 1.0, 0.0, 0.0], dtype=torch.float32
                ),
            ),
            "does not match the policy input",
        ),
        (
            HUMANOID_FULL_ROTATION_LOCAL_XY_DECODE_CONTEXT_SCHEMA,
            replace(
                _valid_reference_decode_context(),
                local_xy_from_frame_zero=torch.zeros((49, 2)),
            ),
            r"finite \[50,2\] matrix",
        ),
        (
            HUMANOID_FULL_ROTATION_LOCAL_XY_DECODE_CONTEXT_SCHEMA,
            replace(
                _valid_reference_decode_context(),
                local_xy_from_frame_zero=torch.cat(
                    (torch.ones((1, 2)), torch.zeros((49, 2)))
                ),
            ),
            "row zero must be exact zero",
        ),
    ),
)
def test_plan_update_decode_context_validation_fails_closed(
    negotiated_schema: str,
    context: HumanoidReferenceDecodeContext | None,
    error: str,
) -> None:
    output, policy_input, session = _motion_reference_wire_fixture(
        negotiated_schema=negotiated_schema,
        decode_context=context,
    )

    with pytest.raises(ValueError, match=error):
        _plan_update_from_output(
            output=output,
            policy_input=policy_input,
            session=session,
        )


def test_humanoid_policy_session_registers_and_releases_model_lease() -> None:
    """The gRPC session owns its immutable behavior model until close."""

    class _LeaseRegistry:
        def __init__(self) -> None:
            self.leases: dict[str, InferenceModelLease] = {}
            self.released: list[str] = []

        def register_session_model_lease(
            self,
            session_uuid: str,
            lease: InferenceModelLease,
        ) -> None:
            self.leases[session_uuid] = lease

        def release_session_model_lease(self, session_uuid: str) -> None:
            self.released.append(session_uuid)
            self.leases.pop(session_uuid)

    registry = _LeaseRegistry()
    servicer = HumanoidPolicyGrpcServicer(
        policy_factory=lambda session_uuid, request: ZeroHumanoidPolicy(
            int(request.action_size)
        ),
        model_lease_registry=registry,
    )
    lease = InferenceModelLease(
        behavior_policy_version=7,
        model=torch.nn.Linear(1, 1),
    )

    servicer.reserve_session(
        "leased-session",
        behavior_policy_version=7,
        model_lease=lease,
    )
    assert registry.leases == {"leased-session": lease}
    servicer.start_session(
        SimpleNamespace(
            session_uuid="leased-session",
            action_size=1,
            observation_schema="test.v1",
            observation_terms=[SimpleNamespace(name="test", size=1)],
            policy_options={},
        ),
        context=None,
    )
    servicer.close_session(
        SimpleNamespace(session_uuid="leased-session"),
        context=None,
    )

    assert registry.leases == {}
    assert registry.released == ["leased-session"]

    aborted_lease = InferenceModelLease(
        behavior_policy_version=8,
        model=torch.nn.Linear(1, 1),
    )
    servicer.reserve_session(
        "aborted-session",
        behavior_policy_version=8,
        model_lease=aborted_lease,
    )
    servicer.start_session(
        SimpleNamespace(
            session_uuid="aborted-session",
            action_size=1,
            observation_schema="test.v1",
            observation_terms=[SimpleNamespace(name="test", size=1)],
            policy_options={},
        ),
        context=None,
    )
    abort = SimpleNamespace(session_uuid="aborted-session")
    servicer.abort_session(abort, context=None)
    servicer.abort_session(abort, context=None)

    assert registry.leases == {}
    assert registry.released == ["leased-session", "aborted-session"]
    with pytest.raises(KeyError):
        servicer.pop_session_record("aborted-session")


@pytest.mark.parametrize(
    ("terminated", "truncated", "bootstrap_env_ids", "expected_bootstrap"),
    (
        (True, False, [], {}),
        (False, True, [0], {0: 3.0}),
    ),
)
def test_motion_reference_finalize_records_terminal_k_prefix_without_replan(
    terminated: bool,
    truncated: bool,
    bootstrap_env_ids: list[int],
    expected_bootstrap: dict[int, float],
) -> None:
    """A partial macro is retained; only truncation requests a final value."""
    servicer = HumanoidPolicyGrpcServicer(
        policy_factory=lambda session_uuid, request: _MotionReferencePolicy(
            frame_count=int(request.reference_spec.frame_count)
        )
    )
    servicer.reserve_session("motion-session", behavior_policy_version=11)
    servicer.start_session(
        SimpleNamespace(
            session_uuid="motion-session",
            random_seed=7,
            action_size=0,
            execution_mode=2,
            joint_names=list(MOTION_REFERENCE_JOINT_NAMES),
            observation_schema="humanoid_motion_reference_navigation_xy.v1",
            action_schema="g1_motion_reference_29d_50hz_h50.v1",
            observation_terms=[SimpleNamespace(name="navigation_position_xy", size=2)],
            reference_spec=SimpleNamespace(
                schema="g1_motion_reference_29d_50hz_h50.v1",
                joint_names=list(MOTION_REFERENCE_JOINT_NAMES),
                frame_count=50,
                sample_period_us=20_000,
                control_ticks_per_policy_step=25,
                decode_context_schema="",
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
    first = initial.plan_updates[0]
    duration = 7
    finalized = servicer.act(
        _motion_act_request(
            decision_id=1,
            timestamp_us=duration * 20_000,
            request_kind=4,
            feedback_traces=[
                _motion_feedback(
                    source_decision_id=0,
                    reference_id=first.reference_id,
                    digest=first.reference_sha256,
                    start_timestamp_us=20_000,
                    first_control_step=1,
                    duration=duration,
                    terminated=terminated,
                    truncated=truncated,
                )
            ],
            bootstrap_env_ids=bootstrap_env_ids,
        ),
        context=None,
    )

    assert len(finalized.plan_updates) == 0
    assert {
        int(item.env_id): float(item.value) for item in finalized.value_estimates
    } == expected_bootstrap

    servicer.close_session(
        SimpleNamespace(session_uuid="motion-session"),
        context=None,
    )
    record = servicer.pop_session_record("motion-session")
    assert len(record.outputs) == 1
    assert record.final_bootstrap_values == expected_bootstrap
    replay_payload = record.outputs[0].replay_data.payload
    assert len(replay_payload["feedback_trace"]["ticks"]) == duration
    assert bool(replay_payload.get("outer_truncated", False)) is truncated


def test_invalid_action_terminal_tick_closes_source_plan_without_missing_row() -> None:
    """One source-owned safe-hold tick is sufficient physical feedback to close."""
    servicer = HumanoidPolicyGrpcServicer(
        policy_factory=lambda session_uuid, request: _MotionReferencePolicy(
            frame_count=int(request.reference_spec.frame_count)
        )
    )
    servicer.reserve_session("motion-session", behavior_policy_version=11)
    servicer.start_session(
        SimpleNamespace(
            session_uuid="motion-session",
            random_seed=7,
            action_size=0,
            execution_mode=2,
            joint_names=list(MOTION_REFERENCE_JOINT_NAMES),
            observation_schema="humanoid_motion_reference_navigation_xy.v1",
            action_schema="g1_motion_reference_29d_50hz_h50.v1",
            observation_terms=[SimpleNamespace(name="navigation_position_xy", size=2)],
            reference_spec=SimpleNamespace(
                schema="g1_motion_reference_29d_50hz_h50.v1",
                joint_names=list(MOTION_REFERENCE_JOINT_NAMES),
                frame_count=50,
                sample_period_us=20_000,
                control_ticks_per_policy_step=25,
                decode_context_schema="",
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
    source = initial.plan_updates[0]
    safe_hold_sha256 = "c" * 64
    feedback = _motion_feedback(
        source_decision_id=0,
        reference_id=source.reference_id,
        digest=source.reference_sha256,
        applied_digest=safe_hold_sha256,
        start_timestamp_us=20_000,
        first_control_step=1,
        duration=1,
        terminated=True,
    )
    feedback.ticks[0].reward = -10.0
    feedback.ticks[0].metrics = {
        "terminal_invalid_policy_action": 1.0,
        "reference_install_valid": 0.0,
        "reference_safety_hold_applied": 1.0,
    }

    finalized = servicer.act(
        _motion_act_request(
            decision_id=1,
            timestamp_us=20_000,
            request_kind=4,
            feedback_traces=[feedback],
        ),
        context=None,
    )

    assert len(finalized.plan_updates) == 0
    servicer.close_session(
        SimpleNamespace(session_uuid="motion-session"),
        context=None,
    )
    record = servicer.pop_session_record("motion-session")
    assert len(record.outputs) == 1
    trace = record.outputs[0].replay_data.payload["feedback_trace"]
    assert trace["source_decision_id"] == 0
    assert len(trace["ticks"]) == 1
    tick = trace["ticks"][0]
    assert tick["active_reference_id"] == source.reference_id
    assert tick["active_reference_sha256"] == source.reference_sha256
    assert tick["applied_reference_sha256"] == safe_hold_sha256
    assert tick["applied_reference_sha256"] != tick["active_reference_sha256"]
    assert tick["reward"] == -10.0
    assert tick["terminated"] is True
    assert tick["truncated"] is False
    assert tick["metrics"] == {
        "terminal_invalid_policy_action": 1.0,
        "reference_install_valid": 0.0,
        "reference_safety_hold_applied": 1.0,
    }


def test_policy_server_accepts_stable_runtime_applied_reference_hash() -> None:
    """Runtime transformations retain a separate stable applied-plan identity."""
    schema = "g1_motion_reference_29d_50hz_h50.v1"
    frame_count = 50
    control_ticks = 25
    servicer = HumanoidPolicyGrpcServicer(
        policy_factory=lambda session_uuid, request: _MotionReferencePolicy(
            frame_count=int(request.reference_spec.frame_count)
        )
    )
    servicer.reserve_session("motion-session", behavior_policy_version=11)
    servicer.start_session(
        SimpleNamespace(
            session_uuid="motion-session",
            random_seed=7,
            action_size=0,
            execution_mode=2,
            joint_names=list(MOTION_REFERENCE_JOINT_NAMES),
            observation_schema="humanoid_motion_reference_navigation_xy.v1",
            action_schema=schema,
            observation_terms=[SimpleNamespace(name="navigation_position_xy", size=2)],
            reference_spec=SimpleNamespace(
                schema=schema,
                joint_names=list(MOTION_REFERENCE_JOINT_NAMES),
                frame_count=frame_count,
                sample_period_us=20_000,
                control_ticks_per_policy_step=control_ticks,
                decode_context_schema="",
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
    first = initial.plan_updates[0]
    request = _motion_act_request(
        decision_id=1,
        timestamp_us=control_ticks * 20_000,
        request_kind=3,
        feedback_traces=[
            _motion_feedback(
                source_decision_id=0,
                reference_id=first.reference_id,
                digest=first.reference_sha256,
                applied_digest="c" * 64,
                start_timestamp_us=20_000,
                first_control_step=1,
                duration=control_ticks,
            )
        ],
    )
    response = servicer.act(request, context=None)
    assert len(response.plan_updates) == 1
    servicer.discard_session("motion-session")
