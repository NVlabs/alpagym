# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import sys
import types
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch


def install_alpasim_grpc_stubs() -> None:
    """Install minimal generated-stub stand-ins for runtime unit tests."""
    alpasim_grpc: Any = types.ModuleType("alpasim_grpc")
    alpasim_grpc_v0: Any = types.ModuleType("alpasim_grpc.v0")
    common_pb2: Any = types.ModuleType("alpasim_grpc.v0.common_pb2")
    egodriver_pb2: Any = types.ModuleType("alpasim_grpc.v0.egodriver_pb2")
    runtime_pb2: Any = types.ModuleType("alpasim_grpc.v0.runtime_pb2")
    sensorsim_pb2: Any = types.ModuleType("alpasim_grpc.v0.sensorsim_pb2")
    humanoid_pb2: Any = types.ModuleType("alpasim_grpc.v0.humanoid_pb2")
    humanoid_contracts: Any = types.ModuleType("alpasim_grpc.v0.humanoid_contracts")
    egodriver_pb2_grpc: Any = types.ModuleType("alpasim_grpc.v0.egodriver_pb2_grpc")
    humanoid_pb2_grpc: Any = types.ModuleType("alpasim_grpc.v0.humanoid_pb2_grpc")
    runtime_pb2_grpc: Any = types.ModuleType("alpasim_grpc.v0.runtime_pb2_grpc")

    class Empty:
        """Tiny stand-in for common.Empty."""

    class SessionRequestStatus:
        """Tiny stand-in for common.SessionRequestStatus."""

    class VersionId:
        """Tiny stand-in for common.VersionId."""

        def __init__(self, version_id: str = "", git_hash: str = "") -> None:
            """Store version fields."""
            self.version_id = version_id
            self.git_hash = git_hash

    class Vec3:
        """Tiny stand-in for common.Vec3."""

        def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
            self.x = x
            self.y = y
            self.z = z

    class Quat:
        """Tiny stand-in for common.Quat."""

        def __init__(
            self,
            w: float = 0.0,
            x: float = 0.0,
            y: float = 0.0,
            z: float = 0.0,
        ) -> None:
            self.w = w
            self.x = x
            self.y = y
            self.z = z

    class PoseAtTime:
        """Tiny stand-in for common.PoseAtTime."""

        def __init__(self, timestamp_us: int = 0) -> None:
            """Create a timestamped pose with mutable vec/quat fields."""
            self.timestamp_us = timestamp_us
            self.pose = SimpleNamespace(
                vec=SimpleNamespace(x=0.0, y=0.0, z=0.0),
                quat=SimpleNamespace(w=0.0, x=0.0, y=0.0, z=0.0),
            )

    class Trajectory:
        """Tiny stand-in for common.Trajectory."""

        def __init__(self, poses=None) -> None:
            """Create a trajectory with optional poses."""
            self.poses = list(poses or [])

    class DriveResponse:
        """Tiny stand-in for egodriver.DriveResponse."""

        def __init__(self) -> None:
            """Create an empty response trajectory."""
            self.trajectory = SimpleNamespace(poses=[])

    class Route:
        """Tiny stand-in for egodriver.Route."""

        def __init__(self, timestamp_us: int = 0, waypoints=None) -> None:
            """Store route waypoints."""
            self.timestamp_us = timestamp_us
            self.waypoints = list(waypoints or [])

    class GroundTruth:
        """Tiny stand-in for egodriver.GroundTruth."""

        def __init__(self, trajectory=None, timestamp_us: int = 0) -> None:
            """Store an optional trajectory and the rig anchor timestamp."""
            self.trajectory = trajectory
            self.timestamp_us = timestamp_us

    class DriveSessionRequest:
        """Tiny stand-in for egodriver.DriveSessionRequest."""

        class RolloutSpec:
            """Tiny stand-in for DriveSessionRequest.RolloutSpec."""

            def __init__(self) -> None:
                """Create empty vehicle camera metadata."""
                self.vehicle = SimpleNamespace(available_cameras=[])

    class RolloutCameraImage:
        """Tiny stand-in for egodriver.RolloutCameraImage."""

        class CameraImage:
            """Tiny stand-in for RolloutCameraImage.CameraImage."""

            def __init__(
                self,
                logical_id: str = "",
                image_bytes: bytes = b"",
                frame_end_us: int = 0,
            ) -> None:
                """Store camera-image attributes accessed by the converter."""
                self.logical_id = logical_id
                self.image_bytes = image_bytes
                self.frame_end_us = frame_end_us

    class DriveRequest:
        """Tiny stand-in for egodriver.DriveRequest."""

    class DriveSessionCloseRequest:
        """Tiny stand-in for egodriver.DriveSessionCloseRequest."""

    class GroundTruthRequest:
        """Tiny stand-in for egodriver.GroundTruthRequest."""

    class RolloutEgoTrajectory:
        """Tiny stand-in for egodriver.RolloutEgoTrajectory."""

    class RouteRequest:
        """Tiny stand-in for egodriver.RouteRequest."""

    class CameraSpec:
        """Tiny stand-in for sensorsim.CameraSpec."""

    class HumanoidAction:
        """Tiny stand-in for humanoid.HumanoidAction."""

        def __init__(self, env_id: int = 0, values=None) -> None:
            self.env_id = env_id
            self.values = list(values or [])

    class HumanoidEnvState:
        """Tiny stand-in for humanoid.HumanoidEnvState."""

        def __init__(
            self,
            env_id: int = 0,
            timestamp_us: int = 0,
            qpos=None,
            qvel=None,
            observation=None,
            scalars=None,
            reset_id: int = 0,
        ) -> None:
            self.env_id = env_id
            self.timestamp_us = timestamp_us
            self.qpos = list(qpos or [])
            self.qvel = list(qvel or [])
            self.observation = list(observation or [])
            self.scalars = dict(scalars or {})
            self.reset_id = reset_id
            self.named_observations = []
            self.observation_schema = "videomimic_v9_direct.v1"

    class HumanoidPolicyRequest:
        """Tiny stand-in for humanoid.HumanoidPolicyRequest."""

    class HumanoidPolicyResponse:
        """Tiny stand-in for humanoid.HumanoidPolicyResponse."""

        def __init__(
            self,
            actions=None,
            terminate_session: bool = False,
            behavior_policy_version: str = "",
            value_estimates=None,
            plan_updates=None,
        ) -> None:
            self.actions = list(actions or [])
            self.terminate_session = terminate_session
            self.behavior_policy_version = behavior_policy_version
            self.value_estimates = list(value_estimates or [])
            self.plan_updates = list(plan_updates or [])

    class HumanoidMotionFrame:
        """Tiny stand-in for humanoid.HumanoidMotionFrame."""

        def __init__(self, **kwargs: object) -> None:
            self.__dict__.update(kwargs)

    class HumanoidMotionReferenceSpec:
        """Tiny stand-in for humanoid.HumanoidMotionReferenceSpec."""

        def __init__(self, **kwargs: object) -> None:
            self.__dict__.update(kwargs)

    class HumanoidReferenceDecodeContext:
        """Tiny stand-in for humanoid.HumanoidReferenceDecodeContext."""

        def __init__(self, **kwargs: object) -> None:
            self.__dict__.update(kwargs)

    class HumanoidPlanUpdate:
        """Tiny stand-in for humanoid.HumanoidPlanUpdate."""

        def __init__(self, **kwargs: object) -> None:
            self.decode_context = kwargs.pop("decode_context", None)
            self.reference_sha256 = kwargs.pop("reference_sha256", "")
            self.__dict__.update(kwargs)

        def HasField(self, name: str) -> bool:
            """Return protobuf-like message presence for decode_context."""
            if name != "decode_context":
                raise ValueError(name)
            return self.decode_context is not None

    class HumanoidEnvValue:
        """Tiny stand-in for humanoid.HumanoidEnvValue."""

        def __init__(self, env_id: int = 0, value: float = 0.0) -> None:
            self.env_id = env_id
            self.value = value

    class HumanoidPolicySessionRequest:
        """Tiny stand-in for humanoid.HumanoidPolicySessionRequest."""

    class HumanoidRenderState:
        """Tiny stand-in whose receipt lets transport tests detect rewrites."""

        def __init__(self, **kwargs: object) -> None:
            """Retain the shared-contract constructor fields."""
            self.__dict__.update(kwargs)

        def canonical_sha256(self) -> str:
            """Return the fixture receipt used by strict camera tests."""
            return "e" * 64

    def humanoid_image_sha256(image_bytes: bytes) -> str:
        """Match the shared encoded-image receipt without state encoding."""
        return hashlib.sha256(image_bytes).hexdigest()

    def humanoid_render_receipt_sha256(**_: object) -> str:
        """Return the fixture's shared combined receipt."""
        return "f" * 64

    def humanoid_render_receipt_v2_sha256(**values: object) -> str:
        """Return the fixture's renderer-evidence-bound receipt."""
        names = (
            "render_state_sha256",
            "camera_contract_sha256",
            "image_sha256",
            "image_format",
            "width",
            "height",
            "scene_fingerprint",
            "model_signature_sha256",
            "camera_to_world_sha256",
            "renderer_binding_sha256",
        )
        encoded = "\0".join(str(values[name]) for name in names).encode()
        return hashlib.sha256(b"test-render-receipt-v2\0" + encoded).hexdigest()

    def humanoid_reference_sha256(spec: object, update: object) -> str:
        """Expose distinct fixture identities for the shared v1/v2 branches."""
        del update
        return "2" * 64 if str(spec.decode_context_schema) else "1" * 64

    class HumanoidSessionCloseRequest:
        """Tiny stand-in for humanoid.HumanoidSessionCloseRequest."""

    class HumanoidSessionAbortRequest:
        """Tiny stand-in for humanoid.HumanoidSessionAbortRequest."""

    class HumanoidPolicyServiceServicer:
        """Tiny stand-in for generated humanoid policy servicer base."""

    def add_HumanoidPolicyServiceServicer_to_server(
        servicer: object, server: Any
    ) -> None:
        """Attach a humanoid policy servicer to a fake server."""
        server.servicer = servicer

    class _Repeated(list):
        """List with protobuf-style add()."""

        def __init__(self, item_type: type) -> None:
            """Store the item type to construct in add()."""
            super().__init__()
            self._item_type = item_type

        def add(self):
            """Append and return one item."""
            item = self._item_type()
            self.append(item)
            return item

    class _ServiceAddress:
        """Tiny stand-in for SimulationRequest.ServiceAddress."""

        def __init__(self) -> None:
            """Initialize address fields."""
            self.ip = ""
            self.port = 0

    class _DriverAddress:
        """Tiny stand-in for SimulationRequest.DriverAddress."""

        def __init__(self) -> None:
            """Initialize address fields."""
            self.ip = ""
            self.port = 0

    class _RolloutSpec:
        """Tiny stand-in for runtime.RolloutSpec."""

        def __init__(self) -> None:
            """Initialize rollout spec fields."""
            self.scenario_id = ""
            self.nr_rollouts = 0
            self.session_uuids: list[str] = []
            self.attempt_ids: list[str] = []
            self.scene_id = ""
            self.random_seed = 0
            self.expected_behavior_policy_version = ""

    class SimulationRequest:
        """Tiny stand-in for runtime.SimulationRequest."""

        def __init__(self) -> None:
            """Create repeated request fields."""
            self.available_drivers = _Repeated(_DriverAddress)
            self.rollout_specs = _Repeated(_RolloutSpec)
            self.n_concurrent_per_driver = 0
            self.available_humanoid_policies = _Repeated(_ServiceAddress)
            self.n_concurrent_per_humanoid_policy = 0

    class SimulationReturn:
        """Tiny stand-in for runtime.SimulationReturn."""

        def __init__(self) -> None:
            """Create repeated response fields."""
            self.rollout_returns = []

    class EgodriverServiceServicer:
        """Tiny stand-in for generated servicer base."""

    class RuntimeServiceStub:
        """Tiny stand-in for generated RuntimeService stub."""

        def __init__(self, channel: object) -> None:
            """Accept a channel."""
            self.channel = channel

    def add_EgodriverServiceServicer_to_server(servicer: object, server: Any) -> None:
        """Attach a servicer to a fake server."""
        server.servicer = servicer

    common_pb2.Empty = Empty
    common_pb2.PoseAtTime = PoseAtTime
    common_pb2.Pose = SimpleNamespace
    common_pb2.SessionRequestStatus = SessionRequestStatus
    common_pb2.Trajectory = Trajectory
    common_pb2.VersionId = VersionId
    common_pb2.Vec3 = Vec3
    common_pb2.Quat = Quat
    egodriver_pb2.DriveRequest = DriveRequest
    egodriver_pb2.DriveResponse = DriveResponse
    egodriver_pb2.DriveSessionCloseRequest = DriveSessionCloseRequest
    egodriver_pb2.DriveSessionRequest = DriveSessionRequest
    egodriver_pb2.GroundTruth = GroundTruth
    egodriver_pb2.GroundTruthRequest = GroundTruthRequest
    egodriver_pb2.RolloutCameraImage = RolloutCameraImage
    egodriver_pb2.RolloutEgoTrajectory = RolloutEgoTrajectory
    egodriver_pb2.Route = Route
    egodriver_pb2.RouteRequest = RouteRequest
    egodriver_pb2_grpc.EgodriverServiceServicer = EgodriverServiceServicer
    humanoid_pb2.HumanoidAction = HumanoidAction
    humanoid_pb2.HumanoidEnvState = HumanoidEnvState
    humanoid_pb2.HumanoidEnvValue = HumanoidEnvValue
    humanoid_pb2.HumanoidMotionFrame = HumanoidMotionFrame
    humanoid_pb2.HumanoidMotionReferenceSpec = HumanoidMotionReferenceSpec
    humanoid_pb2.HumanoidPlanUpdate = HumanoidPlanUpdate
    humanoid_pb2.HumanoidPolicyRequest = HumanoidPolicyRequest
    humanoid_pb2.HumanoidPolicyResponse = HumanoidPolicyResponse
    humanoid_pb2.HumanoidPolicySessionRequest = HumanoidPolicySessionRequest
    humanoid_pb2.HumanoidReferenceDecodeContext = HumanoidReferenceDecodeContext
    humanoid_pb2.HumanoidSessionAbortRequest = HumanoidSessionAbortRequest
    humanoid_pb2.HumanoidSessionCloseRequest = HumanoidSessionCloseRequest
    humanoid_pb2.HUMANOID_EXECUTION_MODE_DIRECT_ACTION = 1
    humanoid_pb2.HUMANOID_EXECUTION_MODE_MOTION_REFERENCE = 2
    humanoid_pb2.HUMANOID_POLICY_REQUEST_KIND_DIRECT_ACTION = 1
    humanoid_pb2.HUMANOID_POLICY_REQUEST_KIND_INITIAL_PLAN = 2
    humanoid_pb2.HUMANOID_POLICY_REQUEST_KIND_REPLAN_WITH_FEEDBACK = 3
    humanoid_pb2.HUMANOID_POLICY_REQUEST_KIND_FINALIZE_WITH_FEEDBACK = 4
    humanoid_contracts.HUMANOID_RENDER_STATE_SCHEMA = "humanoid_render_state_qpos.v1"
    humanoid_contracts.HUMANOID_RENDER_RECEIPT_SCHEMA = "humanoid_render_receipt.v1"
    humanoid_contracts.HUMANOID_RENDER_RECEIPT_V2_SCHEMA = "humanoid_render_receipt.v2"
    humanoid_contracts.HUMANOID_FULL_ROTATION_LOCAL_XY_DECODE_CONTEXT_SCHEMA = (
        "full_pelvis_rotation_local_xy_completed_z/v1"
    )
    humanoid_contracts.HUMANOID_REFERENCE_HASH_DOMAIN_V1 = b"motion-reference.v1\0"
    humanoid_contracts.HUMANOID_REFERENCE_HASH_DOMAIN_V2 = (
        b"motion-reference+decode-context.v2\0"
    )
    humanoid_contracts.HumanoidRenderState = HumanoidRenderState
    humanoid_contracts.humanoid_image_sha256 = humanoid_image_sha256
    humanoid_contracts.humanoid_reference_sha256 = humanoid_reference_sha256
    humanoid_contracts.humanoid_render_receipt_sha256 = humanoid_render_receipt_sha256
    humanoid_contracts.humanoid_render_receipt_v2_sha256 = (
        humanoid_render_receipt_v2_sha256
    )
    humanoid_pb2_grpc.HumanoidPolicyServiceServicer = HumanoidPolicyServiceServicer
    humanoid_pb2_grpc.add_HumanoidPolicyServiceServicer_to_server = (
        add_HumanoidPolicyServiceServicer_to_server
    )
    egodriver_pb2_grpc.add_EgodriverServiceServicer_to_server = (
        add_EgodriverServiceServicer_to_server
    )
    runtime_pb2.SimulationRequest = SimulationRequest
    runtime_pb2.SimulationReturn = SimulationReturn
    runtime_pb2_grpc.RuntimeServiceStub = RuntimeServiceStub
    sensorsim_pb2.CameraSpec = CameraSpec

    def _descriptor_message(*fields: str, nested=None):
        return SimpleNamespace(
            fields_by_name={
                field: SimpleNamespace(message_type=None, number=number)
                for number, field in enumerate(fields, start=1)
            },
            nested_types_by_name=nested or {},
        )

    humanoid_pb2.DESCRIPTOR = SimpleNamespace(
        message_types_by_name={
            "HumanoidSessionAbortRequest": _descriptor_message("session_uuid"),
            "HumanoidPolicySessionRequest": _descriptor_message(
                "joint_names",
                "observation_schema",
                "action_schema",
                "observation_terms",
                "attempt_id",
                "scene_id",
                "scenario_id",
                "random_seed",
                "execution_mode",
                "reference_spec",
                "policy_camera_spec",
            ),
            "HumanoidPolicyCameraSpec": _descriptor_message(
                "schema",
                "logical_id",
                "width",
                "height",
                "image_format",
                "max_frame_age_us",
                "contract_sha256",
            ),
            "HumanoidEnvState": _descriptor_message(
                "timestamp_us",
                "named_observations",
                "observation_schema",
                "reset_id",
            ),
            "HumanoidPolicyRequest": _descriptor_message(
                "bootstrap_only", "bootstrap_env_ids", "request_kind", "observation"
            ),
            "HumanoidObservation": _descriptor_message(
                "camera_images",
                "decision_id",
                "feedback_traces",
                "timestamp_us",
            ),
            "HumanoidCameraImage": _descriptor_message(
                "frame_start_us",
                "frame_end_us",
                "image_bytes",
                "logical_id",
                "env_id",
                "render_timestamp_us",
                "observation_decision_id",
                "render_qpos",
                "render_state_sha256",
                "camera_contract_sha256",
                "image_sha256",
                "render_receipt_sha256",
                "scene_fingerprint",
                "model_signature_sha256",
                "camera_to_world_sha256",
                "renderer_binding_sha256",
            ),
            "HumanoidPolicyResponse": _descriptor_message(
                "actions",
                "behavior_policy_version",
                "value_estimates",
                "plan_updates",
            ),
            "HumanoidPlanUpdate": _descriptor_message(
                "reference_id",
                "source_decision_id",
                "frames",
                "reference_sha256",
                "root_z_alignment_offset_m",
                "decode_context",
            ),
            "HumanoidMotionReferenceSpec": _descriptor_message(
                "schema",
                "joint_names",
                "frame_count",
                "sample_period_us",
                "control_ticks_per_policy_step",
                "decode_context_schema",
            ),
            "HumanoidReferenceDecodeContext": _descriptor_message(
                "schema",
                "chunk_base_quaternion_wxyz",
                "local_xy_from_frame_zero",
            ),
            "HumanoidRealizedControlTick": _descriptor_message(
                "control_tick_offset",
                "state",
                "active_reference_id",
                "reference_action_index",
                "active_reference_sha256",
                "applied_reference_sha256",
                "root_z_alignment_offset_m",
                "reward",
                "terminated",
                "truncated",
                "metrics",
                "control_episode_step",
            ),
            "HumanoidRealizedFeedbackTrace": _descriptor_message(
                "env_id", "source_decision_id", "ticks"
            ),
            "HumanoidStepResult": _descriptor_message(
                "state",
                "reward",
                "terminated",
                "truncated",
                "final_state",
                "episode_step",
                "control_ticks",
                "executed_control_ticks",
                "active_reference_sha256",
                "applied_reference_sha256",
            ),
        },
        enum_types_by_name={
            "HumanoidExecutionMode": SimpleNamespace(
                values_by_name={
                    "HUMANOID_EXECUTION_MODE_DIRECT_ACTION": object(),
                    "HUMANOID_EXECUTION_MODE_MOTION_REFERENCE": object(),
                }
            ),
            "HumanoidPolicyRequestKind": SimpleNamespace(
                values_by_name={
                    "HUMANOID_POLICY_REQUEST_KIND_DIRECT_ACTION": object(),
                    "HUMANOID_POLICY_REQUEST_KIND_INITIAL_PLAN": object(),
                    "HUMANOID_POLICY_REQUEST_KIND_REPLAN_WITH_FEEDBACK": object(),
                    "HUMANOID_POLICY_REQUEST_KIND_FINALIZE_WITH_FEEDBACK": object(),
                }
            ),
        },
        services_by_name={},
    )
    humanoid_messages = humanoid_pb2.DESCRIPTOR.message_types_by_name
    abort_request_descriptor = humanoid_messages["HumanoidSessionAbortRequest"]
    humanoid_pb2.DESCRIPTOR.services_by_name.update(
        {
            service_name: SimpleNamespace(
                methods_by_name={
                    "abort_session": SimpleNamespace(
                        input_type=abort_request_descriptor
                    )
                }
            )
            for service_name in (
                "HumanoidPolicyService",
                "HumanoidDynamicsService",
            )
        }
    )
    humanoid_messages["HumanoidPolicySessionRequest"].fields_by_name[
        "policy_camera_spec"
    ].message_type = humanoid_messages["HumanoidPolicyCameraSpec"]
    humanoid_messages["HumanoidObservation"].fields_by_name[
        "camera_images"
    ].message_type = humanoid_messages["HumanoidCameraImage"]
    humanoid_messages["HumanoidPlanUpdate"].fields_by_name[
        "decode_context"
    ].message_type = humanoid_messages["HumanoidReferenceDecodeContext"]
    runtime_pb2.DESCRIPTOR = SimpleNamespace(
        message_types_by_name={
            "RolloutSpec": _descriptor_message(
                "scenario_id",
                "session_uuids",
                "random_seed",
                "attempt_ids",
                "scene_id",
                "expected_behavior_policy_version",
            ),
            "SimulationReturn": _descriptor_message(
                nested={"RolloutReturn": _descriptor_message("behavior_policy_version")}
            ),
        }
    )

    sys.modules["alpasim_grpc"] = alpasim_grpc
    sys.modules["alpasim_grpc.v0"] = alpasim_grpc_v0
    sys.modules["alpasim_grpc.v0.common_pb2"] = common_pb2
    sys.modules["alpasim_grpc.v0.egodriver_pb2"] = egodriver_pb2
    sys.modules["alpasim_grpc.v0.egodriver_pb2_grpc"] = egodriver_pb2_grpc
    sys.modules["alpasim_grpc.v0.humanoid_pb2"] = humanoid_pb2
    sys.modules["alpasim_grpc.v0.humanoid_pb2_grpc"] = humanoid_pb2_grpc
    sys.modules["alpasim_grpc.v0.humanoid_contracts"] = humanoid_contracts
    sys.modules["alpasim_grpc.v0.runtime_pb2"] = runtime_pb2
    sys.modules["alpasim_grpc.v0.runtime_pb2_grpc"] = runtime_pb2_grpc
    sys.modules["alpasim_grpc.v0.sensorsim_pb2"] = sensorsim_pb2


install_alpasim_grpc_stubs()

from alpagym_runtime.alpasim.proto_conversion import (  # noqa: E402
    build_simulation_request_proto,
    drive_response_from_policy_output,
    ground_truth_from_proto,
    policy_input_from_tick_buffer,
)
from alpagym_runtime.alpasim.tick_buffer import TickBuffer  # noqa: E402
from alpagym_runtime.types import PolicyOutput, RolloutCalibration  # noqa: E402
from alpasim_grpc.v0.egodriver_pb2 import GroundTruth, RolloutCameraImage, Route  # noqa: E402


def test_simulation_request_carries_batch_scenes_generation_and_driver() -> None:
    """Builds one RuntimeService request for a batch of scene sessions."""
    request = build_simulation_request_proto(
        scene_ids=("scene_a", "scene_b", "scene_a"),
        n_generation=2,
        driver_host="localhost",
        driver_port=50052,
        n_concurrent_per_driver=3,
    )

    assert [spec.scenario_id for spec in request.rollout_specs] == [
        "scene_a",
        "scene_b",
        "scene_a",
    ]
    assert [spec.nr_rollouts for spec in request.rollout_specs] == [2, 2, 2]
    assert request.available_drivers[0].ip == "localhost"
    assert request.available_drivers[0].port == 50052
    assert request.n_concurrent_per_driver == 3


def test_simulation_request_carries_humanoid_policy_endpoint() -> None:
    """Humanoid runtime requests use available_humanoid_policies, not egodriver."""
    request = build_simulation_request_proto(
        scene_ids=("stairs_scene",),
        n_generation=1,
        humanoid_policy_host="localhost",
        humanoid_policy_port=50057,
        n_concurrent_per_humanoid_policy=2,
        session_uuid="humanoid-session",
        expected_behavior_policy_version=7,
        humanoid_scenario_ids=("ascend",),
    )

    assert len(request.available_drivers) == 0
    assert request.available_humanoid_policies[0].ip == "localhost"
    assert request.available_humanoid_policies[0].port == 50057
    assert request.n_concurrent_per_humanoid_policy == 2
    assert request.rollout_specs[0].scene_id == "stairs_scene"
    assert request.rollout_specs[0].scenario_id == "ascend"
    assert list(request.rollout_specs[0].session_uuids) == ["humanoid-session"]
    legacy_seed = (
        int.from_bytes(
            hashlib.sha256(b"alpagym-humanoid-v1:humanoid-session").digest()[:8],
            "big",
        )
        or 1
    )
    assert request.rollout_specs[0].random_seed == legacy_seed
    assert request.rollout_specs[0].expected_behavior_policy_version == "7"


@pytest.mark.parametrize(
    "behavior_version",
    [-1, True, 1.5],
)
def test_simulation_request_rejects_invalid_behavior_version(
    behavior_version: object,
) -> None:
    with pytest.raises(ValueError, match="expected_behavior_policy_version"):
        build_simulation_request_proto(
            scene_ids=("stairs_scene",),
            n_generation=1,
            humanoid_policy_host="localhost",
            humanoid_policy_port=50057,
            n_concurrent_per_humanoid_policy=1,
            session_uuid="humanoid-session",
            expected_behavior_policy_version=behavior_version,  # type: ignore[arg-type]
            humanoid_scenario_ids=("ascend",),
        )


def test_simulation_request_threads_explicit_humanoid_random_seed() -> None:
    """An explicit panel seed bypasses the legacy session-UUID hash."""
    request = build_simulation_request_proto(
        scene_ids=("stairs_scene",),
        n_generation=1,
        humanoid_policy_host="localhost",
        humanoid_policy_port=50057,
        n_concurrent_per_humanoid_policy=1,
        session_uuid="humanoid-session",
        random_seed=0,
        humanoid_scenario_ids=("ascend",),
    )

    assert request.rollout_specs[0].random_seed == 0


@pytest.mark.parametrize("random_seed", [-1, 1 << 64, True, 1.5])
def test_simulation_request_rejects_non_uint64_random_seed(random_seed: object) -> None:
    """Explicit rollout seeds are strictly typed and bounded by the proto uint64 ABI."""
    with pytest.raises(ValueError, match="random_seed must be a uint64"):
        build_simulation_request_proto(
            scene_ids=("stairs_scene",),
            n_generation=1,
            humanoid_policy_host="localhost",
            humanoid_policy_port=50057,
            n_concurrent_per_humanoid_policy=1,
            session_uuid="humanoid-session",
            random_seed=random_seed,  # type: ignore[arg-type]
            humanoid_scenario_ids=("ascend",),
        )


def test_simulation_request_threads_session_uuid_for_per_rollout_dispatch() -> None:
    """Per-rollout streaming dispatch pins the session uuid on `RolloutSpec.session_uuids`."""
    request = build_simulation_request_proto(
        scene_ids=("scene_a",),
        n_generation=1,
        driver_host="localhost",
        driver_port=50052,
        n_concurrent_per_driver=1,
        session_uuid="abc-123",
    )

    assert list(request.rollout_specs[0].session_uuids) == ["abc-123"]


def test_simulation_request_rejects_session_uuid_outside_per_rollout_shape() -> None:
    """`session_uuid` requires the request to describe exactly one rollout of one scene."""
    with pytest.raises(ValueError, match="session_uuid is only valid"):
        build_simulation_request_proto(
            scene_ids=("scene_a", "scene_b"),
            n_generation=1,
            driver_host="localhost",
            driver_port=50052,
            n_concurrent_per_driver=1,
            session_uuid="abc-123",
        )
    with pytest.raises(ValueError, match="session_uuid is only valid"):
        build_simulation_request_proto(
            scene_ids=("scene_a",),
            n_generation=2,
            driver_host="localhost",
            driver_port=50052,
            n_concurrent_per_driver=1,
            session_uuid="abc-123",
        )


def test_policy_input_from_tick_buffer_surfaces_buffered_observations() -> None:
    """Snapshots tick-local + sticky proto fields into a populated PolicyInput."""
    calibration: RolloutCalibration = ()
    tick_buffer = TickBuffer(
        camera_images=[
            RolloutCameraImage.CameraImage(
                logical_id="front",
                image_bytes=b"abc",
                frame_end_us=99,
            )
        ],
        ego_trajectory=cast(
            Any,
            SimpleNamespace(
                poses=[
                    SimpleNamespace(
                        timestamp_us=90,
                        pose=SimpleNamespace(
                            vec=SimpleNamespace(x=1.0, y=2.0, z=3.0),
                            quat=SimpleNamespace(w=1.0, x=0.0, y=0.0, z=0.0),
                        ),
                    )
                ]
            ),
        ),
        route=Route(
            timestamp_us=80,
            waypoints=[
                SimpleNamespace(x=10.0, y=20.0, z=0.0),
                SimpleNamespace(x=11.0, y=21.0, z=0.0),
            ],
        ),
    )

    policy_input = policy_input_from_tick_buffer(
        step_index=3,
        time_now_us=100,
        time_query_us=200,
        calibration=calibration,
        tick_buffer=tick_buffer,
    )

    assert policy_input.step_index == 3
    assert policy_input.time_now_us == 100
    assert policy_input.time_query_us == 200
    assert policy_input.calibration is calibration
    assert len(policy_input.camera_images) == 1
    assert policy_input.camera_images[0].logical_id == "front"
    assert policy_input.camera_images[0].image_bytes == b"abc"
    assert policy_input.ego_trajectory.poses[0].pose.vec.x == 1.0
    assert policy_input.route_timestamp_us == 80
    assert policy_input.route_waypoints[1].y == 21.0


def test_policy_input_from_tick_buffer_rejects_missing_ego_trajectory() -> None:
    """Refuses to build a `PolicyInput` when AlpaSim has not submitted ego data."""
    with pytest.raises(ValueError, match="has no ego trajectory"):
        policy_input_from_tick_buffer(
            step_index=0,
            time_now_us=10,
            time_query_us=20,
            calibration=(),
            tick_buffer=TickBuffer(),
        )


def test_policy_input_from_tick_buffer_allows_empty_cameras() -> None:
    """Buffers without cameras still yield a `PolicyInput` once ego and route are set."""
    policy_input = policy_input_from_tick_buffer(
        step_index=0,
        time_now_us=10,
        time_query_us=20,
        calibration=(),
        tick_buffer=TickBuffer(
            ego_trajectory=cast(
                Any,
                SimpleNamespace(
                    poses=[
                        SimpleNamespace(
                            timestamp_us=10,
                            pose=SimpleNamespace(
                                vec=SimpleNamespace(x=0.0, y=0.0, z=0.0),
                                quat=SimpleNamespace(w=1.0, x=0.0, y=0.0, z=0.0),
                            ),
                        )
                    ]
                ),
            ),
            route=Route(
                timestamp_us=5,
                waypoints=[SimpleNamespace(x=0.0, y=0.0, z=0.0)],
            ),
        ),
    )

    assert policy_input.camera_images == ()
    assert len(policy_input.ego_trajectory.poses) == 1
    assert policy_input.ego_trajectory.poses[0].timestamp_us == 10
    assert policy_input.route_timestamp_us == 5
    assert policy_input.route_waypoints[0].x == 0.0


def test_policy_input_from_tick_buffer_rejects_missing_route() -> None:
    """Refuses to build a `PolicyInput` when AlpaSim has not submitted a route."""
    with pytest.raises(ValueError, match="has no route"):
        policy_input_from_tick_buffer(
            step_index=0,
            time_now_us=10,
            time_query_us=20,
            calibration=(),
            tick_buffer=TickBuffer(
                ego_trajectory=cast(
                    Any,
                    SimpleNamespace(
                        poses=[
                            SimpleNamespace(
                                timestamp_us=10,
                                pose=SimpleNamespace(
                                    vec=SimpleNamespace(x=0.0, y=0.0, z=0.0),
                                    quat=SimpleNamespace(w=1.0, x=0.0, y=0.0, z=0.0),
                                ),
                            )
                        ]
                    ),
                ),
            ),
        )


def test_ground_truth_from_proto_carries_trajectory_and_anchor_timestamp() -> None:
    """Forwards trajectory poses and the rig anchor timestamp into `GroundTruth`."""
    proto = GroundTruth(
        trajectory=cast(
            Any,
            SimpleNamespace(
                poses=[
                    SimpleNamespace(
                        timestamp_us=70,
                        pose=SimpleNamespace(
                            vec=SimpleNamespace(x=5.0, y=6.0, z=7.0),
                            quat=SimpleNamespace(w=1.0, x=0.0, y=0.0, z=0.0),
                        ),
                    )
                ]
            ),
        ),
        timestamp_us=12345,
    )

    ground_truth = ground_truth_from_proto(proto)

    assert ground_truth.timestamp_us == 12345
    assert ground_truth.ego_trajectory.poses[0].pose.vec.z == 7.0


def test_drive_response_from_policy_output_anchors_at_current_ego_pose() -> None:
    """Serializes policy-owned trajectory rows, including the current-pose row."""
    policy_input = policy_input_from_tick_buffer(
        step_index=0,
        time_now_us=1_000,
        time_query_us=400_000,
        calibration=(),
        tick_buffer=TickBuffer(
            ego_trajectory=cast(
                Any,
                SimpleNamespace(
                    poses=[
                        SimpleNamespace(
                            timestamp_us=1_000,
                            pose=SimpleNamespace(
                                vec=SimpleNamespace(x=10.0, y=20.0, z=30.0),
                                quat=SimpleNamespace(w=1.0, x=0.0, y=0.0, z=0.0),
                            ),
                        )
                    ]
                ),
            ),
            route=Route(
                timestamp_us=1_000,
                waypoints=[SimpleNamespace(x=0.0, y=0.0, z=0.0)],
            ),
        ),
    )
    chosen_xyz = torch.tensor(
        [[10.0, 20.0, 30.0], [11.0, 22.0, 33.0], [14.0, 25.0, 36.0]],
        dtype=torch.float32,
    )
    chosen_quat = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    chosen_dt_us = torch.tensor([0, 100_000, 200_000], dtype=torch.int64)
    output = PolicyOutput(
        chosen_xyz=chosen_xyz,
        chosen_quat=chosen_quat,
        chosen_dt_us=chosen_dt_us,
    )

    response = drive_response_from_policy_output(policy_input, output)

    timestamps = [pose.timestamp_us for pose in response.trajectory.poses]
    assert timestamps == [1_000, 101_000, 201_000]
    assert response.trajectory.poses[0].pose.vec.x == 10.0
    assert response.trajectory.poses[0].pose.vec.y == 20.0
    assert response.trajectory.poses[0].pose.vec.z == 30.0
    assert response.trajectory.poses[1].pose.vec.x == 11.0
    assert response.trajectory.poses[2].pose.vec.z == 36.0
    assert response.trajectory.poses[2].pose.quat.y == 1.0


def test_drive_response_from_policy_output_rejects_mismatched_leading_dims() -> None:
    """Rejects chosen tensors with mismatched `[T]` leading dims."""
    policy_input = policy_input_from_tick_buffer(
        step_index=0,
        time_now_us=0,
        time_query_us=1,
        calibration=(),
        tick_buffer=TickBuffer(
            ego_trajectory=cast(
                Any,
                SimpleNamespace(
                    poses=[
                        SimpleNamespace(
                            timestamp_us=0,
                            pose=SimpleNamespace(
                                vec=SimpleNamespace(x=0.0, y=0.0, z=0.0),
                                quat=SimpleNamespace(w=1.0, x=0.0, y=0.0, z=0.0),
                            ),
                        )
                    ]
                ),
            ),
            route=Route(
                timestamp_us=0,
                waypoints=[SimpleNamespace(x=0.0, y=0.0, z=0.0)],
            ),
        ),
    )
    chosen_xyz = torch.zeros(4, 3, dtype=torch.float32)
    chosen_quat = torch.zeros(4, 4, dtype=torch.float32)
    chosen_quat[:, 0] = 1.0
    chosen_dt_us = torch.tensor([1, 2, 3], dtype=torch.int64)
    output = PolicyOutput(
        chosen_xyz=chosen_xyz,
        chosen_quat=chosen_quat,
        chosen_dt_us=chosen_dt_us,
    )

    with pytest.raises(ValueError, match="Mismatched leading dims"):
        drive_response_from_policy_output(policy_input, output)
