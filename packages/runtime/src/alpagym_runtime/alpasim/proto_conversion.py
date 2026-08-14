# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402

from alpasim_grpc.v0.common_pb2 import Pose as ProtoPose, PoseAtTime, Trajectory as ProtoTrajectory
from alpasim_grpc.v0.egodriver_pb2 import (
    DriveResponse,
    DriveSessionRequest,
    GroundTruth as ProtoGroundTruth,
    RolloutCameraImage,
    Route,
)
from alpasim_grpc.v0.runtime_pb2 import SimulationRequest
from alpasim_grpc.v0.sensorsim_pb2 import CameraSpec

from alpagym_runtime.alpasim.tick_buffer import TickBuffer
from alpagym_runtime.perf.instrument.scope import measure_perf
from alpagym_runtime.types import (
    CameraCalibration,
    CameraImage,
    CameraIntrinsics,
    EgoPose,
    GroundTruth,
    PolicyInput,
    PolicyOutput,
    Pose,
    Quaternion,
    RolloutCalibration,
    RouteWaypoint,
    Trajectory,
    Vec3,
)


def build_simulation_request_proto(
    scene_ids: tuple[str, ...],
    n_generation: int,
    driver_host: str | None = None,
    driver_port: int | None = None,
    n_concurrent_per_driver: int = 0,
    session_uuid: str | None = None,
    humanoid_policy_host: str | None = None,
    humanoid_policy_port: int | None = None,
    n_concurrent_per_humanoid_policy: int = 0,
) -> SimulationRequest:
    """Build one RuntimeService simulate request."""
    if not scene_ids:
        raise ValueError("scene_ids must not be empty")
    if n_generation < 1:
        raise ValueError("n_generation must be at least 1")
    # When `session_uuid` is set, the streaming worker is dispatching a
    # single rollout of a single scene and wants AlpaSim to open the policy
    # session with our predetermined uuid instead of allocating one.
    if session_uuid is not None and (n_generation != 1 or len(scene_ids) != 1):
        raise ValueError(
            "session_uuid is only valid for a single-rollout single-scene request "
            f"(n_generation={n_generation}, scene_ids={scene_ids})"
        )

    uses_driver = (
        driver_host is not None or driver_port is not None or n_concurrent_per_driver > 0
    )
    uses_humanoid_policy = (
        humanoid_policy_host is not None
        or humanoid_policy_port is not None
        or n_concurrent_per_humanoid_policy > 0
    )
    if uses_driver == uses_humanoid_policy:
        raise ValueError(
            "build_simulation_request_proto requires exactly one policy endpoint kind: "
            "driver or humanoid_policy"
        )

    simulation_request = SimulationRequest()
    if uses_driver:
        if driver_host is None or driver_port is None:
            raise ValueError("driver_host and driver_port are required for driver requests")
        if n_concurrent_per_driver < 1:
            raise ValueError("n_concurrent_per_driver must be at least 1")
        driver = simulation_request.available_drivers.add()
        driver.ip = driver_host
        driver.port = driver_port
        simulation_request.n_concurrent_per_driver = n_concurrent_per_driver
    else:
        if humanoid_policy_host is None or humanoid_policy_port is None:
            raise ValueError(
                "humanoid_policy_host and humanoid_policy_port are required for humanoid requests"
            )
        if n_concurrent_per_humanoid_policy < 1:
            raise ValueError("n_concurrent_per_humanoid_policy must be at least 1")
        humanoid_policy = simulation_request.available_humanoid_policies.add()
        humanoid_policy.ip = humanoid_policy_host
        humanoid_policy.port = humanoid_policy_port
        simulation_request.n_concurrent_per_humanoid_policy = n_concurrent_per_humanoid_policy

    for scene_id in scene_ids:
        rollout_spec = simulation_request.rollout_specs.add()
        rollout_spec.scenario_id = scene_id
        rollout_spec.nr_rollouts = n_generation
        if session_uuid is not None:
            rollout_spec.session_uuids.append(session_uuid)
    return simulation_request


@measure_perf("driver/input_build", category="compute_cpu")
def policy_input_from_tick_buffer(
    step_index: int,
    time_now_us: int,
    time_query_us: int,
    calibration: RolloutCalibration,
    tick_buffer: TickBuffer,
) -> PolicyInput:
    """Snapshot one drive tick into a `PolicyInput`."""
    if tick_buffer.ego_trajectory is None:
        raise ValueError(
            f"AlpaSim drive tick at step {step_index} (time_now_us={int(time_now_us)}) "
            "has no ego trajectory; submit_egomotion_observation must precede drive()."
        )
    if tick_buffer.route is None:
        raise ValueError(
            f"AlpaSim drive tick at step {step_index} (time_now_us={int(time_now_us)}) "
            "has no route; submit_route must precede drive()."
        )
    return PolicyInput(
        step_index=step_index,
        time_now_us=int(time_now_us),
        time_query_us=int(time_query_us),
        camera_images=tuple(_camera_image_from_proto(image) for image in tick_buffer.camera_images),
        ego_trajectory=_trajectory_from_proto(tick_buffer.ego_trajectory),
        route_waypoints=_route_waypoints_from_proto(tick_buffer.route),
        route_timestamp_us=_route_timestamp_us(tick_buffer.route),
        calibration=calibration,
    )


@measure_perf("driver/output_build", category="compute_cpu")
def drive_response_from_policy_output(
    policy_input: PolicyInput,
    policy_output: PolicyOutput,
) -> DriveResponse:
    """Convert a policy output into an AlpaSim `DriveResponse`."""
    chosen_xyz = policy_output.chosen_xyz
    chosen_quat = policy_output.chosen_quat
    chosen_dt_us = policy_output.chosen_dt_us

    horizon = chosen_xyz.shape[0]
    if chosen_quat.shape[0] != horizon or chosen_dt_us.shape[0] != horizon:
        raise ValueError(
            "Mismatched leading dims for chosen tensors: "
            f"chosen_xyz={tuple(chosen_xyz.shape)}, "
            f"chosen_quat={tuple(chosen_quat.shape)}, "
            f"chosen_dt_us={tuple(chosen_dt_us.shape)}"
        )

    response = DriveResponse()
    time_now_us = int(policy_input.time_now_us)
    for i in range(horizon):
        pose_at_time = PoseAtTime(timestamp_us=time_now_us + int(chosen_dt_us[i].item()))
        pose_at_time.pose.vec.x = float(chosen_xyz[i, 0].item())
        pose_at_time.pose.vec.y = float(chosen_xyz[i, 1].item())
        pose_at_time.pose.vec.z = float(chosen_xyz[i, 2].item())
        pose_at_time.pose.quat.w = float(chosen_quat[i, 0].item())
        pose_at_time.pose.quat.x = float(chosen_quat[i, 1].item())
        pose_at_time.pose.quat.y = float(chosen_quat[i, 2].item())
        pose_at_time.pose.quat.z = float(chosen_quat[i, 3].item())
        response.trajectory.poses.append(pose_at_time)
    return response


def _camera_image_from_proto(proto: RolloutCameraImage.CameraImage) -> CameraImage:
    """Convert a proto-like camera image into an AlpaGym camera image."""
    return CameraImage(
        logical_id=str(proto.logical_id),
        image_bytes=bytes(proto.image_bytes),
        frame_end_us=int(proto.frame_end_us),
    )


def _trajectory_from_proto(proto: ProtoTrajectory) -> Trajectory:
    """Convert a proto trajectory into an AlpaGym trajectory (sorted by timestamp)."""
    poses = [
        EgoPose(
            timestamp_us=int(pose_at_time.timestamp_us),
            pose=_pose_from_proto(pose_at_time.pose),
        )
        for pose_at_time in proto.poses
    ]
    poses.sort(key=lambda pose: pose.timestamp_us)
    return Trajectory(poses=tuple(poses))


def _route_waypoints_from_proto(route: Route) -> tuple[RouteWaypoint, ...]:
    """Convert proto-like route waypoints into AlpaGym route waypoints."""
    return tuple(
        RouteWaypoint(
            x=float(waypoint.x),
            y=float(waypoint.y),
            z=float(waypoint.z),
        )
        for waypoint in route.waypoints
    )


def _route_timestamp_us(route: Route) -> int:
    """Return the route timestamp from a proto-like route."""
    return int(route.timestamp_us)


def ground_truth_from_proto(proto: ProtoGroundTruth) -> GroundTruth:
    """Convert proto-level ground truth into `GroundTruth`."""
    return GroundTruth(
        ego_trajectory=_trajectory_from_proto(proto.trajectory),
        timestamp_us=int(proto.timestamp_us),
    )


def calibration_from_proto(proto: DriveSessionRequest.RolloutSpec) -> RolloutCalibration:
    """Convert session rollout camera metadata into AlpaGym calibration.

    AlpaSim sends calibration once at session start.
    """
    return tuple(
        CameraCalibration(
            name=str(camera.logical_id),
            logical_id=str(camera.logical_id),
            intrinsics=_camera_intrinsics_from_proto(camera.intrinsics),
            extrinsic_pose=_pose_from_proto(camera.rig_to_camera),
        )
        for camera in proto.vehicle.available_cameras
    )


def _camera_intrinsics_from_proto(proto: CameraSpec) -> CameraIntrinsics:
    """Convert pinhole camera intrinsics from AlpaSim camera metadata."""
    pinhole = proto.opencv_pinhole_param
    return CameraIntrinsics(
        fx=float(pinhole.focal_length_x),
        fy=float(pinhole.focal_length_y),
        cx=float(pinhole.principal_point_x),
        cy=float(pinhole.principal_point_y),
    )


def _pose_from_proto(proto: ProtoPose) -> Pose:
    """Convert an AlpaSim pose proto into an AlpaGym pose."""
    return Pose(
        vec=Vec3(
            x=float(proto.vec.x),
            y=float(proto.vec.y),
            z=float(proto.vec.z),
        ),
        quat=Quaternion(
            w=float(proto.quat.w),
            x=float(proto.quat.x),
            y=float(proto.quat.y),
            z=float(proto.quat.z),
        ),
    )
