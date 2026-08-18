# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, TypeAlias, runtime_checkable

import torch

from alpagym_runtime.replay import PolicyReplayData


@dataclass(frozen=True)
class Vec3:
    """Three-dimensional vector in simulator coordinates."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass(frozen=True)
class Quaternion:
    """Quaternion rotation."""

    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass(frozen=True)
class Pose:
    """Six-degree pose with translation and rotation."""

    vec: Vec3 = field(default_factory=Vec3)
    quat: Quaternion = field(default_factory=Quaternion)


@dataclass(frozen=True)
class EgoPose:
    """Timestamped ego pose from the simulator."""

    timestamp_us: int
    pose: Pose = field(default_factory=Pose)


@dataclass(frozen=True)
class Trajectory:
    """Sequence of timestamped ego poses."""

    poses: tuple[EgoPose, ...] = ()


@dataclass(frozen=True)
class CameraImage:
    """Raw camera image submitted for one simulator step."""

    logical_id: str
    image_bytes: bytes
    frame_end_us: int


@dataclass(frozen=True)
class RouteWaypoint:
    """Route waypoint in the ego-rig frame at `route_timestamp_us`."""

    x: float
    y: float
    z: float = 0.0


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera intrinsic parameters."""

    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class CameraCalibration:
    """Calibration for one simulator camera."""

    name: str
    logical_id: str
    intrinsics: CameraIntrinsics | None = None
    extrinsic_pose: Pose | None = None


RolloutCalibration: TypeAlias = tuple[CameraCalibration, ...]
"""Session-level camera calibration: per-session ordered camera list."""


@dataclass(frozen=True)
class GroundTruth:
    """Per-session ground-truth ego trajectory from the recording."""

    ego_trajectory: Trajectory
    timestamp_us: int = 0


@dataclass(frozen=True)
class PolicyInput:
    """Snapshot of one simulator drive tick handed to `Policy.step`."""

    step_index: int
    time_now_us: int
    time_query_us: int
    camera_images: tuple[CameraImage, ...]
    ego_trajectory: Trajectory
    route_waypoints: tuple[RouteWaypoint, ...]
    route_timestamp_us: int
    calibration: RolloutCalibration


@dataclass(frozen=True)
class PolicyOutput:
    """Policy output for one simulator drive tick."""

    chosen_xyz: torch.Tensor
    chosen_quat: torch.Tensor
    chosen_dt_us: torch.Tensor
    chosen_logprob: torch.Tensor | None = None
    replay_data: PolicyReplayData | None = None
    all_pred_xyz: torch.Tensor | None = None
    all_pred_quat: torch.Tensor | None = None
    model_extra: dict[str, Any] | None = None


@dataclass(frozen=True)
class ChosenTrajectory:
    """One trajectory selected from a `ModelOutput`."""

    set_ix: int
    sample_ix: int
    xyz: torch.Tensor
    rot: torch.Tensor
    dt_us: torch.Tensor
    time_now_us: int
    ego_pose_at_choice: Pose


@runtime_checkable
class Policy(Protocol):
    """Per-session policy contract consumed by the simulator layer (egodriver gRPC servicer)."""

    def step(self, policy_input: PolicyInput) -> PolicyOutput:
        """Run one drive tick and return the chosen trajectory."""

    def close(self) -> None:
        """Release any per-session resources held by the policy."""


@dataclass(frozen=True)
class EpisodeMetrics:
    """Metrics returned by the simulator for a completed episode."""

    aggregated: Mapping[str, float]
    dense: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RewardResult:
    """Reward value and supporting metrics for one episode."""

    total: float
    report_metrics: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class EpisodeOutput:
    """Completed simulator session before persistence."""

    scene_id: str
    session_uuid: str
    num_steps: int
    policy_outputs: tuple[PolicyOutput, ...]
    # Exact seed shared by simulator dynamics and per-lane policy sampling.
    # JSON integers preserve the full uint64 range; float-valued metric maps do not.
    rollout_seed: int | None = None
    executed_ego_trajectory: Trajectory = field(default_factory=Trajectory)
    route_waypoints: tuple[RouteWaypoint, ...] = ()
    metrics: EpisodeMetrics | None = None
    reward: RewardResult | None = None
    is_valid: bool = True

    def __post_init__(self) -> None:
        """Reject seeds that cannot round-trip through the rollout uint64 ABI."""
        if self.rollout_seed is not None and (
            isinstance(self.rollout_seed, bool)
            or not isinstance(self.rollout_seed, int)
            or not 0 <= self.rollout_seed <= (1 << 64) - 1
        ):
            raise ValueError("EpisodeOutput.rollout_seed must be a uint64 or null")


@dataclass(frozen=True)
class RolloutArtifact:
    """Completed rollout artifact paired with its transport-emitted handle."""

    handle: str
    episode: EpisodeOutput
