# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402

"""Humanoid policy gRPC server used by AlpaGym rollout workers."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import os
import re
import threading
from concurrent import futures
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterable, Mapping, Protocol, runtime_checkable

import grpc
import torch
from PIL import Image, UnidentifiedImageError
from alpagym_runtime.alpasim.grpc_import import (
    ensure_alpasim_grpc_source,
    ensure_humanoid_policy_camera_abi,
    ensure_humanoid_reference_decode_context_abi,
)

ensure_alpasim_grpc_source()
from alpagym_host.endpoint_registry import TopologyEndpoint
from alpasim_grpc.v0 import humanoid_contracts, humanoid_pb2
from alpasim_grpc.v0.common_pb2 import (
    Empty,
    Quat,
    SessionRequestStatus,
    Vec3,
    VersionId,
)
from alpasim_grpc.v0.humanoid_pb2 import (
    HUMANOID_EXECUTION_MODE_MOTION_REFERENCE,
    HUMANOID_POLICY_REQUEST_KIND_FINALIZE_WITH_FEEDBACK,
    HUMANOID_POLICY_REQUEST_KIND_INITIAL_PLAN,
    HUMANOID_POLICY_REQUEST_KIND_REPLAN_WITH_FEEDBACK,
    HumanoidAction,
    HumanoidEnvValue,
    HumanoidEnvState,
    HumanoidMotionFrame,
    HumanoidMotionReferenceSpec,
    HumanoidPlanUpdate,
    HumanoidPolicyRequest,
    HumanoidPolicyResponse,
    HumanoidPolicySessionRequest,
    HumanoidSessionAbortRequest,
    HumanoidSessionCloseRequest,
)
from alpasim_grpc.v0.humanoid_pb2_grpc import (
    HumanoidPolicyServiceServicer,
    add_HumanoidPolicyServiceServicer_to_server,
)

from alpagym_runtime.inference.inference_engine import (
    InferenceModelLease,
)
from alpagym_runtime.replay import PolicyReplayData
from alpagym_runtime.types import PolicyOutput

logger = logging.getLogger(__name__)

_LOWERCASE_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MOTION_REFERENCE_SCHEMA_H50 = "g1_motion_reference_29d_50hz_h50.v1"
HUMANOID_VISUAL_INPUT_MANIFEST_SCHEMA = "alpagym.humanoid_visual_input.v1"
HUMANOID_FEEDBACK_STATE_CONTRACT_SCHEMA = (
    "alpagym.humanoid_feedback_state.named_joint_order.v1"
)
MOTION_REFERENCE_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)


@dataclass(frozen=True)
class HumanoidCameraFrame:
    """One immutable camera frame routed to its matching humanoid env lane."""

    env_id: int
    frame_start_us: int
    frame_end_us: int
    logical_id: str
    image_bytes: bytes
    render_timestamp_us: int
    observation_decision_id: int
    render_qpos: tuple[float, ...]
    render_state_sha256: str
    camera_contract_sha256: str
    image_sha256: str
    render_receipt_sha256: str
    scene_fingerprint: str
    model_signature_sha256: str
    camera_to_world_sha256: str
    renderer_binding_sha256: str

    @property
    def policy_joint_position(self) -> tuple[float, ...]:
        """Return the image-paired 29-D joint state, never a later live state."""
        if len(self.render_qpos) != 36:
            raise ValueError("camera frame does not carry a 36-D G1 render_qpos")
        return self.render_qpos[7:]


@dataclass(frozen=True)
class HumanoidCameraFrameIdentity:
    """Compact audit identity for a camera frame; never contains pixels/qpos."""

    env_id: int
    frame_start_us: int
    frame_end_us: int
    logical_id: str
    byte_length: int
    sha256: str
    render_timestamp_us: int
    observation_decision_id: int
    render_state_sha256: str
    camera_contract_sha256: str
    image_sha256: str
    render_receipt_sha256: str
    scene_fingerprint: str
    model_signature_sha256: str
    camera_to_world_sha256: str
    renderer_binding_sha256: str


@dataclass(frozen=True)
class HumanoidPolicyCameraContract:
    """Immutable session contract for policy-visible humanoid RGB."""

    schema: str
    logical_id: str
    width: int
    height: int
    image_format: str
    max_frame_age_us: int
    contract_sha256: str


@dataclass(frozen=True)
class _HumanoidCameraRouting:
    """Policy-visible and all-packet views of one camera observation."""

    policy_frames_by_env: dict[int, tuple[HumanoidCameraFrame, ...]]
    all_frames_by_env: dict[int, tuple[HumanoidCameraFrame, ...]]


@dataclass(frozen=True)
class HumanoidPolicyInput:
    """One humanoid env-lane observation delivered to a policy."""

    session_uuid: str
    episode_id: int
    step_index: int
    timestamp_us: int
    env_id: int
    qpos: torch.Tensor
    qvel: torch.Tensor
    observation: torch.Tensor
    scalars: Mapping[str, float]
    camera_frames: tuple[HumanoidCameraFrame, ...] = ()
    camera_frame_identities: tuple[HumanoidCameraFrameIdentity, ...] = ()
    decision_id: int = 0
    feedback_trace: HumanoidRealizedFeedbackTrace | None = None
    bootstrap_requested: bool = False


@dataclass(frozen=True)
class HumanoidRealizedControlTick:
    """One exact post-controller-tick state and reference receipt."""

    control_tick_offset: int
    qpos: torch.Tensor
    qvel: torch.Tensor
    observation: torch.Tensor
    scalars: Mapping[str, float]
    timestamp_us: int
    active_reference_id: int
    reference_action_index: int
    active_reference_sha256: str
    applied_reference_sha256: str
    root_z_alignment_offset_m: float
    reward: float
    terminated: bool
    truncated: bool
    metrics: Mapping[str, float]
    control_episode_step: int


@dataclass(frozen=True)
class HumanoidRealizedFeedbackTrace:
    """The realized controller-tick prefix for one prior macro decision."""

    env_id: int
    source_decision_id: int
    ticks: tuple[HumanoidRealizedControlTick, ...]


@dataclass(frozen=True)
class HumanoidMotionReferenceFrame:
    """One 50 Hz frame in a policy-produced motion reference."""

    timestamp_us: int
    joint_position: torch.Tensor
    joint_velocity: torch.Tensor
    root_position: torch.Tensor
    root_quaternion_wxyz: torch.Tensor


@dataclass(frozen=True)
class HumanoidReferenceDecodeContext:
    """Policy-owned context needed to finish a reference at the physics boundary."""

    schema: str
    chunk_base_quaternion_wxyz: torch.Tensor
    local_xy_from_frame_zero: torch.Tensor


@dataclass(frozen=True)
class HumanoidMotionReference:
    """Typed fixed-horizon reference returned instead of a direct joint action."""

    reference_id: int
    source_decision_id: int
    frames: tuple[HumanoidMotionReferenceFrame, ...]
    root_z_alignment_offset_m: float
    decode_context: HumanoidReferenceDecodeContext | None = None


@dataclass(frozen=True)
class HumanoidPolicyStepOutput:
    """One humanoid env-lane action plus optional trainer replay payload."""

    env_id: int
    action: torch.Tensor | None = None
    motion_reference: HumanoidMotionReference | None = None
    logprob: torch.Tensor | None = None
    value: torch.Tensor | None = None
    replay_data: PolicyReplayData | None = None
    model_extra: dict[str, object] | None = None


@runtime_checkable
class HumanoidPolicy(Protocol):
    """Per-session humanoid policy contract consumed by the policy gRPC servicer."""

    def step(
        self,
        policy_inputs: tuple[HumanoidPolicyInput, ...],
        *,
        sample_actions: bool = True,
    ) -> tuple[HumanoidPolicyStepOutput, ...]:
        """Return one action for each env-lane input."""

    def close(self) -> None:
        """Release per-session resources."""


class SessionModelLeaseRegistry(Protocol):
    """Lifecycle surface used to bind immutable models to gRPC sessions."""

    def register_session_model_lease(
        self,
        session_uuid: str,
        lease: InferenceModelLease,
    ) -> None:
        """Register a behavior-model snapshot before session construction."""

    def release_session_model_lease(self, session_uuid: str) -> None:
        """Release a snapshot after close, failure, or retry."""


@dataclass(frozen=True)
class HumanoidSessionRecord:
    """Frozen per-session humanoid policy outputs drained by the rollout worker."""

    outputs: tuple[PolicyOutput, ...]
    final_bootstrap_values: Mapping[int, float]
    behavior_policy_version: int


@dataclass
class _Session:
    """Per-session mutable state held by ``HumanoidPolicyGrpcServicer``."""

    policy: HumanoidPolicy
    action_size: int
    observation_schema: str
    observation_terms: tuple[tuple[str, int], ...]
    joint_names: tuple[str, ...]
    execution_mode: int
    policy_camera_contract: HumanoidPolicyCameraContract | None = None
    reference_joint_names: tuple[str, ...] = ()
    reference_frame_count: int = 0
    reference_sample_period_us: int = 0
    reference_decode_context_schema: str = ""
    control_ticks_per_policy_step: int = 1
    behavior_policy_version: int = 0
    save_camera_dir: Path | None = None
    step_index: int = 0
    outputs: list[PolicyOutput] = field(default_factory=list)
    final_bootstrap_values: dict[int, float] = field(default_factory=dict)
    last_control_episode_steps: dict[int, int] = field(default_factory=dict)
    routed_current_camera_ledger: dict[
        tuple[int, int], list[HumanoidCameraFrameIdentity]
    ] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def consume_step_index(self) -> int:
        """Return the current step index and advance the session clock."""
        with self.lock:
            step_index = self.step_index
            self.step_index += 1
        return step_index

    def record_outputs(self, outputs: tuple[PolicyOutput, ...]) -> None:
        """Append policy outputs in env-lane order."""
        with self.lock:
            self.outputs.extend(outputs)

    def prior_current_camera_identities(
        self,
        policy_input: HumanoidPolicyInput,
    ) -> tuple[HumanoidCameraFrameIdentity, ...]:
        """Return the previously accepted strict-camera ledger for one episode lane."""

        if self.policy_camera_contract is None:
            return ()
        key = (int(policy_input.episode_id), int(policy_input.env_id))
        with self.lock:
            return tuple(self.routed_current_camera_ledger.get(key, ()))

    def record_validated_current_camera_identities(
        self,
        policy_inputs: tuple[HumanoidPolicyInput, ...],
    ) -> None:
        """Commit routed current identities only after all outputs validate."""

        if self.policy_camera_contract is None:
            return
        with self.lock:
            pending = {
                key: list(identities)
                for key, identities in self.routed_current_camera_ledger.items()
            }
            for policy_input in policy_inputs:
                if len(policy_input.camera_frames) != 1:
                    raise ValueError(
                        "strict policy-camera ledger requires one current frame"
                    )
                identity = _camera_frame_identity(policy_input.camera_frames[0])
                if identity.sha256 != identity.image_sha256:
                    raise ValueError(
                        "strict policy-camera ledger encoded-image identity changed"
                    )
                key = (int(policy_input.episode_id), int(policy_input.env_id))
                prior = pending.setdefault(key, [])
                if prior and (
                    identity.render_timestamp_us <= prior[-1].render_timestamp_us
                ):
                    raise ValueError(
                        "strict policy-camera ledger current frames must advance in time"
                    )
                if identity in prior:
                    raise ValueError(
                        "strict policy-camera ledger contains a duplicate current frame"
                    )
                prior.append(identity)
            self.routed_current_camera_ledger = pending

    def attach_feedback_traces(
        self,
        traces: Mapping[int, HumanoidRealizedFeedbackTrace],
        *,
        outer_truncated_env_ids: frozenset[int],
    ) -> None:
        """Join realized async controller intervals to their policy samples."""
        if not traces:
            return
        with self.lock:
            for env_id, trace in traces.items():
                match_index = None
                for index in reversed(range(len(self.outputs))):
                    extra = self.outputs[index].model_extra or {}
                    if (
                        int(extra.get("humanoid_env_id", -1)) == env_id
                        and int(extra.get("humanoid_decision_id", -1))
                        == trace.source_decision_id
                    ):
                        match_index = index
                        break
                if match_index is None:
                    raise ValueError(
                        "feedback trace has no recorded source plan: "
                        f"env_id={env_id}, decision_id={trace.source_decision_id}"
                    )
                output = self.outputs[match_index]
                if output.replay_data is None:
                    raise ValueError("feedback source plan has no replay payload")
                payload = dict(output.replay_data.payload)
                if "feedback_trace" in payload:
                    raise ValueError("source plan already has a feedback trace")
                reference_id = int(payload.get("reference_id", -1))
                reference_sha256 = _require_lowercase_sha256(
                    "source plan reference_sha256",
                    payload.get("reference_sha256", ""),
                )
                root_z_offset = float(
                    payload.get("root_z_alignment_offset_m", math.nan)
                )
                source_timestamp_us = int(output.chosen_dt_us[0].item())

                predecessor: PolicyOutput | None = None
                for index in reversed(range(match_index)):
                    extra = self.outputs[index].model_extra or {}
                    if int(extra.get("humanoid_env_id", -1)) == env_id:
                        predecessor = self.outputs[index]
                        break
                predecessor_identity: tuple[int, str, float, int] | None = None
                if predecessor is not None:
                    if predecessor.replay_data is None:
                        raise ValueError("predecessor plan has no replay payload")
                    predecessor_payload = predecessor.replay_data.payload
                    predecessor_identity = (
                        int(predecessor_payload.get("reference_id", -1)),
                        _require_lowercase_sha256(
                            "predecessor plan reference_sha256",
                            predecessor_payload.get("reference_sha256", ""),
                        ),
                        float(
                            predecessor_payload.get(
                                "root_z_alignment_offset_m", math.nan
                            )
                        ),
                        int(predecessor.chosen_dt_us[0].item()),
                    )

                seen_source = False
                active_segment: str | None = None
                applied_sha256_by_segment: dict[str, str] = {}
                for tick in trace.ticks:
                    active_identity = (
                        tick.active_reference_id,
                        tick.active_reference_sha256,
                    )
                    if active_identity == (reference_id, reference_sha256):
                        segment = "source"
                        expected_root_z_offset = root_z_offset
                        segment_timestamp_us = source_timestamp_us
                        seen_source = True
                    elif (
                        predecessor_identity is not None
                        and active_identity == predecessor_identity[:2]
                    ):
                        if seen_source:
                            raise ValueError(
                                "feedback reference sequence reversed from source "
                                "to predecessor"
                            )
                        segment = "predecessor"
                        expected_root_z_offset = predecessor_identity[2]
                        segment_timestamp_us = predecessor_identity[3]
                    else:
                        raise ValueError(
                            "feedback contains a third or unknown active reference"
                        )
                    if active_segment is not None and segment != active_segment:
                        if active_segment != "predecessor" or segment != "source":
                            raise ValueError(
                                "feedback reference sequence must be predecessor then source"
                            )
                    active_segment = segment
                    if not math.isclose(
                        tick.root_z_alignment_offset_m,
                        expected_root_z_offset,
                        rel_tol=0.0,
                        abs_tol=1.0e-6,
                    ):
                        raise ValueError(
                            "feedback root_z_alignment_offset_m does not match its "
                            "active plan"
                        )
                    action_timestamp_us = (
                        tick.timestamp_us - self.reference_sample_period_us
                    )
                    delta_us = action_timestamp_us - segment_timestamp_us
                    if delta_us < 0 or delta_us % self.reference_sample_period_us:
                        raise ValueError(
                            "feedback reference_action_index is off the source time grid"
                        )
                    expected_action_index = min(
                        delta_us // self.reference_sample_period_us,
                        self.reference_frame_count - 1,
                    )
                    if tick.reference_action_index != expected_action_index:
                        raise ValueError(
                            "feedback reference_action_index does not match its "
                            "active plan timestamp"
                        )
                    previous_applied_sha256 = applied_sha256_by_segment.setdefault(
                        segment, tick.applied_reference_sha256
                    )
                    if tick.applied_reference_sha256 != previous_applied_sha256:
                        raise ValueError(
                            "feedback applied_reference_sha256 changed within an "
                            "active-reference segment"
                        )
                if not seen_source and not (
                    trace.ticks[-1].terminated
                    or trace.ticks[-1].truncated
                    or env_id in outer_truncated_env_ids
                ):
                    raise ValueError(
                        "a predecessor-only feedback interval must end in "
                        "termination or truncation"
                    )
                payload["feedback_trace"] = _feedback_trace_payload(
                    trace,
                    joint_names=self.joint_names,
                )
                self.outputs[match_index] = replace(
                    output,
                    replay_data=replace(output.replay_data, payload=payload),
                )

    def advance_control_episode_steps(
        self,
        traces: Mapping[int, HumanoidRealizedFeedbackTrace],
    ) -> None:
        """Require controller tick identities to stay contiguous across macros."""
        with self.lock:
            updated = dict(self.last_control_episode_steps)
            for env_id, trace in traces.items():
                expected = updated.get(env_id, 0) + 1
                for tick in trace.ticks:
                    if tick.control_episode_step != expected:
                        raise ValueError(
                            "feedback control_episode_step is not contiguous: "
                            f"env_id={env_id}, expected={expected}, "
                            f"actual={tick.control_episode_step}"
                        )
                    expected += 1
                updated[env_id] = expected - 1
            self.last_control_episode_steps = updated

    def mark_outer_truncated(self, bootstrap_env_ids: frozenset[int]) -> None:
        """Mark horizon/any-lane truncation on the already attached source row."""
        if not bootstrap_env_ids:
            return
        with self.lock:
            for env_id in bootstrap_env_ids:
                match_index = None
                for index in reversed(range(len(self.outputs))):
                    extra = self.outputs[index].model_extra or {}
                    if int(extra.get("humanoid_env_id", -1)) == env_id:
                        match_index = index
                        break
                if match_index is None:
                    raise ValueError(f"no source plan to truncate for env_id={env_id}")
                output = self.outputs[match_index]
                if output.replay_data is None:
                    raise ValueError(
                        "outer-truncated source plan has no replay payload"
                    )
                payload = dict(output.replay_data.payload)
                trace = payload.get("feedback_trace")
                if not isinstance(trace, Mapping):
                    raise ValueError(
                        "outer truncation requires an attached feedback trace"
                    )
                ticks = trace.get("ticks")
                if not isinstance(ticks, list) or not ticks:
                    raise ValueError("outer truncation feedback trace is empty")
                if bool(ticks[-1]["terminated"]):
                    raise ValueError(
                        "terminated lane cannot request truncation bootstrap"
                    )
                payload["outer_truncated"] = True
                self.outputs[match_index] = replace(
                    output,
                    replay_data=replace(output.replay_data, payload=payload),
                )

    def record_final_values(
        self,
        outputs: tuple[HumanoidPolicyStepOutput, ...],
        bootstrap_env_ids: frozenset[int],
    ) -> None:
        """Store values only for truncated lanes after final feedback commit."""
        with self.lock:
            if self.final_bootstrap_values:
                raise ValueError(
                    "humanoid session received multiple bootstrap-only requests"
                )
            for output in outputs:
                env_id = int(output.env_id)
                if env_id not in bootstrap_env_ids:
                    if output.value is not None:
                        raise ValueError(
                            "terminated lane unexpectedly returned a bootstrap value"
                        )
                    continue
                if output.value is None:
                    raise ValueError("humanoid bootstrap output is missing a value")
                value = float(
                    torch.as_tensor(output.value, dtype=torch.float32)
                    .reshape(())
                    .item()
                )
                if not math.isfinite(value):
                    raise ValueError(
                        f"humanoid final value for env_id={env_id} is non-finite"
                    )
                self.final_bootstrap_values[env_id] = value
            if set(self.final_bootstrap_values) != set(bootstrap_env_ids):
                raise ValueError("humanoid final outputs do not cover truncated lanes")

    def get_record(self) -> HumanoidSessionRecord:
        """Freeze this session's outputs for the rollout worker."""
        with self.lock:
            if self.execution_mode == HUMANOID_EXECUTION_MODE_MOTION_REFERENCE:
                missing = [
                    index
                    for index, output in enumerate(self.outputs)
                    if output.replay_data is None
                    or "feedback_trace" not in output.replay_data.payload
                ]
                if missing:
                    raise ValueError(
                        "motion-reference session closed before every plan received "
                        f"physical feedback; missing rows={missing}"
                    )
            return HumanoidSessionRecord(
                outputs=tuple(self.outputs),
                final_bootstrap_values=dict(self.final_bootstrap_values),
                behavior_policy_version=self.behavior_policy_version,
            )


class ZeroHumanoidPolicy:
    """Deterministic no-op humanoid policy for wiring smoke tests."""

    def __init__(self, action_size: int) -> None:
        self._action_size = int(action_size)

    def step(
        self,
        policy_inputs: tuple[HumanoidPolicyInput, ...],
        *,
        sample_actions: bool = True,
    ) -> tuple[HumanoidPolicyStepOutput, ...]:
        del sample_actions
        return tuple(
            HumanoidPolicyStepOutput(
                env_id=policy_input.env_id,
                action=torch.zeros(self._action_size, dtype=torch.float32),
                logprob=torch.zeros((), dtype=torch.float32),
                value=torch.zeros((), dtype=torch.float32),
            )
            for policy_input in policy_inputs
        )

    def close(self) -> None:
        return None


class HumanoidPolicyGrpcServicer(HumanoidPolicyServiceServicer):
    """AlpaSim HumanoidPolicyService backed by per-session policy objects."""

    def __init__(
        self,
        policy_factory: Callable[[str, HumanoidPolicySessionRequest], HumanoidPolicy],
        model_lease_registry: SessionModelLeaseRegistry | None = None,
        *,
        require_policy_camera: bool = False,
    ) -> None:
        ensure_humanoid_reference_decode_context_abi()
        if require_policy_camera:
            ensure_humanoid_policy_camera_abi()
        self._policy_factory = policy_factory
        self._model_lease_registry = model_lease_registry
        self._require_policy_camera = require_policy_camera
        self._sessions: dict[str, _Session] = {}
        self._sessions_lock = threading.Lock()
        self._session_records: dict[str, HumanoidSessionRecord] = {}
        self._reserved_versions: dict[str, int] = {}

    def reserve_session(
        self,
        session_uuid: str,
        behavior_policy_version: int,
        model_lease: InferenceModelLease | None = None,
    ) -> None:
        """Bind queued simulator work to the exact rollout-weight version."""
        if not session_uuid or behavior_policy_version < 0:
            raise ValueError(
                "humanoid session reservation requires UUID and non-negative version"
            )
        with self._sessions_lock:
            if (
                session_uuid in self._reserved_versions
                or session_uuid in self._sessions
                or session_uuid in self._session_records
            ):
                raise ValueError(
                    f"humanoid session {session_uuid!r} is already reserved"
                )
            if self._model_lease_registry is None:
                if model_lease is not None:
                    raise ValueError(
                        "colocated humanoid sessions must not carry model leases"
                    )
            else:
                if model_lease is None:
                    raise ValueError(
                        "disaggregated humanoid sessions require a model lease"
                    )
                if model_lease.behavior_policy_version != behavior_policy_version:
                    raise ValueError(
                        "humanoid model lease version does not match the reserved "
                        "behavior version"
                    )
                self._model_lease_registry.register_session_model_lease(
                    session_uuid,
                    model_lease,
                )
            self._reserved_versions[session_uuid] = behavior_policy_version

    def start_session(
        self,
        request: HumanoidPolicySessionRequest,
        context: grpc.ServicerContext,
    ) -> SessionRequestStatus:
        del context
        session_uuid = str(request.session_uuid)
        with self._sessions_lock:
            if session_uuid not in self._reserved_versions:
                raise ValueError(f"humanoid session {session_uuid!r} was not reserved")
            behavior_policy_version = self._reserved_versions.pop(session_uuid)
            policy: HumanoidPolicy | None = None
            try:
                # Proto zero/missing preserves the legacy direct-action contract.
                execution_mode = int(getattr(request, "execution_mode", 0))
                reference_joint_names: tuple[str, ...] = ()
                reference_frame_count = 0
                reference_sample_period_us = 0
                reference_decode_context_schema = ""
                control_ticks_per_policy_step = 1
                if execution_mode == HUMANOID_EXECUTION_MODE_MOTION_REFERENCE:
                    (
                        reference_joint_names,
                        reference_frame_count,
                        reference_sample_period_us,
                        control_ticks_per_policy_step,
                        reference_decode_context_schema,
                    ) = _validate_motion_reference_session(request)
                else:
                    if _message_has_fields(getattr(request, "reference_spec", None)):
                        raise ValueError(
                            "direct-action session must not contain a motion reference spec"
                        )
                joint_names = tuple(
                    str(name) for name in getattr(request, "joint_names", ())
                )
                policy_camera_contract = _policy_camera_contract(request)
                if self._require_policy_camera and policy_camera_contract is None:
                    raise ValueError(
                        "strict policy-camera server requires a non-empty "
                        "policy_camera_spec on every session"
                    )
                if policy_camera_contract is not None:
                    ensure_humanoid_policy_camera_abi()
                if policy_camera_contract is not None and (
                    len(joint_names) != 29
                    or len(set(joint_names)) != 29
                    or any(not name for name in joint_names)
                ):
                    raise ValueError(
                        "policy camera session requires 29 unique non-empty joint_names"
                    )
                policy = self._policy_factory(session_uuid, request)
                save_camera_dir = _session_save_camera_dir(session_uuid, request)
                session = _Session(
                    policy=policy,
                    action_size=int(request.action_size),
                    observation_schema=str(request.observation_schema),
                    observation_terms=tuple(
                        (str(term.name), int(term.size))
                        for term in request.observation_terms
                    ),
                    joint_names=joint_names,
                    execution_mode=execution_mode,
                    policy_camera_contract=policy_camera_contract,
                    reference_joint_names=reference_joint_names,
                    reference_frame_count=reference_frame_count,
                    reference_sample_period_us=reference_sample_period_us,
                    reference_decode_context_schema=reference_decode_context_schema,
                    control_ticks_per_policy_step=control_ticks_per_policy_step,
                    behavior_policy_version=behavior_policy_version,
                    save_camera_dir=save_camera_dir,
                )
            except BaseException:
                try:
                    if policy is not None:
                        policy.close()
                finally:
                    if self._model_lease_registry is not None:
                        self._model_lease_registry.release_session_model_lease(
                            session_uuid
                        )
                raise
            self._sessions[session_uuid] = session
        logger.info(
            "Started AlpaGym humanoid policy session=%s action_size=%d",
            session_uuid,
            int(request.action_size),
        )
        return SessionRequestStatus()

    def act(
        self,
        request: HumanoidPolicyRequest,
        context: grpc.ServicerContext,
    ) -> HumanoidPolicyResponse:
        del context
        session_uuid = str(request.session_uuid)
        with self._sessions_lock:
            session = self._sessions[session_uuid]
        states = tuple(request.observation.env_states)
        decision_id = int(request.observation.decision_id)
        camera_routing = _route_camera_frames(
            states=states,
            camera_images=request.observation.camera_images,
            observation_timestamp_us=int(request.observation.timestamp_us),
            observation_decision_id=decision_id,
            joint_names=session.joint_names,
            policy_camera_contract=session.policy_camera_contract,
        )
        _save_camera_images(
            (
                frame
                for state in states
                for frame in camera_routing.all_frames_by_env[int(state.env_id)]
            ),
            session.save_camera_dir,
        )
        bootstrap_only = bool(request.bootstrap_only)
        request_kind = int(getattr(request, "request_kind", 0))
        is_finalize = (
            session.execution_mode == HUMANOID_EXECUTION_MODE_MOTION_REFERENCE
            and request_kind == HUMANOID_POLICY_REQUEST_KIND_FINALIZE_WITH_FEEDBACK
        )
        step_index = (
            session.step_index
            if bootstrap_only or is_finalize
            else session.consume_step_index()
        )
        _validate_policy_request_kind(
            session=session,
            request_kind=request_kind,
            bootstrap_only=bootstrap_only,
            step_index=step_index,
        )
        feedback_traces = _feedback_traces_from_request(
            request=request,
            session=session,
            step_index=step_index,
        )
        bootstrap_env_ids = frozenset(
            int(env_id) for env_id in request.bootstrap_env_ids
        )
        if len(bootstrap_env_ids) != len(tuple(request.bootstrap_env_ids)):
            raise ValueError("humanoid bootstrap_env_ids contains duplicates")
        session.attach_feedback_traces(
            feedback_traces,
            outer_truncated_env_ids=(bootstrap_env_ids if is_finalize else frozenset()),
        )
        if is_finalize:
            session.mark_outer_truncated(bootstrap_env_ids)
        policy_inputs = tuple(
            _policy_input_from_state(
                session_uuid=session_uuid,
                step_index=step_index,
                state=state,
                observation_schema=session.observation_schema,
                observation_terms=session.observation_terms,
                decision_id=decision_id,
                feedback_trace=feedback_traces.get(int(state.env_id)),
                bootstrap_requested=int(state.env_id) in bootstrap_env_ids,
                camera_frames=camera_routing.policy_frames_by_env[int(state.env_id)],
                camera_frame_identities=tuple(
                    _camera_frame_identity(frame)
                    for frame in camera_routing.all_frames_by_env[int(state.env_id)]
                ),
            )
            for state in states
        )
        policy_outputs = session.policy.step(
            policy_inputs,
            sample_actions=not (bootstrap_only or is_finalize),
        )
        if len(policy_outputs) != len(policy_inputs):
            raise ValueError(
                "humanoid policy returned "
                f"{len(policy_outputs)} outputs for {len(policy_inputs)} env states"
            )

        outputs_by_env = {int(output.env_id): output for output in policy_outputs}
        input_env_ids = [int(policy_input.env_id) for policy_input in policy_inputs]
        if len(outputs_by_env) != len(policy_outputs) or set(outputs_by_env) != set(
            input_env_ids
        ):
            raise ValueError(
                "humanoid policy outputs must match request env_ids exactly: "
                f"inputs={sorted(input_env_ids)}, outputs={sorted(outputs_by_env)}"
            )
        ordered_outputs = tuple(outputs_by_env[env_id] for env_id in input_env_ids)
        if bootstrap_only or is_finalize:
            if not bootstrap_env_ids.issubset(set(input_env_ids)):
                raise ValueError(
                    "humanoid bootstrap env_ids must be a subset of final observation lanes"
                )
            session.record_final_values(ordered_outputs, bootstrap_env_ids)
            return HumanoidPolicyResponse(
                value_estimates=[
                    HumanoidEnvValue(
                        env_id=int(output.env_id),
                        value=float(
                            torch.as_tensor(output.value, dtype=torch.float32)
                            .reshape(())
                            .item()
                        ),
                    )
                    for output in ordered_outputs
                    if int(output.env_id) in bootstrap_env_ids
                ],
                behavior_policy_version=str(session.behavior_policy_version),
            )

        actions: list[HumanoidAction] = []
        plan_updates: list[HumanoidPlanUpdate] = []
        recorded: list[PolicyOutput] = []
        for policy_input, output in zip(policy_inputs, ordered_outputs, strict=True):
            action_values: torch.Tensor | None = None
            motion_reference: HumanoidMotionReference | None = None
            reference_sha256: str | None = None
            if session.execution_mode == HUMANOID_EXECUTION_MODE_MOTION_REFERENCE:
                if output.action is not None or output.motion_reference is None:
                    raise ValueError(
                        "motion-reference session requires exactly one motion_reference output"
                    )
                motion_reference = output.motion_reference
                plan_update = _plan_update_from_output(
                    output=output,
                    policy_input=policy_input,
                    session=session,
                )
                plan_updates.append(plan_update)
                reference_sha256 = str(plan_update.reference_sha256)
            else:
                if output.action is None or output.motion_reference is not None:
                    raise ValueError(
                        "direct-action session requires exactly one tensor action output"
                    )
                action_values = torch.as_tensor(
                    output.action, dtype=torch.float32
                ).reshape(-1)
                if action_values.numel() != session.action_size:
                    raise ValueError(
                        f"humanoid action for env_id={output.env_id} has "
                        f"{action_values.numel()} values; expected {session.action_size}"
                    )
                if not torch.isfinite(action_values).all():
                    raise ValueError(
                        f"humanoid action for env_id={output.env_id} is non-finite"
                    )
                actions.append(
                    HumanoidAction(
                        env_id=int(output.env_id),
                        values=[float(value) for value in action_values.cpu().tolist()],
                    )
                )
            recorded.append(
                _recorded_policy_output(
                    step_index=step_index,
                    policy_input=policy_input,
                    output=output,
                    action_values=action_values,
                    motion_reference=motion_reference,
                    reference_sha256=reference_sha256,
                    prior_current_frame_identities=(
                        session.prior_current_camera_identities(policy_input)
                    ),
                )
            )
        session.record_validated_current_camera_identities(policy_inputs)
        session.record_outputs(tuple(recorded))
        return HumanoidPolicyResponse(
            actions=actions,
            plan_updates=plan_updates,
            behavior_policy_version=str(session.behavior_policy_version),
        )

    def close_session(
        self,
        request: HumanoidSessionCloseRequest,
        context: grpc.ServicerContext,
    ) -> Empty:
        del context
        session_uuid = str(request.session_uuid)
        with self._sessions_lock:
            session = self._sessions[session_uuid]
            record = session.get_record()
            self._sessions.pop(session_uuid)
            self._session_records[session_uuid] = record
        try:
            session.policy.close()
        finally:
            if self._model_lease_registry is not None:
                self._model_lease_registry.release_session_model_lease(session_uuid)
        logger.info(
            "Closed AlpaGym humanoid policy session=%s recorded_outputs=%d",
            session_uuid,
            len(self._session_records[session_uuid].outputs),
        )
        return Empty()

    def abort_session(
        self,
        request: HumanoidSessionAbortRequest,
        context: grpc.ServicerContext,
    ) -> Empty:
        """Idempotently discard a failed session without publishing replay."""
        del context
        session_uuid = str(request.session_uuid)
        if not session_uuid:
            raise ValueError("humanoid abort requires a session_uuid")
        self.discard_session(session_uuid)
        logger.info("Aborted AlpaGym humanoid policy session=%s", session_uuid)
        return Empty()

    def pop_session_record(self, session_uuid: str) -> HumanoidSessionRecord:
        """Remove and return the frozen record for ``session_uuid``."""
        with self._sessions_lock:
            return self._session_records.pop(session_uuid)

    def discard_session(self, session_uuid: str) -> None:
        """Drop a failed reservation/session/record before scheduling its retry."""
        missing = object()
        with self._sessions_lock:
            reserved_version = self._reserved_versions.pop(session_uuid, missing)
            session = self._sessions.pop(session_uuid, None)
            self._session_records.pop(session_uuid, None)
        try:
            if session is not None:
                session.policy.close()
        finally:
            if self._model_lease_registry is not None and (
                reserved_version is not missing or session is not None
            ):
                self._model_lease_registry.release_session_model_lease(session_uuid)

    def get_version(self, request: Empty, context: grpc.ServicerContext) -> VersionId:
        del request, context
        return VersionId(version_id="alpagym-humanoid-policy", git_hash="unknown")

    def shut_down(self, request: Empty, context: grpc.ServicerContext) -> Empty:
        del request, context
        raise NotImplementedError(
            "HumanoidPolicyService does not support remote shutdown."
        )


class HumanoidPolicyServer:
    """Lifecycle wrapper for the rollout worker's humanoid policy gRPC server."""

    def __init__(
        self,
        name: str,
        max_concurrent_rollouts: int,
        policy_factory: Callable[[str, HumanoidPolicySessionRequest], HumanoidPolicy],
        publish_host: str = "localhost",
        model_lease_registry: SessionModelLeaseRegistry | None = None,
        require_policy_camera: bool = False,
    ) -> None:
        self.name = name
        self.max_concurrent_rollouts = max_concurrent_rollouts
        self.host = publish_host
        bind_host = "localhost" if publish_host == "localhost" else "[::]"
        self._servicer = HumanoidPolicyGrpcServicer(
            policy_factory=policy_factory,
            model_lease_registry=model_lease_registry,
            require_policy_camera=require_policy_camera,
        )
        self._grpc_server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=2 * max_concurrent_rollouts + 2)
        )
        add_HumanoidPolicyServiceServicer_to_server(self._servicer, self._grpc_server)
        self.port = self._grpc_server.add_insecure_port(f"{bind_host}:0")
        if self.port == 0:
            raise RuntimeError(f"Failed to bind HumanoidPolicyService on {bind_host}:0")

    @property
    def topology_endpoint(self) -> TopologyEndpoint:
        return TopologyEndpoint(
            id=self.name,
            host=self.host,
            port=int(self.port),
            capacity=self.max_concurrent_rollouts,
        )

    def start(self) -> None:
        self._grpc_server.start()

    def stop(self, grace_seconds: float = 30.0) -> None:
        shutdown_event = self._grpc_server.stop(grace_seconds)
        shutdown_event.wait()

    @property
    def servicer(self) -> HumanoidPolicyGrpcServicer:
        return self._servicer


def _session_save_camera_dir(
    session_uuid: str,
    request: HumanoidPolicySessionRequest,
) -> Path | None:
    options = dict(getattr(request, "policy_options", {}) or {})
    root = options.get("save_camera_dir") or os.environ.get(
        "ALPAGYM_HUMANOID_SAVE_CAMERA_DIR"
    )
    if not root:
        return None
    path = Path(str(root)) / _safe_filename_component(session_uuid)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_filename_component(value: object) -> str:
    text = str(value)
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)
    return safe.strip("_") or "item"


def _image_suffix(image_bytes: bytes) -> str:
    if image_bytes.startswith(b"\xff\xd8"):
        return ".jpg"
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    return ".bin"


def _encoded_image_size(
    image_bytes: bytes,
    image_format: str,
    *,
    expected_size: tuple[int, int] | None = None,
) -> tuple[int, int]:
    """Fully decode and validate one policy-visible RGB image."""
    expected_format = {"jpeg": "JPEG", "png": "PNG"}[image_format]
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            if image.format != expected_format:
                raise ValueError(
                    "policy camera encoded format does not match its session spec"
                )
            if image.mode != "RGB":
                raise ValueError("policy camera image must be HWC uint8 RGB")
            width, height = image.size
            size = (int(width), int(height))
            if expected_size is not None and size != expected_size:
                raise ValueError(
                    "policy camera encoded raster does not match its session spec"
                )
            image.verify()
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.load()
            if image.format != expected_format or image.mode != "RGB":
                raise ValueError("policy camera decoded image contract changed")
            if tuple(int(value) for value in image.size) != size:
                raise ValueError("policy camera decoded raster size changed")
    except (OSError, SyntaxError, UnidentifiedImageError) as exc:
        raise ValueError("policy camera image cannot be fully decoded") from exc
    return size


def _policy_camera_contract(
    request: HumanoidPolicySessionRequest,
) -> HumanoidPolicyCameraContract | None:
    """Parse the optional strict policy-camera session contract."""
    raw = getattr(request, "policy_camera_spec", None)
    if raw is None:
        return None
    schema = str(raw.schema)
    logical_id = str(raw.logical_id)
    width = int(raw.width)
    height = int(raw.height)
    image_format = str(raw.image_format)
    max_frame_age_us = int(raw.max_frame_age_us)
    contract_sha256 = str(raw.contract_sha256)
    if not any(
        (
            schema,
            logical_id,
            width,
            height,
            image_format,
            max_frame_age_us,
            contract_sha256,
        )
    ):
        return None
    if schema != "humanoid_policy_camera_rgb_qpos.v1":
        raise ValueError("unsupported humanoid policy camera schema")
    if not logical_id.strip() or width <= 0 or height <= 0:
        raise ValueError(
            "policy camera requires a logical_id and positive width/height"
        )
    if image_format not in {"jpeg", "png"}:
        raise ValueError("policy camera image_format must be 'jpeg' or 'png'")
    if max_frame_age_us < 0:
        raise ValueError("policy camera max_frame_age_us must be non-negative")
    contract_sha256 = _require_lowercase_sha256(
        "policy camera contract_sha256", contract_sha256
    )
    return HumanoidPolicyCameraContract(
        schema=schema,
        logical_id=logical_id,
        width=width,
        height=height,
        image_format=image_format,
        max_frame_age_us=max_frame_age_us,
        contract_sha256=contract_sha256,
    )


def _route_camera_frames(
    *,
    states: Iterable[HumanoidEnvState],
    camera_images: Iterable[humanoid_pb2.HumanoidCameraImage],
    observation_timestamp_us: int,
    observation_decision_id: int,
    joint_names: tuple[str, ...],
    policy_camera_contract: HumanoidPolicyCameraContract | None,
) -> _HumanoidCameraRouting:
    """Validate all packets and select the strict policy camera per env lane."""
    states = tuple(states)
    state_by_env = {int(state.env_id): state for state in states}
    env_ids = tuple(int(state.env_id) for state in states)
    if len(set(env_ids)) != len(env_ids):
        raise ValueError("humanoid observation contains duplicate env_ids")
    grouped: dict[int, list[HumanoidCameraFrame]] = {env_id: [] for env_id in env_ids}
    identities: set[tuple[int, str, int, int]] = set()
    for image in camera_images:
        env_id = int(image.env_id)
        if env_id not in grouped:
            raise ValueError(f"humanoid camera frame targets unknown env_id={env_id}")
        frame_start_us = int(image.frame_start_us)
        frame_end_us = int(image.frame_end_us)
        if frame_start_us < 0 or frame_end_us < frame_start_us:
            raise ValueError(
                "humanoid camera frame timestamps must define a non-negative interval"
            )
        if frame_end_us > int(state_by_env[env_id].timestamp_us):
            raise ValueError(
                "humanoid camera frame_end_us must not exceed its env state timestamp"
            )
        logical_id = str(image.logical_id)
        if not logical_id.strip():
            raise ValueError("humanoid camera frame logical_id must be non-empty")
        identity = (env_id, logical_id, frame_start_us, frame_end_us)
        if identity in identities:
            raise ValueError("humanoid observation contains a duplicate camera frame")
        identities.add(identity)
        image_bytes = bytes(image.image_bytes)
        if not image_bytes:
            raise ValueError("humanoid camera frame image_bytes must be non-empty")
        render_qpos = tuple(float(value) for value in getattr(image, "render_qpos", ()))
        if any(not math.isfinite(value) for value in render_qpos):
            raise ValueError("humanoid camera render_qpos must be finite")
        frame = HumanoidCameraFrame(
            env_id=env_id,
            frame_start_us=frame_start_us,
            frame_end_us=frame_end_us,
            logical_id=logical_id,
            image_bytes=image_bytes,
            render_timestamp_us=int(getattr(image, "render_timestamp_us", 0)),
            observation_decision_id=int(getattr(image, "observation_decision_id", 0)),
            render_qpos=render_qpos,
            render_state_sha256=str(getattr(image, "render_state_sha256", "")),
            camera_contract_sha256=str(getattr(image, "camera_contract_sha256", "")),
            image_sha256=str(getattr(image, "image_sha256", "")),
            render_receipt_sha256=str(getattr(image, "render_receipt_sha256", "")),
            scene_fingerprint=str(getattr(image, "scene_fingerprint", "")),
            model_signature_sha256=str(getattr(image, "model_signature_sha256", "")),
            camera_to_world_sha256=str(getattr(image, "camera_to_world_sha256", "")),
            renderer_binding_sha256=str(getattr(image, "renderer_binding_sha256", "")),
        )
        if (
            policy_camera_contract is not None
            and frame.logical_id == policy_camera_contract.logical_id
        ):
            encoded_size = _encoded_image_size(
                frame.image_bytes,
                policy_camera_contract.image_format,
                expected_size=(
                    policy_camera_contract.width,
                    policy_camera_contract.height,
                ),
            )
            if encoded_size != (
                policy_camera_contract.width,
                policy_camera_contract.height,
            ):
                raise ValueError(
                    "policy camera encoded raster does not match its session spec"
                )
            if not (
                frame.frame_start_us == frame.frame_end_us == frame.render_timestamp_us
            ):
                raise ValueError(
                    "policy camera requires zero-shutter frame_start_us == "
                    "frame_end_us == render_timestamp_us"
                )
            if frame.render_timestamp_us > observation_timestamp_us:
                raise ValueError("policy camera frame is newer than its observation")
            if (
                observation_timestamp_us - frame.render_timestamp_us
                > policy_camera_contract.max_frame_age_us
            ):
                raise ValueError("policy camera frame exceeds max_frame_age_us")
            if frame.observation_decision_id != observation_decision_id:
                raise ValueError("policy camera observation_decision_id does not match")
            state = state_by_env[env_id]
            if frame.render_timestamp_us != int(state.timestamp_us):
                raise ValueError(
                    "policy camera render timestamp does not match its captured state"
                )
            if len(frame.render_qpos) != 36 or len(joint_names) != 29:
                raise ValueError(
                    "policy camera requires 36-D render_qpos in session joint order"
                )
            state_qpos = tuple(float(value) for value in state.qpos)
            if frame.render_qpos != state_qpos:
                raise ValueError(
                    "policy camera render_qpos does not match its captured state"
                )
            if frame.camera_contract_sha256 != policy_camera_contract.contract_sha256:
                raise ValueError("policy camera contract identity changed")
            _require_lowercase_sha256(
                "policy camera render_state_sha256", frame.render_state_sha256
            )
            _require_lowercase_sha256("policy camera image_sha256", frame.image_sha256)
            _require_lowercase_sha256(
                "policy camera render_receipt_sha256",
                frame.render_receipt_sha256,
            )
            for name, digest in (
                ("scene_fingerprint", frame.scene_fingerprint),
                ("model_signature_sha256", frame.model_signature_sha256),
                ("camera_to_world_sha256", frame.camera_to_world_sha256),
                ("renderer_binding_sha256", frame.renderer_binding_sha256),
            ):
                _require_lowercase_sha256(f"policy camera {name}", digest)
            from alpasim_grpc.v0.humanoid_contracts import (
                HUMANOID_RENDER_STATE_SCHEMA,
                HumanoidRenderState,
                humanoid_image_sha256,
                humanoid_render_receipt_v2_sha256,
            )

            render_state = HumanoidRenderState(
                schema=HUMANOID_RENDER_STATE_SCHEMA,
                env_id=frame.env_id,
                timestamp_us=frame.render_timestamp_us,
                observation_decision_id=frame.observation_decision_id,
                camera_logical_id=frame.logical_id,
                joint_names=joint_names,
                qpos=frame.render_qpos,
                camera_contract_sha256=frame.camera_contract_sha256,
            )
            if frame.render_state_sha256 != render_state.canonical_sha256():
                raise ValueError("policy camera render-state receipt is invalid")
            if frame.image_sha256 != humanoid_image_sha256(frame.image_bytes):
                raise ValueError("policy camera encoded-image receipt is invalid")
            expected_render_receipt = humanoid_render_receipt_v2_sha256(
                render_state_sha256=frame.render_state_sha256,
                camera_contract_sha256=frame.camera_contract_sha256,
                image_sha256=frame.image_sha256,
                image_format=policy_camera_contract.image_format,
                width=policy_camera_contract.width,
                height=policy_camera_contract.height,
                scene_fingerprint=frame.scene_fingerprint,
                model_signature_sha256=frame.model_signature_sha256,
                camera_to_world_sha256=frame.camera_to_world_sha256,
                renderer_binding_sha256=frame.renderer_binding_sha256,
            )
            if frame.render_receipt_sha256 != expected_render_receipt:
                raise ValueError(
                    "policy camera state/pixel/renderer-evidence receipt is invalid"
                )
        grouped[env_id].append(frame)
    all_frames_by_env = {
        env_id: tuple(
            sorted(
                frames,
                key=lambda frame: (
                    frame.frame_end_us,
                    frame.logical_id,
                    frame.frame_start_us,
                ),
            )
        )
        for env_id, frames in grouped.items()
    }
    if policy_camera_contract is not None:
        policy_frames_by_env = {
            env_id: tuple(
                frame
                for frame in frames
                if frame.logical_id == policy_camera_contract.logical_id
            )
            for env_id, frames in all_frames_by_env.items()
        }
        missing_or_repeated = {
            env_id: len(frames)
            for env_id, frames in policy_frames_by_env.items()
            if len(frames) != 1
        }
        if missing_or_repeated:
            raise ValueError(
                "policy camera requires exactly one current frame per env lane; "
                f"counts={missing_or_repeated}"
            )
    else:
        policy_frames_by_env = all_frames_by_env
    return _HumanoidCameraRouting(
        policy_frames_by_env=policy_frames_by_env,
        all_frames_by_env=all_frames_by_env,
    )


def _camera_frames_by_env(
    *,
    states: Iterable[HumanoidEnvState],
    camera_images: Iterable[humanoid_pb2.HumanoidCameraImage],
    observation_timestamp_us: int,
    observation_decision_id: int,
    joint_names: tuple[str, ...],
    policy_camera_contract: HumanoidPolicyCameraContract | None,
) -> dict[int, tuple[HumanoidCameraFrame, ...]]:
    """Return only policy-visible frames; strict sessions filter auxiliaries."""
    return _route_camera_frames(
        states=states,
        camera_images=camera_images,
        observation_timestamp_us=observation_timestamp_us,
        observation_decision_id=observation_decision_id,
        joint_names=joint_names,
        policy_camera_contract=policy_camera_contract,
    ).policy_frames_by_env


def _camera_frame_identity(frame: HumanoidCameraFrame) -> HumanoidCameraFrameIdentity:
    """Build compact immutable audit metadata without retaining pixels/qpos."""
    return HumanoidCameraFrameIdentity(
        env_id=frame.env_id,
        frame_start_us=frame.frame_start_us,
        frame_end_us=frame.frame_end_us,
        logical_id=frame.logical_id,
        byte_length=len(frame.image_bytes),
        sha256=hashlib.sha256(frame.image_bytes).hexdigest(),
        render_timestamp_us=frame.render_timestamp_us,
        observation_decision_id=frame.observation_decision_id,
        render_state_sha256=frame.render_state_sha256,
        camera_contract_sha256=frame.camera_contract_sha256,
        image_sha256=frame.image_sha256,
        render_receipt_sha256=frame.render_receipt_sha256,
        scene_fingerprint=frame.scene_fingerprint,
        model_signature_sha256=frame.model_signature_sha256,
        camera_to_world_sha256=frame.camera_to_world_sha256,
        renderer_binding_sha256=frame.renderer_binding_sha256,
    )


def _save_camera_images(
    camera_images: Iterable[HumanoidCameraFrame],
    save_camera_dir: Path | None,
) -> None:
    if save_camera_dir is None:
        return
    for index, image in enumerate(camera_images):
        image_bytes = image.image_bytes
        frame_start_us = image.frame_start_us
        env_id = image.env_id
        logical_id = _safe_filename_component(image.logical_id)
        image_sha256 = hashlib.sha256(image_bytes).hexdigest()
        render_state_sha256 = image.render_state_sha256 or "none"
        filename = (
            f"frame_{frame_start_us:012d}_decision_"
            f"{image.observation_decision_id:012d}_env{env_id:03d}_{index:02d}_"
            f"{logical_id}_render_{render_state_sha256}_image_{image_sha256}"
            f"{_image_suffix(image_bytes)}"
        )
        save_camera_dir.joinpath(filename).write_bytes(image_bytes)


def _message_has_fields(value: object | None) -> bool:
    """Return whether a protobuf-like nested message carries any value."""
    if value is None:
        return False
    list_fields = getattr(value, "ListFields", None)
    if callable(list_fields):
        return bool(list_fields())
    for item in vars(value).values() if hasattr(value, "__dict__") else ():
        if isinstance(item, str) and item:
            return True
        if isinstance(item, (bytes, tuple, list, dict, set)) and item:
            return True
        if isinstance(item, (int, float)) and not isinstance(item, bool) and item != 0:
            return True
        if item is not None and not isinstance(
            item, (str, bytes, tuple, list, dict, set, int, float, bool)
        ):
            if _message_has_fields(item):
                return True
    return False


def _validate_motion_reference_session(
    request: HumanoidPolicySessionRequest,
) -> tuple[tuple[str, ...], int, int, int, str]:
    """Validate one supported 50 Hz motion-reference wire ABI."""
    spec = request.reference_spec
    joint_names = tuple(str(name) for name in spec.joint_names)
    schema = str(spec.schema)
    if schema != _MOTION_REFERENCE_SCHEMA_H50:
        raise ValueError(
            "motion-reference session requires the one-second H50 G1 schema"
        )
    if str(request.action_schema) != schema:
        raise ValueError("motion-reference action_schema must match reference schema")
    if not joint_names or joint_names != tuple(
        str(name) for name in request.joint_names
    ):
        raise ValueError(
            "motion-reference spec joint_names must exactly match the session joint ABI"
        )
    if joint_names != MOTION_REFERENCE_JOINT_NAMES:
        raise ValueError(
            "motion-reference session joint_names do not match the canonical "
            "29-joint GRAIL wire order"
        )
    frame_count = int(spec.frame_count)
    sample_period_us = int(spec.sample_period_us)
    control_ticks = int(spec.control_ticks_per_policy_step)
    if frame_count != 50 or sample_period_us != 20_000 or control_ticks != 25:
        raise ValueError(
            "motion-reference spec does not match its schema; got "
            f"H={frame_count}, period={sample_period_us}, K={control_ticks}; "
            "expected H=50, period=20000us, replan_ticks=25"
        )
    decode_context_schema = str(spec.decode_context_schema)
    if decode_context_schema not in {
        "",
        humanoid_contracts.HUMANOID_FULL_ROTATION_LOCAL_XY_DECODE_CONTEXT_SCHEMA,
    }:
        raise ValueError(
            "motion-reference session has an unsupported decode_context_schema"
        )
    return (
        joint_names,
        frame_count,
        sample_period_us,
        control_ticks,
        decode_context_schema,
    )


def _validate_policy_request_kind(
    *,
    session: _Session,
    request_kind: int,
    bootstrap_only: bool,
    step_index: int,
) -> None:
    """Require explicit policy lifecycle requests in motion-reference mode."""
    if session.execution_mode != HUMANOID_EXECUTION_MODE_MOTION_REFERENCE:
        return
    if request_kind == HUMANOID_POLICY_REQUEST_KIND_FINALIZE_WITH_FEEDBACK:
        expected = HUMANOID_POLICY_REQUEST_KIND_FINALIZE_WITH_FEEDBACK
    elif step_index == 0 and not bootstrap_only:
        expected = HUMANOID_POLICY_REQUEST_KIND_INITIAL_PLAN
    elif bootstrap_only:
        expected = HUMANOID_POLICY_REQUEST_KIND_FINALIZE_WITH_FEEDBACK
    else:
        expected = HUMANOID_POLICY_REQUEST_KIND_REPLAN_WITH_FEEDBACK
    if request_kind != expected:
        raise ValueError(
            f"motion-reference request_kind={request_kind} does not match expected {expected}"
        )


def _feedback_traces_from_request(
    *,
    request: HumanoidPolicyRequest,
    session: _Session,
    step_index: int,
) -> dict[int, HumanoidRealizedFeedbackTrace]:
    """Parse realized intervals between consecutive policy samples."""
    raw_traces = tuple(getattr(request.observation, "feedback_traces", ()))
    if session.execution_mode != HUMANOID_EXECUTION_MODE_MOTION_REFERENCE:
        if raw_traces:
            raise ValueError(
                "direct-action request unexpectedly contains feedback traces"
            )
        return {}
    if step_index == 0 and not request.bootstrap_only:
        if raw_traces:
            raise ValueError(
                "initial motion-reference request must not contain feedback"
            )
        return {}

    env_ids = tuple(int(state.env_id) for state in request.observation.env_states)
    current_states = {
        int(state.env_id): state for state in request.observation.env_states
    }
    if len(raw_traces) != len(env_ids):
        raise ValueError(
            "motion-reference feedback trace count must match observation env lanes"
        )
    traces: dict[int, HumanoidRealizedFeedbackTrace] = {}
    for raw_trace in raw_traces:
        env_id = int(raw_trace.env_id)
        if env_id in traces or env_id not in env_ids:
            raise ValueError(f"invalid or duplicate feedback env_id={env_id}")
        source_decision_id = int(raw_trace.source_decision_id)
        decision_id = int(request.observation.decision_id)
        if decision_id <= 0 or source_decision_id != decision_id - 1:
            raise ValueError(
                "feedback source_decision_id must equal current decision_id - 1"
            )
        ticks = tuple(raw_trace.ticks)
        if not ticks:
            raise ValueError("feedback ticks must contain a non-empty interval")
        parsed_ticks: list[HumanoidRealizedControlTick] = []
        for tick_index, tick in enumerate(ticks):
            expected_offset = tick_index + 1
            if int(tick.control_tick_offset) != expected_offset:
                raise ValueError(
                    "feedback control_tick_offset must be contiguous from one"
                )
            reference_action_index = int(tick.reference_action_index)
            if not 0 <= reference_action_index < session.reference_frame_count:
                raise ValueError(
                    "feedback reference_action_index is outside the reference horizon"
                )
            state_input = _policy_input_from_state(
                session_uuid=str(request.session_uuid),
                step_index=step_index,
                state=tick.state,
                observation_schema=session.observation_schema,
                observation_terms=session.observation_terms,
            )
            if state_input.env_id != env_id:
                raise ValueError("native feedback tick state env_id changed")
            active_sha = _require_lowercase_sha256(
                "feedback active_reference_sha256",
                tick.active_reference_sha256,
            )
            active_reference_id = int(tick.active_reference_id)
            if active_reference_id <= 0:
                raise ValueError("feedback active_reference_id must be positive")
            applied_sha = _require_lowercase_sha256(
                "feedback applied_reference_sha256",
                tick.applied_reference_sha256,
            )
            reward = float(tick.reward)
            root_z_offset = float(tick.root_z_alignment_offset_m)
            metrics = {str(name): float(value) for name, value in tick.metrics.items()}
            if (
                not math.isfinite(reward)
                or not math.isfinite(root_z_offset)
                or any(
                    not name or not math.isfinite(value)
                    for name, value in metrics.items()
                )
            ):
                raise ValueError(
                    "feedback tick reward/root offset/metrics must be finite"
                )
            terminated = bool(tick.terminated)
            truncated = bool(tick.truncated)
            if terminated and truncated:
                raise ValueError(
                    "feedback tick cannot be both terminated and truncated"
                )
            if tick_index < len(ticks) - 1 and (terminated or truncated):
                raise ValueError(
                    "only the final feedback tick may end a macro transition"
                )
            if parsed_ticks:
                previous = parsed_ticks[-1]
                if (
                    state_input.timestamp_us
                    != previous.timestamp_us + session.reference_sample_period_us
                ):
                    raise ValueError(
                        "feedback tick timestamps must be contiguous at 20 ms"
                    )
                same_reference = (
                    active_reference_id == previous.active_reference_id
                    and active_sha == previous.active_reference_sha256
                )
                if same_reference and reference_action_index != min(
                    previous.reference_action_index + 1,
                    session.reference_frame_count - 1,
                ):
                    raise ValueError(
                        "feedback reference_action_index must advance within each "
                        "active-reference segment"
                    )
                if same_reference and applied_sha != previous.applied_reference_sha256:
                    raise ValueError(
                        "feedback applied_reference_sha256 changed within an "
                        "active-reference segment"
                    )
                if same_reference and not math.isclose(
                    root_z_offset,
                    previous.root_z_alignment_offset_m,
                    rel_tol=0.0,
                    abs_tol=1.0e-6,
                ):
                    raise ValueError(
                        "feedback root_z_alignment_offset_m changed within an "
                        "active-reference segment"
                    )
            parsed_ticks.append(
                HumanoidRealizedControlTick(
                    control_tick_offset=expected_offset,
                    qpos=state_input.qpos,
                    qvel=state_input.qvel,
                    observation=state_input.observation,
                    scalars=state_input.scalars,
                    timestamp_us=state_input.timestamp_us,
                    active_reference_id=active_reference_id,
                    reference_action_index=reference_action_index,
                    active_reference_sha256=active_sha,
                    applied_reference_sha256=applied_sha,
                    root_z_alignment_offset_m=root_z_offset,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    metrics=metrics,
                    control_episode_step=int(tick.control_episode_step),
                )
            )
        if ticks[-1].state != current_states[env_id]:
            raise ValueError(
                "last feedback tick state must equal the current env state"
            )
        traces[env_id] = HumanoidRealizedFeedbackTrace(
            env_id=env_id,
            source_decision_id=source_decision_id,
            ticks=tuple(parsed_ticks),
        )
    if set(traces) != set(env_ids):
        raise ValueError("motion-reference feedback env lanes are incomplete")
    session.advance_control_episode_steps(traces)
    return traces


def _plan_update_from_output(
    *,
    output: HumanoidPolicyStepOutput,
    policy_input: HumanoidPolicyInput,
    session: _Session,
) -> HumanoidPlanUpdate:
    """Validate and serialize one typed motion-reference output."""
    reference = output.motion_reference
    assert reference is not None
    if reference.reference_id <= 0:
        raise ValueError("motion reference_id must be positive")
    if reference.source_decision_id != policy_input.decision_id:
        raise ValueError("motion reference source_decision_id does not match request")
    if len(reference.frames) != session.reference_frame_count:
        raise ValueError(
            f"motion reference must contain {session.reference_frame_count} frames"
        )
    if not math.isfinite(reference.root_z_alignment_offset_m):
        raise ValueError("motion root_z_alignment_offset_m must be finite")
    messages: list[HumanoidMotionFrame] = []
    first_timestamp: int | None = None
    for frame_index, frame in enumerate(reference.frames):
        timestamp_us = int(frame.timestamp_us)
        if first_timestamp is None:
            first_timestamp = timestamp_us
            if first_timestamp != policy_input.timestamp_us:
                raise ValueError(
                    "motion reference frame zero timestamp must equal policy input timestamp"
                )
        expected_timestamp = (
            first_timestamp + frame_index * session.reference_sample_period_us
        )
        if timestamp_us != expected_timestamp:
            raise ValueError(
                "motion reference timestamps must be a contiguous 20 ms grid"
            )
        joint_position = _finite_vector(
            "motion joint_position",
            frame.joint_position,
            len(session.reference_joint_names),
        )
        joint_velocity = _finite_vector(
            "motion joint_velocity",
            frame.joint_velocity,
            len(session.reference_joint_names),
        )
        root_position = _finite_vector("motion root_position", frame.root_position, 3)
        root_quaternion = _finite_vector(
            "motion root_quaternion_wxyz",
            frame.root_quaternion_wxyz,
            4,
        )
        if not torch.isclose(
            torch.linalg.vector_norm(root_quaternion),
            torch.tensor(1.0),
            rtol=0.0,
            atol=1.0e-5,
        ):
            raise ValueError("motion root_quaternion_wxyz must be normalized")
        if frame_index == 0:
            realized_root_position = policy_input.qpos[:3].clone()
            realized_root_position[2] += float(reference.root_z_alignment_offset_m)
            if not torch.allclose(
                root_position,
                realized_root_position,
                rtol=0.0,
                atol=1.0e-5,
            ):
                raise ValueError(
                    "motion reference frame zero root position is not anchored "
                    "to realized state"
                )
        messages.append(
            HumanoidMotionFrame(
                timestamp_us=timestamp_us,
                joint_position=joint_position.tolist(),
                joint_velocity=joint_velocity.tolist(),
                root_position=Vec3(
                    x=float(root_position[0]),
                    y=float(root_position[1]),
                    z=float(root_position[2]),
                ),
                root_quaternion_wxyz=Quat(
                    w=float(root_quaternion[0]),
                    x=float(root_quaternion[1]),
                    y=float(root_quaternion[2]),
                    z=float(root_quaternion[3]),
                ),
            )
        )

    decode_context_message: humanoid_pb2.HumanoidReferenceDecodeContext | None = None
    decode_context = reference.decode_context
    if not session.reference_decode_context_schema:
        if decode_context is not None:
            raise ValueError(
                "motion reference decode context was not negotiated by the session"
            )
    else:
        if decode_context is None:
            raise ValueError(
                "motion reference is missing its negotiated decode context"
            )
        if decode_context.schema != session.reference_decode_context_schema:
            raise ValueError(
                "motion reference decode-context schema does not match the session"
            )
        chunk_base_quaternion = _finite_vector(
            "motion decode-context chunk_base_quaternion_wxyz",
            decode_context.chunk_base_quaternion_wxyz,
            4,
        )
        if not torch.isclose(
            torch.linalg.vector_norm(chunk_base_quaternion),
            torch.tensor(1.0),
            rtol=0.0,
            atol=1.0e-5,
        ):
            raise ValueError(
                "motion decode-context chunk_base_quaternion_wxyz must be normalized"
            )
        if not torch.allclose(
            chunk_base_quaternion,
            policy_input.qpos[3:7],
            rtol=0.0,
            atol=1.0e-5,
        ):
            raise ValueError(
                "motion decode-context chunk-base quaternion does not match "
                "the policy input"
            )
        local_xy = torch.as_tensor(
            decode_context.local_xy_from_frame_zero,
            dtype=torch.float32,
        )
        if (
            tuple(local_xy.shape)
            != (
                session.reference_frame_count,
                2,
            )
            or not torch.isfinite(local_xy).all()
        ):
            raise ValueError(
                "motion decode-context local_xy_from_frame_zero must be a finite "
                f"[{session.reference_frame_count},2] matrix"
            )
        if not torch.equal(local_xy[0], torch.zeros(2, dtype=torch.float32)):
            raise ValueError(
                "motion decode-context local_xy_from_frame_zero row zero must be "
                "exact zero"
            )
        decode_context_message = humanoid_pb2.HumanoidReferenceDecodeContext(
            schema=decode_context.schema,
            chunk_base_quaternion_wxyz=Quat(
                w=float(chunk_base_quaternion[0]),
                x=float(chunk_base_quaternion[1]),
                y=float(chunk_base_quaternion[2]),
                z=float(chunk_base_quaternion[3]),
            ),
            local_xy_from_frame_zero=local_xy.reshape(-1).tolist(),
        )

    update: HumanoidPlanUpdate = HumanoidPlanUpdate(
        env_id=int(output.env_id),
        reference_id=int(reference.reference_id),
        source_decision_id=int(reference.source_decision_id),
        valid=True,
        failure_reason="",
        frames=messages,
        root_z_alignment_offset_m=float(reference.root_z_alignment_offset_m),
        decode_context=decode_context_message,
    )
    spec: HumanoidMotionReferenceSpec = HumanoidMotionReferenceSpec(
        schema=_MOTION_REFERENCE_SCHEMA_H50,
        joint_names=session.reference_joint_names,
        frame_count=session.reference_frame_count,
        sample_period_us=session.reference_sample_period_us,
        control_ticks_per_policy_step=session.control_ticks_per_policy_step,
        decode_context_schema=session.reference_decode_context_schema,
    )
    update.reference_sha256 = humanoid_contracts.humanoid_reference_sha256(
        spec,
        update,
    )
    return update


def _finite_vector(name: str, value: object, length: int) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
    if tuple(tensor.shape) != (length,) or not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must be a finite vector of length {length}")
    return tensor


def _require_lowercase_sha256(name: str, value: object) -> str:
    digest = str(value)
    if _LOWERCASE_SHA256.fullmatch(digest) is None:
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return digest


def humanoid_model_input_tensor_sha256(value: torch.Tensor) -> str:
    """Hash one dense model-input tensor including dtype, shape, and exact bytes."""

    if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
        raise TypeError("humanoid model input must be one dense tensor")
    tensor = value.detach().cpu().clone(memory_format=torch.contiguous_format)
    descriptor = json.dumps(
        {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(b"alpagym.humanoid.model_input_tensor.v1\0")
    digest.update(len(descriptor).to_bytes(8, byteorder="big", signed=False))
    digest.update(descriptor)
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def humanoid_visual_input_manifest_sha256(manifest: Mapping[str, object]) -> str:
    """Hash one canonical JSON visual-input manifest with domain separation."""

    canonical = json.dumps(
        dict(manifest),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(
        b"alpagym.humanoid.visual_input_manifest.v1\0" + canonical
    ).hexdigest()


def _ordered_joint_names_sha256(joint_names: tuple[str, ...]) -> str:
    """Hash a non-empty ordered joint tuple with a stable JSON encoding."""

    names = tuple(str(name) for name in joint_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("joint_names must be a non-empty unique sequence")
    encoded = json.dumps(
        list(names),
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _feedback_trace_payload(
    trace: HumanoidRealizedFeedbackTrace,
    *,
    joint_names: tuple[str, ...],
) -> dict[str, object]:
    """Return a transport-safe exact controller receipt for replay auditing."""
    names = tuple(str(name) for name in joint_names)
    return {
        "env_id": trace.env_id,
        "source_decision_id": trace.source_decision_id,
        "state_contract_schema": HUMANOID_FEEDBACK_STATE_CONTRACT_SCHEMA,
        "joint_names": list(names),
        "ordered_joint_names_sha256": _ordered_joint_names_sha256(names),
        "ticks": [
            {
                "control_tick_offset": tick.control_tick_offset,
                "timestamp_us": tick.timestamp_us,
                "qpos": tick.qpos.detach().cpu(),
                "qvel": tick.qvel.detach().cpu(),
                "observation": tick.observation.detach().cpu(),
                "scalars": dict(tick.scalars),
                "active_reference_id": tick.active_reference_id,
                "reference_action_index": tick.reference_action_index,
                "active_reference_sha256": tick.active_reference_sha256,
                "applied_reference_sha256": tick.applied_reference_sha256,
                "root_z_alignment_offset_m": tick.root_z_alignment_offset_m,
                "reward": tick.reward,
                "terminated": tick.terminated,
                "truncated": tick.truncated,
                "metrics": dict(tick.metrics),
                "control_episode_step": tick.control_episode_step,
            }
            for tick in trace.ticks
        ],
    }


def _policy_input_from_state(
    *,
    session_uuid: str,
    step_index: int,
    state: HumanoidEnvState,
    observation_schema: str,
    observation_terms: tuple[tuple[str, int], ...],
    camera_frames: tuple[HumanoidCameraFrame, ...] = (),
    camera_frame_identities: tuple[HumanoidCameraFrameIdentity, ...] = (),
    decision_id: int = 0,
    feedback_trace: HumanoidRealizedFeedbackTrace | None = None,
    bootstrap_requested: bool = False,
) -> HumanoidPolicyInput:
    if str(state.observation_schema) != observation_schema:
        raise ValueError(
            f"humanoid state for env_id={state.env_id} has observation_schema="
            f"{state.observation_schema!r}; expected {observation_schema!r}"
        )
    qpos = torch.as_tensor(list(state.qpos), dtype=torch.float32)
    qvel = torch.as_tensor(list(state.qvel), dtype=torch.float32)
    observation = torch.as_tensor(list(state.observation), dtype=torch.float32)
    expected_observation_size = sum(size for _, size in observation_terms)
    if observation.shape != (expected_observation_size,):
        raise ValueError(
            f"humanoid state for env_id={state.env_id} has flat observation shape "
            f"{tuple(observation.shape)}; expected ({expected_observation_size},)"
        )
    named_observations = tuple(state.named_observations)
    actual_terms = tuple(
        (str(named.name), len(named.values)) for named in named_observations
    )
    if actual_terms != observation_terms:
        raise ValueError(
            f"humanoid state for env_id={state.env_id} named observation terms do not "
            f"match the session ABI: expected={observation_terms}, actual={actual_terms}"
        )
    named_flat_parts: list[torch.Tensor] = []
    for named, (_, expected_size) in zip(
        named_observations,
        observation_terms,
        strict=True,
    ):
        shape = tuple(int(width) for width in named.shape)
        if math.prod(shape) != expected_size:
            raise ValueError(
                f"humanoid named observation {named.name!r} shape {shape} does not "
                f"contain {expected_size} elements"
            )
        named_flat_parts.append(
            torch.as_tensor(list(named.values), dtype=torch.float32)
        )
    named_flat = torch.cat(named_flat_parts)
    if not torch.equal(named_flat, observation):
        raise ValueError(
            f"humanoid state for env_id={state.env_id} flat and named observations differ"
        )
    if not all(torch.isfinite(tensor).all() for tensor in (qpos, qvel, observation)):
        raise ValueError(
            f"humanoid state for env_id={state.env_id} contains non-finite values"
        )
    scalars = {str(key): float(value) for key, value in state.scalars.items()}
    if not all(math.isfinite(value) for value in scalars.values()):
        raise ValueError(
            f"humanoid scalars for env_id={state.env_id} contain non-finite values"
        )
    return HumanoidPolicyInput(
        session_uuid=session_uuid,
        episode_id=int(state.reset_id),
        step_index=step_index,
        timestamp_us=int(state.timestamp_us),
        env_id=int(state.env_id),
        qpos=qpos,
        qvel=qvel,
        observation=observation,
        scalars=scalars,
        camera_frames=camera_frames,
        camera_frame_identities=camera_frame_identities,
        decision_id=decision_id,
        feedback_trace=feedback_trace,
        bootstrap_requested=bootstrap_requested,
    )


def _visual_source_frame_entry(
    identity: HumanoidCameraFrameIdentity,
    *,
    role: str,
) -> dict[str, object]:
    """Project one server-routed identity onto the policy manifest contract."""

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


def _validate_recorded_visual_input_manifest(
    *,
    model_extra: Mapping[str, object],
    payload: Mapping[str, object],
    policy_input: HumanoidPolicyInput,
    step_index: int,
    prior_current_frame_identities: tuple[HumanoidCameraFrameIdentity, ...] = (),
) -> None:
    """Bind one policy-authored visual manifest to its exact replay tensor.

    The render receipt is provenance evidence, not a cryptographic signature: this
    validator assumes a trusted renderer producer and local AlpaSim/AlpaGym transport.
    Recomputing it here rejects stale or mix-and-match frame evidence in replay.
    """

    manifest_value = model_extra.get("humanoid_visual_input_manifest")
    digest_value = payload.get("visual_input_manifest_sha256")
    if manifest_value is None and digest_value is None:
        if "pixel_values" in payload and policy_input.camera_frames:
            raise ValueError("humanoid visual replay is missing its input manifest")
        return
    if not isinstance(manifest_value, Mapping) or digest_value is None:
        raise ValueError(
            "humanoid visual input requires both manifest and replay digest"
        )
    manifest = dict(manifest_value)
    expected_manifest_keys = {
        "schema",
        "session_uuid",
        "episode_id",
        "step_index",
        "timestamp_us",
        "env_id",
        "decision_id",
        "camera_logical_id",
        "image_format",
        "preprocess_profile",
        "instruction_sha256",
        "source_frames",
        "pixel_values",
    }
    if set(manifest) != expected_manifest_keys:
        raise ValueError("humanoid visual input manifest has invalid fields")
    expected_identity = {
        "schema": HUMANOID_VISUAL_INPUT_MANIFEST_SCHEMA,
        "session_uuid": policy_input.session_uuid,
        "episode_id": int(policy_input.episode_id),
        "step_index": int(step_index),
        "timestamp_us": int(policy_input.timestamp_us),
        "env_id": int(policy_input.env_id),
        "decision_id": int(policy_input.decision_id),
    }
    for name, expected in expected_identity.items():
        if manifest[name] != expected:
            raise ValueError(f"humanoid visual input manifest {name} changed")
    for name in (
        "camera_logical_id",
        "preprocess_profile",
    ):
        if not isinstance(manifest[name], str) or not manifest[name]:
            raise ValueError(f"humanoid visual input manifest {name} must be non-empty")
    image_format = manifest["image_format"]
    if image_format not in {"jpeg", "png"}:
        raise ValueError(
            "humanoid visual input manifest image_format must be 'jpeg' or 'png'"
        )
    instruction_sha256 = _require_lowercase_sha256(
        "humanoid visual input instruction_sha256", manifest["instruction_sha256"]
    )
    if payload.get("instruction_sha256") != instruction_sha256:
        raise ValueError("humanoid visual input instruction digest differs from replay")

    source_frames = manifest["source_frames"]
    if not isinstance(source_frames, list) or not source_frames:
        raise ValueError("humanoid visual input source_frames must be non-empty")
    frame_keys = {
        "role",
        "env_id",
        "frame_start_us",
        "frame_end_us",
        "logical_id",
        "byte_length",
        "render_timestamp_us",
        "observation_decision_id",
        "render_state_sha256",
        "camera_contract_sha256",
        "image_sha256",
        "render_receipt_sha256",
        "scene_fingerprint",
        "model_signature_sha256",
        "camera_to_world_sha256",
        "renderer_binding_sha256",
    }
    typed_frames: list[dict[str, object]] = []
    for index, frame_value in enumerate(source_frames):
        if not isinstance(frame_value, Mapping):
            raise TypeError("humanoid visual input source frame must be a mapping")
        frame = dict(frame_value)
        if set(frame) != frame_keys:
            raise ValueError("humanoid visual input source frame has invalid fields")
        expected_role = "current" if index == len(source_frames) - 1 else "history"
        if frame["role"] != expected_role:
            raise ValueError("humanoid visual input source frame order is invalid")
        for name in (
            "env_id",
            "frame_start_us",
            "frame_end_us",
            "byte_length",
            "render_timestamp_us",
            "observation_decision_id",
        ):
            if type(frame[name]) is not int or int(frame[name]) < 0:
                raise ValueError(
                    f"humanoid visual input source frame {name} must be non-negative"
                )
        if frame["env_id"] != int(policy_input.env_id) or frame["byte_length"] == 0:
            raise ValueError("humanoid visual input source frame identity is invalid")
        if (
            frame["frame_start_us"] != frame["render_timestamp_us"]
            or frame["frame_end_us"] != frame["render_timestamp_us"]
        ):
            raise ValueError("humanoid visual input source frame is not zero-shutter")
        if not isinstance(frame["logical_id"], str) or not frame["logical_id"]:
            raise ValueError("humanoid visual input logical_id must be non-empty")
        for name in (
            "render_state_sha256",
            "camera_contract_sha256",
            "image_sha256",
            "render_receipt_sha256",
            "scene_fingerprint",
            "model_signature_sha256",
            "camera_to_world_sha256",
            "renderer_binding_sha256",
        ):
            _require_lowercase_sha256(
                f"humanoid visual input source frame {name}", frame[name]
            )
        typed_frames.append(frame)

    render_timestamps = [int(frame["render_timestamp_us"]) for frame in typed_frames]
    if any(
        current <= previous
        for previous, current in zip(render_timestamps, render_timestamps[1:])
    ):
        raise ValueError("humanoid visual input source frames are not chronological")
    image_grid = payload.get("image_grid_thw")
    selected_history_indices = payload.get("selected_history_indices")
    if (
        not isinstance(image_grid, torch.Tensor)
        or image_grid.ndim != 2
        or image_grid.shape[1] != 3
        or not isinstance(selected_history_indices, torch.Tensor)
        or selected_history_indices.ndim != 1
    ):
        raise ValueError(
            "humanoid visual input replay is missing processor image-order metadata"
        )
    if len(typed_frames) != int(image_grid.shape[0]) or len(typed_frames) != (
        int(selected_history_indices.numel()) + 1
    ):
        raise ValueError(
            "humanoid visual input source order differs from processor image order"
        )
    current_frame = typed_frames[-1]
    if len(policy_input.camera_frames) != 1:
        raise ValueError("humanoid visual input requires one current policy camera")
    policy_frame = policy_input.camera_frames[0]
    if (
        hashlib.sha256(policy_frame.image_bytes).hexdigest()
        != policy_frame.image_sha256
    ):
        raise ValueError("humanoid visual input current JPEG digest is invalid")
    expected_current = {
        "role": "current",
        "env_id": policy_frame.env_id,
        "frame_start_us": policy_frame.frame_start_us,
        "frame_end_us": policy_frame.frame_end_us,
        "logical_id": policy_frame.logical_id,
        "byte_length": len(policy_frame.image_bytes),
        "render_timestamp_us": policy_frame.render_timestamp_us,
        "observation_decision_id": policy_frame.observation_decision_id,
        "render_state_sha256": policy_frame.render_state_sha256,
        "camera_contract_sha256": policy_frame.camera_contract_sha256,
        "image_sha256": policy_frame.image_sha256,
        "render_receipt_sha256": policy_frame.render_receipt_sha256,
        "scene_fingerprint": policy_frame.scene_fingerprint,
        "model_signature_sha256": policy_frame.model_signature_sha256,
        "camera_to_world_sha256": policy_frame.camera_to_world_sha256,
        "renderer_binding_sha256": policy_frame.renderer_binding_sha256,
    }
    if current_frame != expected_current:
        raise ValueError(
            "humanoid visual input current frame differs from policy input"
        )
    if manifest["camera_logical_id"] != policy_frame.logical_id:
        raise ValueError("humanoid visual input camera logical_id changed")
    for frame in typed_frames:
        if (
            frame["logical_id"] != current_frame["logical_id"]
            or frame["camera_contract_sha256"]
            != current_frame["camera_contract_sha256"]
            or frame["scene_fingerprint"] != current_frame["scene_fingerprint"]
            or frame["model_signature_sha256"]
            != current_frame["model_signature_sha256"]
        ):
            raise ValueError(
                "humanoid visual input history changed camera, scene, or renderer model"
            )
    native_width, native_height = _encoded_image_size(
        policy_frame.image_bytes,
        image_format,
    )
    for index, frame in enumerate(typed_frames):
        expected_receipt = humanoid_contracts.humanoid_render_receipt_v2_sha256(
            render_state_sha256=frame["render_state_sha256"],
            camera_contract_sha256=frame["camera_contract_sha256"],
            image_sha256=frame["image_sha256"],
            image_format=image_format,
            width=native_width,
            height=native_height,
            scene_fingerprint=frame["scene_fingerprint"],
            model_signature_sha256=frame["model_signature_sha256"],
            camera_to_world_sha256=frame["camera_to_world_sha256"],
            renderer_binding_sha256=frame["renderer_binding_sha256"],
        )
        if frame["render_receipt_sha256"] != expected_receipt:
            raise ValueError(
                "humanoid visual input source frame render receipt is invalid "
                f"at index {index}"
            )

    history_frames = typed_frames[:-1]
    prior_entries: list[dict[str, object]] = []
    for identity in prior_current_frame_identities:
        if identity.env_id != int(policy_input.env_id):
            raise ValueError("humanoid visual input camera ledger crossed an env lane")
        if identity.sha256 != identity.image_sha256:
            raise ValueError(
                "humanoid visual input camera ledger encoded-image identity changed"
            )
        prior_entries.append(_visual_source_frame_entry(identity, role="history"))
    matched_prior_indices: set[int] = set()
    current_timestamp_us = int(current_frame["render_timestamp_us"])
    for history in history_frames:
        matches = [
            index for index, prior in enumerate(prior_entries) if prior == history
        ]
        if len(matches) != 1 or matches[0] in matched_prior_indices:
            raise ValueError(
                "humanoid visual input history was not a unique previously routed "
                "current frame"
            )
        if int(history["render_timestamp_us"]) >= current_timestamp_us:
            raise ValueError(
                "humanoid visual input history is not strictly earlier than current"
            )
        matched_prior_indices.add(matches[0])

    pixel_value = manifest["pixel_values"]
    if not isinstance(pixel_value, Mapping):
        raise TypeError("humanoid visual input pixel_values must be a mapping")
    pixel_descriptor = dict(pixel_value)
    if set(pixel_descriptor) != {"dtype", "shape", "sha256"}:
        raise ValueError("humanoid visual input pixel descriptor has invalid fields")
    replay_pixels = payload.get("pixel_values")
    if not isinstance(replay_pixels, torch.Tensor):
        raise TypeError("humanoid visual input replay is missing pixel_values tensor")
    expected_pixel_descriptor = {
        "dtype": str(replay_pixels.dtype),
        "shape": list(replay_pixels.shape),
        "sha256": humanoid_model_input_tensor_sha256(replay_pixels),
    }
    if pixel_descriptor != expected_pixel_descriptor:
        raise ValueError("humanoid visual input pixel tensor identity changed")
    manifest_sha256 = humanoid_visual_input_manifest_sha256(manifest)
    if (
        _require_lowercase_sha256("humanoid visual input manifest digest", digest_value)
        != manifest_sha256
    ):
        raise ValueError("humanoid visual input manifest digest is invalid")


def _recorded_policy_output(
    *,
    step_index: int,
    policy_input: HumanoidPolicyInput,
    output: HumanoidPolicyStepOutput,
    action_values: torch.Tensor | None,
    motion_reference: HumanoidMotionReference | None,
    reference_sha256: str | None = None,
    prior_current_frame_identities: tuple[HumanoidCameraFrameIdentity, ...] = (),
) -> PolicyOutput:
    model_extra = dict(output.model_extra or {})
    server_owned_metadata = {
        "humanoid_env_id": int(output.env_id),
        "humanoid_episode_id": int(policy_input.episode_id),
        "humanoid_step_index": int(step_index),
        "humanoid_timestamp_us": int(policy_input.timestamp_us),
        "humanoid_decision_id": int(policy_input.decision_id),
    }
    for name, expected in server_owned_metadata.items():
        if model_extra.setdefault(name, expected) != expected:
            raise ValueError(f"humanoid model metadata {name} changed")
    audited_identities = policy_input.camera_frame_identities or tuple(
        _camera_frame_identity(frame) for frame in policy_input.camera_frames
    )
    camera_frame_identities = [
        {
            "env_id": identity.env_id,
            "frame_start_us": identity.frame_start_us,
            "frame_end_us": identity.frame_end_us,
            "logical_id": identity.logical_id,
            "byte_length": identity.byte_length,
            "sha256": identity.sha256,
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
        for identity in audited_identities
    ]
    recorded_camera_frames = model_extra.setdefault(
        "humanoid_camera_frames", camera_frame_identities
    )
    if recorded_camera_frames != camera_frame_identities:
        raise ValueError(
            "humanoid model metadata camera identities do not match policy input"
        )
    if output.value is not None:
        model_extra.setdefault(
            "humanoid_value",
            float(
                torch.as_tensor(output.value, dtype=torch.float32).reshape(()).item()
            ),
        )

    for name, value in (("logprob", output.logprob), ("value", output.value)):
        if value is not None and not torch.isfinite(torch.as_tensor(value)).all():
            raise ValueError(
                f"humanoid actor-critic {name} for env_id={output.env_id} is non-finite"
            )

    replay_data = output.replay_data
    if replay_data is not None:
        payload = dict(replay_data.payload)
        _validate_recorded_visual_input_manifest(
            model_extra=model_extra,
            payload=payload,
            policy_input=policy_input,
            step_index=step_index,
            prior_current_frame_identities=prior_current_frame_identities,
        )
        humanoid_payload = dict(payload.get("humanoid", {}))
        for name, expected in (
            ("env_id", int(output.env_id)),
            ("step_index", int(step_index)),
        ):
            if humanoid_payload.setdefault(name, expected) != expected:
                raise ValueError(f"humanoid replay {name} changed")
        if action_values is not None:
            humanoid_payload.setdefault("action", action_values.detach().cpu())
        if motion_reference is not None:
            if reference_sha256 is None:
                raise ValueError("recorded motion reference is missing its wire digest")
            expected_identity = {
                "reference_id": int(motion_reference.reference_id),
                "source_decision_id": int(motion_reference.source_decision_id),
                "reference_sha256": _require_lowercase_sha256(
                    "motion reference wire digest",
                    reference_sha256,
                ),
                "root_z_alignment_offset_m": float(
                    motion_reference.root_z_alignment_offset_m
                ),
            }
            for name, expected in expected_identity.items():
                actual = payload.setdefault(name, expected)
                if actual != expected:
                    raise ValueError(
                        f"policy replay {name} does not match emitted motion reference"
                    )
        if output.value is not None:
            humanoid_payload.setdefault(
                "value", torch.as_tensor(output.value, dtype=torch.float32).reshape(())
            )
        payload["humanoid"] = humanoid_payload
        replay_data = replace(replay_data, payload=payload)

    if motion_reference is None:
        assert action_values is not None
        chosen_xyz = action_values.detach().cpu().reshape(1, -1)
        chosen_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        chosen_dt_us = torch.tensor([0], dtype=torch.int64)
    else:
        chosen_xyz = torch.stack(
            [frame.joint_position.detach().cpu() for frame in motion_reference.frames]
        )
        chosen_quat = torch.stack(
            [
                frame.root_quaternion_wxyz.detach().cpu()
                for frame in motion_reference.frames
            ]
        )
        chosen_dt_us = torch.tensor(
            [frame.timestamp_us for frame in motion_reference.frames],
            dtype=torch.int64,
        )
    return PolicyOutput(
        chosen_xyz=chosen_xyz,
        chosen_quat=chosen_quat,
        chosen_dt_us=chosen_dt_us,
        chosen_logprob=(
            None
            if output.logprob is None
            else torch.as_tensor(output.logprob, dtype=torch.float32).reshape(())
        ),
        replay_data=replay_data,
        model_extra=model_extra,
    )
