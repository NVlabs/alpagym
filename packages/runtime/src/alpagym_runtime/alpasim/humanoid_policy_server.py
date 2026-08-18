# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402

"""Humanoid policy gRPC server used by AlpaGym rollout workers."""

from __future__ import annotations

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
from alpagym_runtime.alpasim.grpc_import import ensure_alpasim_grpc_source

ensure_alpasim_grpc_source()
from alpagym_host.endpoint_registry import TopologyEndpoint
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
    HumanoidPlanUpdate,
    HumanoidPolicyRequest,
    HumanoidPolicyResponse,
    HumanoidPolicySessionRequest,
    HumanoidSessionCloseRequest,
)
from alpasim_grpc.v0.humanoid_pb2_grpc import (
    HumanoidPolicyServiceServicer,
    add_HumanoidPolicyServiceServicer_to_server,
)

from alpagym_runtime.replay import PolicyReplayData
from alpagym_runtime.types import PolicyOutput

logger = logging.getLogger(__name__)

_LOWERCASE_SHA256 = re.compile(r"^[0-9a-f]{64}$")
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
    """One 50 Hz frame in a planner-produced motion reference."""

    timestamp_us: int
    joint_position: torch.Tensor
    joint_velocity: torch.Tensor
    root_position: torch.Tensor
    root_quaternion_wxyz: torch.Tensor


@dataclass(frozen=True)
class HumanoidMotionReference:
    """Typed 50-frame reference returned instead of a direct joint action."""

    reference_id: int
    source_decision_id: int
    frames: tuple[HumanoidMotionReferenceFrame, ...]
    reference_sha256: str
    root_z_alignment_offset_m: float


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
    execution_mode: int
    reference_joint_names: tuple[str, ...] = ()
    reference_frame_count: int = 0
    reference_sample_period_us: int = 0
    control_ticks_per_policy_step: int = 1
    behavior_policy_version: int = 0
    save_camera_dir: Path | None = None
    step_index: int = 0
    outputs: list[PolicyOutput] = field(default_factory=list)
    final_bootstrap_values: dict[int, float] = field(default_factory=dict)
    last_control_episode_steps: dict[int, int] = field(default_factory=dict)
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

    def attach_feedback_traces(
        self,
        traces: Mapping[int, HumanoidRealizedFeedbackTrace],
    ) -> None:
        """Join next-request controller receipts to their source plan rows."""
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
                reference_sha256 = str(payload.get("reference_sha256", ""))
                root_z_offset = float(
                    payload.get("root_z_alignment_offset_m", math.nan)
                )
                for tick in trace.ticks:
                    if tick.active_reference_id != reference_id:
                        raise ValueError(
                            "feedback active_reference_id does not match source plan"
                        )
                    if tick.active_reference_sha256 != reference_sha256:
                        raise ValueError(
                            "feedback active_reference_sha256 does not match source plan"
                        )
                    if not math.isclose(
                        tick.root_z_alignment_offset_m,
                        root_z_offset,
                        rel_tol=0.0,
                        abs_tol=1.0e-6,
                    ):
                        raise ValueError(
                            "feedback root_z_alignment_offset_m does not match source plan"
                        )
                payload["feedback_trace"] = _feedback_trace_payload(trace)
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
    ) -> None:
        self._policy_factory = policy_factory
        self._sessions: dict[str, _Session] = {}
        self._sessions_lock = threading.Lock()
        self._session_records: dict[str, HumanoidSessionRecord] = {}
        self._reserved_versions: dict[str, int] = {}

    def reserve_session(self, session_uuid: str, behavior_policy_version: int) -> None:
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
            # Proto zero/missing preserves the legacy direct-action contract.
            execution_mode = int(getattr(request, "execution_mode", 0))
            reference_joint_names: tuple[str, ...] = ()
            reference_frame_count = 0
            reference_sample_period_us = 0
            control_ticks_per_policy_step = 1
            if execution_mode == HUMANOID_EXECUTION_MODE_MOTION_REFERENCE:
                (
                    reference_joint_names,
                    reference_frame_count,
                    reference_sample_period_us,
                    control_ticks_per_policy_step,
                ) = _validate_motion_reference_session(request)
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
                execution_mode=execution_mode,
                reference_joint_names=reference_joint_names,
                reference_frame_count=reference_frame_count,
                reference_sample_period_us=reference_sample_period_us,
                control_ticks_per_policy_step=control_ticks_per_policy_step,
                behavior_policy_version=behavior_policy_version,
                save_camera_dir=save_camera_dir,
            )
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
        _save_camera_images(request.observation.camera_images, session.save_camera_dir)
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
        decision_id = int(getattr(request.observation, "decision_id", 0))
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
        session.attach_feedback_traces(feedback_traces)
        bootstrap_env_ids = frozenset(
            int(env_id) for env_id in request.bootstrap_env_ids
        )
        if len(bootstrap_env_ids) != len(tuple(request.bootstrap_env_ids)):
            raise ValueError("humanoid bootstrap_env_ids contains duplicates")
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
            )
            for state in request.observation.env_states
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
            if session.execution_mode == HUMANOID_EXECUTION_MODE_MOTION_REFERENCE:
                if output.action is not None or output.motion_reference is None:
                    raise ValueError(
                        "motion-reference session requires exactly one motion_reference output"
                    )
                motion_reference = output.motion_reference
                plan_updates.append(
                    _plan_update_from_output(
                        output=output,
                        policy_input=policy_input,
                        session=session,
                    )
                )
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
                )
            )
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
            session = self._sessions.pop(session_uuid)
            self._session_records[session_uuid] = session.get_record()
        session.policy.close()
        logger.info(
            "Closed AlpaGym humanoid policy session=%s recorded_outputs=%d",
            session_uuid,
            len(self._session_records[session_uuid].outputs),
        )
        return Empty()

    def pop_session_record(self, session_uuid: str) -> HumanoidSessionRecord:
        """Remove and return the frozen record for ``session_uuid``."""
        with self._sessions_lock:
            return self._session_records.pop(session_uuid)

    def discard_session(self, session_uuid: str) -> None:
        """Drop a failed reservation/session/record before scheduling its retry."""
        with self._sessions_lock:
            self._reserved_versions.pop(session_uuid, None)
            session = self._sessions.pop(session_uuid, None)
            self._session_records.pop(session_uuid, None)
        if session is not None:
            session.policy.close()

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
    ) -> None:
        self.name = name
        self.max_concurrent_rollouts = max_concurrent_rollouts
        self.host = publish_host
        bind_host = "localhost" if publish_host == "localhost" else "[::]"
        self._servicer = HumanoidPolicyGrpcServicer(policy_factory=policy_factory)
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


def _save_camera_images(
    camera_images: Iterable[object],
    save_camera_dir: Path | None,
) -> None:
    if save_camera_dir is None:
        return
    for index, image in enumerate(camera_images):
        image_bytes = bytes(getattr(image, "image_bytes", b""))
        if not image_bytes:
            continue
        frame_start_us = int(getattr(image, "frame_start_us", 0))
        env_id = int(getattr(image, "env_id", 0))
        logical_id = _safe_filename_component(getattr(image, "logical_id", "camera"))
        filename = (
            f"frame_{frame_start_us:012d}_env{env_id:03d}_{index:02d}_"
            f"{logical_id}{_image_suffix(image_bytes)}"
        )
        save_camera_dir.joinpath(filename).write_bytes(image_bytes)


def _validate_motion_reference_session(
    request: HumanoidPolicySessionRequest,
) -> tuple[tuple[str, ...], int, int, int]:
    """Validate the fixed 50 Hz/H=50 motion-reference wire ABI."""
    spec = request.reference_spec
    joint_names = tuple(str(name) for name in spec.joint_names)
    if str(spec.schema) != "g1_motion_reference_29d_50hz_h50.v1":
        raise ValueError(
            "motion-reference session requires schema "
            "'g1_motion_reference_29d_50hz_h50.v1'"
        )
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
    if (frame_count, sample_period_us, control_ticks) != (50, 20_000, 5):
        raise ValueError(
            "motion-reference spec must be H=50, period=20000us, macro K=5; got "
            f"H={frame_count}, period={sample_period_us}, K={control_ticks}"
        )
    return joint_names, frame_count, sample_period_us, control_ticks


def _validate_policy_request_kind(
    *,
    session: _Session,
    request_kind: int,
    bootstrap_only: bool,
    step_index: int,
) -> None:
    """Require explicit planner lifecycle requests in motion-reference mode."""
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
    """Parse and validate exact K-prefix post-controller feedback traces."""
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
        if not 1 <= len(ticks) <= session.control_ticks_per_policy_step:
            raise ValueError(
                "feedback ticks must be a non-empty K-prefix no longer than "
                f"{session.control_ticks_per_policy_step}"
            )
        parsed_ticks: list[HumanoidRealizedControlTick] = []
        for action_index, tick in enumerate(ticks):
            expected_offset = action_index + 1
            if int(tick.control_tick_offset) != expected_offset:
                raise ValueError(
                    "feedback control_tick_offset must be contiguous from one"
                )
            if int(tick.reference_action_index) != action_index:
                raise ValueError(
                    "feedback reference_action_index must be contiguous from zero"
                )
            state_input = _policy_input_from_state(
                session_uuid=str(request.session_uuid),
                step_index=step_index,
                state=tick.state,
                observation_schema=session.observation_schema,
                observation_terms=session.observation_terms,
            )
            active_sha = _require_lowercase_sha256(
                "feedback active_reference_sha256",
                tick.active_reference_sha256,
            )
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
            if action_index < len(ticks) - 1 and (terminated or truncated):
                raise ValueError(
                    "only the final feedback tick may end a macro transition"
                )
            if action_index > 0:
                previous_timestamp = parsed_ticks[-1].timestamp_us
                if (
                    state_input.timestamp_us
                    != previous_timestamp + session.reference_sample_period_us
                ):
                    raise ValueError(
                        "feedback tick timestamps must be contiguous at 20 ms"
                    )
            parsed_ticks.append(
                HumanoidRealizedControlTick(
                    control_tick_offset=expected_offset,
                    qpos=state_input.qpos,
                    qvel=state_input.qvel,
                    observation=state_input.observation,
                    scalars=state_input.scalars,
                    timestamp_us=state_input.timestamp_us,
                    active_reference_id=int(tick.active_reference_id),
                    reference_action_index=action_index,
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
        if len(parsed_ticks) < session.control_ticks_per_policy_step and not (
            parsed_ticks[-1].terminated or parsed_ticks[-1].truncated
        ):
            raise ValueError(
                "a short feedback trace must end in termination or truncation"
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
    """Validate and serialize one typed planner output."""
    reference = output.motion_reference
    assert reference is not None
    if reference.reference_id <= 0:
        raise ValueError("motion reference_id must be positive")
    if reference.source_decision_id != policy_input.decision_id:
        raise ValueError("motion reference source_decision_id does not match request")
    digest = _require_lowercase_sha256(
        "motion reference_sha256",
        reference.reference_sha256,
    )
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
            realized_joint_position = policy_input.qpos[7:]
            realized_joint_velocity = policy_input.qvel[6:]
            realized_root_position = policy_input.qpos[:3].clone()
            realized_root_position[2] += float(reference.root_z_alignment_offset_m)
            comparisons = (
                (joint_position, realized_joint_position, "joint position"),
                (joint_velocity, realized_joint_velocity, "joint velocity"),
                (root_position, realized_root_position, "root position"),
                (root_quaternion, policy_input.qpos[3:7], "root quaternion"),
            )
            for actual, expected, label in comparisons:
                if actual.shape != expected.shape or not torch.allclose(
                    actual,
                    expected,
                    rtol=0.0,
                    atol=1.0e-5,
                ):
                    raise ValueError(
                        f"motion reference frame zero {label} is not anchored to realized state"
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
    return HumanoidPlanUpdate(
        env_id=int(output.env_id),
        reference_id=int(reference.reference_id),
        source_decision_id=int(reference.source_decision_id),
        valid=True,
        failure_reason="",
        frames=messages,
        reference_sha256=digest,
        root_z_alignment_offset_m=float(reference.root_z_alignment_offset_m),
    )


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


def _feedback_trace_payload(trace: HumanoidRealizedFeedbackTrace) -> dict[str, object]:
    """Return a transport-safe exact controller receipt for replay auditing."""
    return {
        "env_id": trace.env_id,
        "source_decision_id": trace.source_decision_id,
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
        decision_id=decision_id,
        feedback_trace=feedback_trace,
        bootstrap_requested=bootstrap_requested,
    )


def _recorded_policy_output(
    *,
    step_index: int,
    policy_input: HumanoidPolicyInput,
    output: HumanoidPolicyStepOutput,
    action_values: torch.Tensor | None,
    motion_reference: HumanoidMotionReference | None,
) -> PolicyOutput:
    model_extra = dict(output.model_extra or {})
    model_extra.setdefault("humanoid_env_id", int(output.env_id))
    model_extra.setdefault("humanoid_episode_id", int(policy_input.episode_id))
    model_extra.setdefault("humanoid_step_index", int(step_index))
    model_extra.setdefault("humanoid_timestamp_us", int(policy_input.timestamp_us))
    model_extra.setdefault("humanoid_decision_id", int(policy_input.decision_id))
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
        humanoid_payload = dict(payload.get("humanoid", {}))
        humanoid_payload.setdefault("env_id", int(output.env_id))
        humanoid_payload.setdefault("step_index", int(step_index))
        if action_values is not None:
            humanoid_payload.setdefault("action", action_values.detach().cpu())
        if motion_reference is not None:
            expected_identity = {
                "reference_id": int(motion_reference.reference_id),
                "source_decision_id": int(motion_reference.source_decision_id),
                "reference_sha256": str(motion_reference.reference_sha256),
                "root_z_alignment_offset_m": float(
                    motion_reference.root_z_alignment_offset_m
                ),
            }
            for name, expected in expected_identity.items():
                actual = payload.setdefault(name, expected)
                if actual != expected:
                    raise ValueError(
                        f"planner replay {name} does not match emitted motion reference"
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
