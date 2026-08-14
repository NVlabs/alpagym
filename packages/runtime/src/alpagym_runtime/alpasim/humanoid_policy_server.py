# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402

"""Humanoid policy gRPC server used by AlpaGym rollout workers."""

from __future__ import annotations

import logging
import os
import threading
from concurrent import futures
from pathlib import Path
from dataclasses import dataclass, field, replace
from typing import Callable, Mapping, Protocol, runtime_checkable

import grpc
import torch
from alpagym_runtime.alpasim.grpc_import import ensure_alpasim_grpc_source

ensure_alpasim_grpc_source()
from alpagym_host.endpoint_registry import TopologyEndpoint
from alpasim_grpc.v0.common_pb2 import Empty, SessionRequestStatus, VersionId
from alpasim_grpc.v0.humanoid_pb2 import (
    HumanoidAction,
    HumanoidEnvState,
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


@dataclass(frozen=True)
class HumanoidPolicyInput:
    """One humanoid env-lane observation delivered to a policy."""

    session_uuid: str
    step_index: int
    timestamp_us: int
    env_id: int
    qpos: torch.Tensor
    qvel: torch.Tensor
    observation: torch.Tensor
    scalars: Mapping[str, float]


@dataclass(frozen=True)
class HumanoidPolicyStepOutput:
    """One humanoid env-lane action plus optional trainer replay payload."""

    env_id: int
    action: torch.Tensor
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
    ) -> tuple[HumanoidPolicyStepOutput, ...]:
        """Return one action for each env-lane input."""

    def close(self) -> None:
        """Release per-session resources."""


@dataclass(frozen=True)
class HumanoidSessionRecord:
    """Frozen per-session humanoid policy outputs drained by the rollout worker."""

    outputs: tuple[PolicyOutput, ...]


@dataclass
class _Session:
    """Per-session mutable state held by ``HumanoidPolicyGrpcServicer``."""

    policy: HumanoidPolicy
    action_size: int
    save_camera_dir: Path | None = None
    step_index: int = 0
    outputs: list[PolicyOutput] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def consume_step_index(self) -> int:
        """Return the current step index and advance the session clock."""
        with self.lock:
            step_index = self.step_index
            self.step_index += 1
        return step_index

    def record_outputs(self, outputs: tuple[PolicyOutput, ...]) -> None:
        """Append policy outputs in env-lane order."""
        self.outputs.extend(outputs)

    def get_record(self) -> HumanoidSessionRecord:
        """Freeze this session's outputs for the rollout worker."""
        return HumanoidSessionRecord(outputs=tuple(self.outputs))


class ZeroHumanoidPolicy:
    """Deterministic no-op humanoid policy for wiring smoke tests."""

    def __init__(self, action_size: int) -> None:
        self._action_size = int(action_size)

    def step(
        self,
        policy_inputs: tuple[HumanoidPolicyInput, ...],
    ) -> tuple[HumanoidPolicyStepOutput, ...]:
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

    def start_session(
        self,
        request: HumanoidPolicySessionRequest,
        context: grpc.ServicerContext,
    ) -> SessionRequestStatus:
        del context
        session_uuid = str(request.session_uuid)
        policy = self._policy_factory(session_uuid, request)
        save_camera_dir = _session_save_camera_dir(session_uuid, request)
        session = _Session(
            policy=policy,
            action_size=int(request.action_size),
            save_camera_dir=save_camera_dir,
        )
        with self._sessions_lock:
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
        step_index = session.consume_step_index()
        policy_inputs = tuple(
            _policy_input_from_state(
                session_uuid=session_uuid,
                step_index=step_index,
                state=state,
            )
            for state in request.observation.env_states
        )
        policy_outputs = session.policy.step(policy_inputs)
        if len(policy_outputs) != len(policy_inputs):
            raise ValueError(
                "humanoid policy returned "
                f"{len(policy_outputs)} outputs for {len(policy_inputs)} env states"
            )

        actions: list[HumanoidAction] = []
        recorded: list[PolicyOutput] = []
        for output in sorted(policy_outputs, key=lambda item: int(item.env_id)):
            action_values = torch.as_tensor(output.action, dtype=torch.float32).reshape(-1)
            if action_values.numel() != session.action_size:
                raise ValueError(
                    f"humanoid action for env_id={output.env_id} has "
                    f"{action_values.numel()} values; expected {session.action_size}"
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
                    output=output,
                    action_values=action_values,
                )
            )
        session.record_outputs(tuple(recorded))
        return HumanoidPolicyResponse(actions=actions)

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
        return self._session_records.pop(session_uuid)

    def get_version(self, request: Empty, context: grpc.ServicerContext) -> VersionId:
        del request, context
        return VersionId(version_id="alpagym-humanoid-policy", git_hash="unknown")

    def shut_down(self, request: Empty, context: grpc.ServicerContext) -> Empty:
        del request, context
        raise NotImplementedError("HumanoidPolicyService does not support remote shutdown.")


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
    root = options.get("save_camera_dir") or os.environ.get("ALPAGYM_HUMANOID_SAVE_CAMERA_DIR")
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


def _save_camera_images(camera_images: object, save_camera_dir: Path | None) -> None:
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


def _policy_input_from_state(
    *,
    session_uuid: str,
    step_index: int,
    state: HumanoidEnvState,
) -> HumanoidPolicyInput:
    return HumanoidPolicyInput(
        session_uuid=session_uuid,
        step_index=step_index,
        timestamp_us=int(state.timestamp_us),
        env_id=int(state.env_id),
        qpos=torch.as_tensor(list(state.qpos), dtype=torch.float32),
        qvel=torch.as_tensor(list(state.qvel), dtype=torch.float32),
        observation=torch.as_tensor(list(state.observation), dtype=torch.float32),
        scalars={str(key): float(value) for key, value in state.scalars.items()},
    )


def _recorded_policy_output(
    *,
    step_index: int,
    output: HumanoidPolicyStepOutput,
    action_values: torch.Tensor,
) -> PolicyOutput:
    model_extra = dict(output.model_extra or {})
    model_extra.setdefault("humanoid_env_id", int(output.env_id))
    model_extra.setdefault("humanoid_step_index", int(step_index))
    if output.value is not None:
        model_extra.setdefault(
            "humanoid_value",
            float(torch.as_tensor(output.value, dtype=torch.float32).reshape(()).item()),
        )

    replay_data = output.replay_data
    if replay_data is not None:
        payload = dict(replay_data.payload)
        humanoid_payload = dict(payload.get("humanoid", {}))
        humanoid_payload.setdefault("env_id", int(output.env_id))
        humanoid_payload.setdefault("step_index", int(step_index))
        humanoid_payload.setdefault("action", action_values.detach().cpu())
        if output.value is not None:
            humanoid_payload.setdefault(
                "value", torch.as_tensor(output.value, dtype=torch.float32).reshape(())
            )
        payload["humanoid"] = humanoid_payload
        replay_data = replace(replay_data, payload=payload)

    return PolicyOutput(
        chosen_xyz=action_values.detach().cpu().reshape(1, -1),
        chosen_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        chosen_dt_us=torch.tensor([0], dtype=torch.int64),
        chosen_logprob=(
            None
            if output.logprob is None
            else torch.as_tensor(output.logprob, dtype=torch.float32).reshape(())
        ),
        replay_data=replay_data,
        model_extra=model_extra,
    )
