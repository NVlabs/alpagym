# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402

"""Humanoid policy gRPC server used by AlpaGym rollout workers."""

from __future__ import annotations

import logging
import math
import os
import threading
from concurrent import futures
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Mapping, Protocol, runtime_checkable

import grpc
import torch
from alpagym_runtime.alpasim.grpc_import import ensure_alpasim_grpc_source

ensure_alpasim_grpc_source()
from alpagym_host.endpoint_registry import TopologyEndpoint
from alpasim_grpc.v0.common_pb2 import Empty, SessionRequestStatus, VersionId
from alpasim_grpc.v0.humanoid_pb2 import (
    HumanoidAction,
    HumanoidEnvValue,
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
    episode_id: int
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
    behavior_policy_version: int = 0
    save_camera_dir: Path | None = None
    step_index: int = 0
    outputs: list[PolicyOutput] = field(default_factory=list)
    final_bootstrap_values: dict[int, float] = field(default_factory=dict)
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

    def record_final_values(
        self,
        outputs: tuple[HumanoidPolicyStepOutput, ...],
    ) -> None:
        """Store one bootstrap value for every truncated final-state lane."""
        with self.lock:
            if self.final_bootstrap_values:
                raise ValueError("humanoid session received multiple bootstrap-only requests")
            for output in outputs:
                if output.value is None:
                    raise ValueError("humanoid bootstrap output is missing a value")
                value = float(
                    torch.as_tensor(output.value, dtype=torch.float32).reshape(()).item()
                )
                if not math.isfinite(value):
                    raise ValueError(
                        f"humanoid final value for env_id={output.env_id} is non-finite"
                    )
                self.final_bootstrap_values[int(output.env_id)] = value

    def get_record(self) -> HumanoidSessionRecord:
        """Freeze this session's outputs for the rollout worker."""
        with self.lock:
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
            raise ValueError("humanoid session reservation requires UUID and non-negative version")
        with self._sessions_lock:
            if (
                session_uuid in self._reserved_versions
                or session_uuid in self._sessions
                or session_uuid in self._session_records
            ):
                raise ValueError(f"humanoid session {session_uuid!r} is already reserved")
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
            policy = self._policy_factory(session_uuid, request)
            save_camera_dir = _session_save_camera_dir(session_uuid, request)
            session = _Session(
                policy=policy,
                action_size=int(request.action_size),
                observation_schema=str(request.observation_schema),
                observation_terms=tuple(
                    (str(term.name), int(term.size)) for term in request.observation_terms
                ),
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
        step_index = session.step_index if bootstrap_only else session.consume_step_index()
        policy_inputs = tuple(
            _policy_input_from_state(
                session_uuid=session_uuid,
                step_index=step_index,
                state=state,
                observation_schema=session.observation_schema,
                observation_terms=session.observation_terms,
            )
            for state in request.observation.env_states
        )
        policy_outputs = session.policy.step(policy_inputs, sample_actions=not bootstrap_only)
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
        if bootstrap_only:
            bootstrap_env_ids = [int(env_id) for env_id in request.bootstrap_env_ids]
            if set(bootstrap_env_ids) != set(input_env_ids) or len(bootstrap_env_ids) != len(
                set(bootstrap_env_ids)
            ):
                raise ValueError("humanoid bootstrap env_ids must match final observation lanes")
            session.record_final_values(ordered_outputs)
            return HumanoidPolicyResponse(
                value_estimates=[
                    HumanoidEnvValue(
                        env_id=int(output.env_id),
                        value=float(
                            torch.as_tensor(output.value, dtype=torch.float32).reshape(()).item()
                        ),
                    )
                    for output in ordered_outputs
                ],
                behavior_policy_version=str(session.behavior_policy_version),
            )

        actions: list[HumanoidAction] = []
        recorded: list[PolicyOutput] = []
        for policy_input, output in zip(policy_inputs, ordered_outputs, strict=True):
            action_values = torch.as_tensor(output.action, dtype=torch.float32).reshape(-1)
            if action_values.numel() != session.action_size:
                raise ValueError(
                    f"humanoid action for env_id={output.env_id} has "
                    f"{action_values.numel()} values; expected {session.action_size}"
                )
            if not torch.isfinite(action_values).all():
                raise ValueError(f"humanoid action for env_id={output.env_id} is non-finite")
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
                )
            )
        session.record_outputs(tuple(recorded))
        return HumanoidPolicyResponse(
            actions=actions,
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
    observation_schema: str,
    observation_terms: tuple[tuple[str, int], ...],
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
    actual_terms = tuple((str(named.name), len(named.values)) for named in named_observations)
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
        named_flat_parts.append(torch.as_tensor(list(named.values), dtype=torch.float32))
    named_flat = torch.cat(named_flat_parts)
    if not torch.equal(named_flat, observation):
        raise ValueError(
            f"humanoid state for env_id={state.env_id} flat and named observations differ"
        )
    if not all(torch.isfinite(tensor).all() for tensor in (qpos, qvel, observation)):
        raise ValueError(f"humanoid state for env_id={state.env_id} contains non-finite values")
    scalars = {str(key): float(value) for key, value in state.scalars.items()}
    if not all(math.isfinite(value) for value in scalars.values()):
        raise ValueError(f"humanoid scalars for env_id={state.env_id} contain non-finite values")
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
    )


def _recorded_policy_output(
    *,
    step_index: int,
    policy_input: HumanoidPolicyInput,
    output: HumanoidPolicyStepOutput,
    action_values: torch.Tensor,
) -> PolicyOutput:
    model_extra = dict(output.model_extra or {})
    model_extra.setdefault("humanoid_env_id", int(output.env_id))
    model_extra.setdefault("humanoid_episode_id", int(policy_input.episode_id))
    model_extra.setdefault("humanoid_step_index", int(step_index))
    model_extra.setdefault("humanoid_timestamp_us", int(policy_input.timestamp_us))
    if output.value is not None:
        model_extra.setdefault(
            "humanoid_value",
            float(torch.as_tensor(output.value, dtype=torch.float32).reshape(()).item()),
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
