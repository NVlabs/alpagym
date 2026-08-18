# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HumanoidPolicyService adapter for trainable VideoMimic shadow planning."""

from __future__ import annotations

import importlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from alpagym_runtime.alpasim.humanoid_policy_server import (
    HUMANOID_EXECUTION_MODE_MOTION_REFERENCE,
    MOTION_REFERENCE_JOINT_NAMES,
    HumanoidMotionReference,
    HumanoidMotionReferenceFrame,
    HumanoidPolicyInput,
    HumanoidPolicyStepOutput,
)
from alpagym_runtime.inference.inference_engine import InferenceEngine
from alpagym_runtime.replay import ActionSelection, PolicyReplayData

from alpagym_g1_videomimic_planner._tensor_conversion import as_float32_tensor
from alpagym_g1_videomimic_planner.model import (
    ACTION_SCHEMA,
    OBSERVATION_SCHEMA,
    OBS_DIMS,
    OBS_KEYS,
    REPLAY_SCHEMA,
    REFERENCE_FRAME_COUNT,
    SHADOW_ACTION_STEPS,
    G1VideoMimicPlannerActorCriticModel,
)

_OBSERVATION_TERMS = (("navigation_position_xy", 2),)
_VM_ENVIRONMENT_PROFILE = "videomimic.environment_geom_groups.v1"


@dataclass
class _Lane:
    policy: Any
    root_z_alignment_offset_m: float
    generator: torch.Generator
    next_reference_sequence: int = 1


class G1VideoMimicPlannerHumanoidPolicy:
    """One session of independent per-lane VideoMimic shadow simulators."""

    def __init__(
        self,
        inference_engine: InferenceEngine,
        *,
        request: Any,
        humanoid_repo_path: Path,
        scene_store_path: Path,
        expected_scene_fingerprints: Mapping[str, str],
        device: torch.device,
        deterministic: bool,
        lookahead_m: float,
        accept_shadow_fall_reference: bool,
    ) -> None:
        self._inference_engine = inference_engine
        self._request = request
        self._repo_path = humanoid_repo_path
        self._scene_store_path = scene_store_path
        try:
            self._expected_scene_fingerprint = expected_scene_fingerprints[
                str(request.scene_id)
            ]
        except KeyError as exc:
            raise ValueError(
                f"planner has no frozen fingerprint for scene {request.scene_id!r}"
            ) from exc
        self._device = device
        self._deterministic = deterministic
        self._lookahead_m = lookahead_m
        self._accept_shadow_fall_reference = bool(accept_shadow_fall_reference)
        self._lanes: dict[int, _Lane] = {}
        self._sample_actions = True
        self._active_model: G1VideoMimicPlannerActorCriticModel | None = None
        self._active_generator: torch.Generator | None = None
        self._first_sample_value: float | None = None
        self._support = _import_humanoid_support(humanoid_repo_path)

    def step(
        self,
        policy_inputs: tuple[HumanoidPolicyInput, ...],
        *,
        sample_actions: bool = True,
    ) -> tuple[HumanoidPolicyStepOutput, ...]:
        self._sample_actions = bool(sample_actions)
        outputs: list[HumanoidPolicyStepOutput] = []
        for policy_input in policy_inputs:
            lane = self._lanes.get(policy_input.env_id)
            if lane is None:
                if policy_input.feedback_trace is not None:
                    raise ValueError(
                        "initial planner observation unexpectedly has feedback"
                    )
                lane = self._create_lane(policy_input)
                self._lanes[policy_input.env_id] = lane
            else:
                if policy_input.feedback_trace is None:
                    raise ValueError("replan/finalize observation is missing feedback")
                lane.policy.update_many(
                    tuple(
                        self._policy_observation_from_tick(tick)
                        for tick in policy_input.feedback_trace.ticks
                    )
                )

            observation = self._policy_observation(policy_input)
            if not sample_actions:
                value = None
                if policy_input.bootstrap_requested:
                    self._plan_with_current_model(lane, observation)
                    if self._first_sample_value is None:
                        raise ValueError("truncated planner boundary produced no value")
                    value = torch.tensor(self._first_sample_value, dtype=torch.float32)
                outputs.append(
                    HumanoidPolicyStepOutput(env_id=policy_input.env_id, value=value)
                )
                continue

            plan = self._plan_with_current_model(lane, observation)
            if not bool(plan.valid):
                raise RuntimeError(
                    "VideoMimic shadow plan is invalid: "
                    f"{getattr(plan, 'failure_reason', 'unknown')}"
                )
            replay = lane.policy.active_replay
            if replay is None or not replay.trainable:
                raise ValueError(
                    "VideoMimic plan has no trainable 49-step replay trace"
                )
            if len(replay.steps) != SHADOW_ACTION_STEPS:
                raise ValueError(
                    f"VideoMimic replay must contain {SHADOW_ACTION_STEPS} actions"
                )
            if replay.reference_sha256 != plan.reference.sha256:
                raise ValueError("VideoMimic replay/reference digest mismatch")
            reference_id = (
                int(policy_input.env_id) + 1
            ) << 32 | lane.next_reference_sequence
            lane.next_reference_sequence += 1
            reference = _wire_reference(
                plan.reference,
                reference_id=reference_id,
                source_decision_id=policy_input.decision_id,
                root_z_alignment_offset_m=lane.root_z_alignment_offset_m,
            )
            replay_data = _policy_replay_data(replay)
            assert replay_data.old_logprob is not None
            old_logprob = replay_data.old_logprob
            value = torch.tensor(float(replay.initial_value), dtype=torch.float32)
            diagnostics = dict(plan.diagnostics)
            outputs.append(
                HumanoidPolicyStepOutput(
                    env_id=policy_input.env_id,
                    motion_reference=reference,
                    logprob=old_logprob,
                    value=value,
                    replay_data=replay_data,
                    model_extra={
                        "humanoid_value": float(value),
                        "humanoid_shadow_fell": bool(
                            diagnostics.get("shadow_fell", False)
                        ),
                        "humanoid_shadow_fall_reference_accepted": bool(
                            diagnostics.get("shadow_fall_reference_accepted", False)
                        ),
                        "humanoid_generated_reference_frames": len(plan.reference),
                    },
                )
            )
        return tuple(outputs)

    def close(self) -> None:
        self._lanes.clear()

    def _plan_with_current_model(self, lane: _Lane, observation: Any) -> Any:
        model = self._inference_engine.get_model()
        if not isinstance(model, G1VideoMimicPlannerActorCriticModel):
            raise TypeError(
                "expected G1VideoMimicPlannerActorCriticModel, got "
                f"{type(model).__name__}"
            )
        self._active_model = model
        self._active_generator = lane.generator
        self._first_sample_value = None
        try:
            return lane.policy.plan(
                observation,
                horizon_steps=REFERENCE_FRAME_COUNT,
            )
        finally:
            self._active_model = None
            self._active_generator = None

    def _sample_shadow_action(self, observation: Mapping[str, np.ndarray]) -> Any:
        model = self._active_model
        if model is None:
            raise RuntimeError(
                "VideoMimic action sampler called outside an atomic plan"
            )
        obs = {
            key: as_float32_tensor(observation[key], device=self._device).reshape(
                1, OBS_DIMS[key]
            )
            for key in OBS_KEYS
        }
        model.eval()
        with torch.no_grad():
            action, logprob, value, mean = model.act_step(
                obs,
                deterministic=self._deterministic or not self._sample_actions,
                generator=self._require_active_generator(),
            )
        raw = action[0].detach().cpu().numpy().astype(np.float32, copy=True)
        executed = np.clip(raw, -8.0, 8.0).astype(np.float32, copy=False)
        scalar_value = float(value[0].item())
        if self._first_sample_value is None:
            self._first_sample_value = scalar_value
        return self._support.ShadowActionSample(
            raw_action=raw,
            executed_action=executed,
            log_prob=float(logprob[0].item()),
            value=scalar_value,
            mean=mean[0].detach().cpu().numpy().astype(np.float32, copy=True),
        )

    def _require_active_generator(self) -> torch.Generator:
        if self._active_generator is None:
            raise RuntimeError("planner action sampler has no active lane RNG")
        return self._active_generator

    def _create_lane(self, policy_input: HumanoidPolicyInput) -> _Lane:
        runtime = self._support.ResolvedSimContext.from_scene_id(
            str(self._request.scene_id),
            scene_store_root=self._scene_store_path,
            scenario_id=str(self._request.scenario_id),
            environment_profile_id=_VM_ENVIRONMENT_PROFILE,
            physics_timestep_s=self._support.PHYSICS_TIMESTEP_S,
        )
        if runtime.bundle.scene_fingerprint != self._expected_scene_fingerprint:
            raise ValueError(
                "planner scene fingerprint differs from the negotiated dynamics scene"
            )
        scenario = runtime.scenario
        if not isinstance(scenario, Mapping):
            raise ValueError("planner scenario did not resolve navigation assets")
        route = np.asarray(scenario["route"]["waypoints_xy_m"], dtype=np.float64)
        spawn = np.asarray(scenario["spawn"]["position_xyz_m"], dtype=np.float64)
        heading_index = min(2, len(route) - 1)
        heading = route[heading_index] - route[0]
        root_yaw = math.atan2(float(heading[1]), float(heading[0]))
        policy = self._support.VideoMimicMotionPolicy.create(
            root_z_offset_m=0.0,
            default_horizon_steps=REFERENCE_FRAME_COUNT,
            minimum_action_steps=5,
            # A scratch VideoMimic fall is a property of the sampled reference,
            # not an infrastructure failure.  Complete the 50-frame candidate
            # and let frozen GRAIL + realized MuJoCo state produce the actual
            # terminal reward.  The generic/deployment facade keeps rejection
            # as its default.
            accept_shadow_fall_reference=self._accept_shadow_fall_reference,
            model=runtime.model,
            data=runtime.data,
            action_sampler=self._sample_shadow_action,
            route_xy=route,
            grail_joint_names=tuple(str(name) for name in self._request.joint_names),
            terrain_group_index=int(
                runtime.profile_abi["reserved_groups"]["terrain_heightmap"]
            ),
            lookahead_m=self._lookahead_m,
            device=str(self._device),
        )
        initial_shadow = policy.initialize_standalone(
            spawn_xyz=spawn, root_yaw=root_yaw
        )
        observation = self._policy_observation(policy_input)
        root_z_offset = float(
            initial_shadow.root_position[2] - observation.state.root_position[2]
        )
        policy.set_root_z_offset_m(root_z_offset)
        policy.reset(observation)
        generator = torch.Generator(device=self._device)
        generator.manual_seed(int(self._request.random_seed) + int(policy_input.env_id))
        return _Lane(
            policy=policy,
            root_z_alignment_offset_m=root_z_offset,
            generator=generator,
        )

    def _policy_observation(self, item: HumanoidPolicyInput) -> Any:
        return self._support.PolicyObservation(
            state=self._state(
                time_s=item.timestamp_us / 1_000_000.0,
                qpos=item.qpos,
                qvel=item.qvel,
            ),
            context={"navigation_position_xy": self._navigation_xy(item)},
        )

    def _policy_observation_from_tick(self, tick: Any) -> Any:
        navigation = as_float32_tensor(tick.observation).reshape(-1)
        if tuple(navigation.shape) != (2,):
            raise ValueError("feedback navigation_position_xy must have shape (2,)")
        return self._support.PolicyObservation(
            state=self._state(
                time_s=tick.timestamp_us / 1_000_000.0,
                qpos=tick.qpos,
                qvel=tick.qvel,
            ),
            context={"navigation_position_xy": navigation.cpu().numpy()},
        )

    def _state(self, *, time_s: float, qpos: torch.Tensor, qvel: torch.Tensor) -> Any:
        qpos_np = as_float32_tensor(qpos).cpu().numpy()
        qvel_np = as_float32_tensor(qvel).cpu().numpy()
        joint_count = len(tuple(self._request.joint_names))
        if qpos_np.shape != (7 + joint_count,) or qvel_np.shape != (6 + joint_count,):
            raise ValueError(
                "planner qpos/qvel do not match the 29-joint free-root ABI"
            )
        return self._support.RobotKinematicState(
            time_s=float(time_s),
            joint_names=tuple(str(name) for name in self._request.joint_names),
            joint_position=qpos_np[7:],
            joint_velocity=qvel_np[6:],
            root_position=qpos_np[:3],
            root_quaternion_wxyz=qpos_np[3:7],
            root_linear_velocity_world=qvel_np[:3],
            root_angular_velocity_body=qvel_np[3:6],
        )

    @staticmethod
    def _navigation_xy(item: HumanoidPolicyInput) -> np.ndarray:
        navigation = as_float32_tensor(item.observation).reshape(-1)
        if tuple(navigation.shape) != (2,) or not torch.isfinite(navigation).all():
            raise ValueError("navigation_position_xy must be a finite shape-(2,) term")
        for index, scalar_name in enumerate(
            ("navigation_position_x_m", "navigation_position_y_m")
        ):
            if scalar_name in item.scalars and not math.isclose(
                float(item.scalars[scalar_name]),
                float(navigation[index]),
                rel_tol=0.0,
                abs_tol=1.0e-6,
            ):
                raise ValueError(
                    "navigation scalar audit disagrees with named observation"
                )
        return navigation.cpu().numpy().copy()


def build_humanoid_policy_factory(
    run_config: Any,
    inference_engine: InferenceEngine,
):
    """Build a session factory with explicit support-repo and SceneStore roots."""
    config = dict(run_config.policy.model.bundle_config)
    repo_path = _required_directory(config, "humanoid_repo_path")
    scene_store_path = _required_directory(config, "scene_store_path")
    expected_scene_fingerprints = _parse_scene_fingerprints(config)
    device = torch.device(run_config.policy.model.device)
    deterministic = bool(config.get("deterministic", False))
    shadow_fall_reference_mode = str(config.get("shadow_fall_reference_mode", "reject"))
    if shadow_fall_reference_mode not in {"reject", "accept_for_training"}:
        raise ValueError(
            "bundle_config.shadow_fall_reference_mode must be 'reject' or "
            "'accept_for_training'"
        )
    lookahead_m = float(config.get("lookahead_m", 0.35))
    if not math.isfinite(lookahead_m) or lookahead_m <= 0.0:
        raise ValueError("bundle_config.lookahead_m must be finite and positive")

    def _factory(session_uuid: str, request: Any) -> G1VideoMimicPlannerHumanoidPolicy:
        del session_uuid
        _validate_session_request(request)
        return G1VideoMimicPlannerHumanoidPolicy(
            inference_engine,
            request=request,
            humanoid_repo_path=repo_path,
            scene_store_path=scene_store_path,
            expected_scene_fingerprints=expected_scene_fingerprints,
            device=device,
            deterministic=deterministic,
            lookahead_m=lookahead_m,
            accept_shadow_fall_reference=(
                shadow_fall_reference_mode == "accept_for_training"
            ),
        )

    return _factory


def _validate_session_request(request: Any) -> None:
    if int(request.execution_mode) != HUMANOID_EXECUTION_MODE_MOTION_REFERENCE:
        raise ValueError("VideoMimic planner requires motion-reference execution mode")
    if str(request.observation_schema) != OBSERVATION_SCHEMA:
        raise ValueError(f"planner observation_schema must be {OBSERVATION_SCHEMA!r}")
    if str(request.action_schema) != ACTION_SCHEMA:
        raise ValueError(f"planner action_schema must be {ACTION_SCHEMA!r}")
    if int(request.action_size) != 0:
        raise ValueError("motion-reference planner action_size must be zero")
    joint_names = tuple(str(name) for name in request.joint_names)
    if joint_names != MOTION_REFERENCE_JOINT_NAMES:
        raise ValueError("planner joint_names do not match canonical 29-joint order")
    terms = tuple(
        (str(term.name), int(term.size)) for term in request.observation_terms
    )
    if terms != _OBSERVATION_TERMS:
        raise ValueError(f"planner observation terms must be {_OBSERVATION_TERMS!r}")
    if str(request.reference_spec.schema) != ACTION_SCHEMA:
        raise ValueError("planner reference_spec schema does not match action schema")
    reference_spec = request.reference_spec
    if tuple(str(name) for name in reference_spec.joint_names) != joint_names:
        raise ValueError("planner reference_spec joint_names do not match session")
    if (
        int(reference_spec.frame_count),
        int(reference_spec.sample_period_us),
        int(reference_spec.control_ticks_per_policy_step),
    ) != (50, 20_000, 5):
        raise ValueError("planner reference_spec must be H=50, period=20ms, K=5")
    for field in ("attempt_id", "scene_id", "scenario_id"):
        if not str(getattr(request, field)):
            raise ValueError(f"planner session requires non-empty {field}")


def _wire_reference(
    reference: Any,
    *,
    reference_id: int,
    source_decision_id: int,
    root_z_alignment_offset_m: float,
) -> HumanoidMotionReference:
    frames = tuple(
        HumanoidMotionReferenceFrame(
            timestamp_us=round(float(frame.time_s) * 1_000_000.0),
            joint_position=as_float32_tensor(frame.joint_position),
            joint_velocity=as_float32_tensor(frame.joint_velocity),
            root_position=as_float32_tensor(frame.root_position),
            root_quaternion_wxyz=as_float32_tensor(frame.root_quaternion_wxyz),
        )
        for frame in reference.frames
    )
    return HumanoidMotionReference(
        reference_id=reference_id,
        source_decision_id=source_decision_id,
        frames=frames,
        reference_sha256=str(reference.sha256),
        root_z_alignment_offset_m=root_z_alignment_offset_m,
    )


def _policy_replay_data(replay: Any) -> PolicyReplayData:
    observations = {
        key: torch.stack(
            [as_float32_tensor(step.observation[key]) for step in replay.steps]
        )
        for key in OBS_KEYS
    }
    raw_actions = torch.stack(
        [as_float32_tensor(step.raw_action) for step in replay.steps]
    )
    executed_actions = torch.stack(
        [as_float32_tensor(step.executed_action) for step in replay.steps]
    )
    token_logprobs = torch.tensor(
        [float(step.old_logprob) for step in replay.steps], dtype=torch.float32
    )
    return PolicyReplayData(
        replay_schema_version=1,
        payload_schema=REPLAY_SCHEMA,
        payload_schema_version=1,
        model_family="g1_videomimic_planner",
        action_selection=ActionSelection(set_ix=0, sample_ix=0),
        old_logprob=token_logprobs.sum(),
        payload={
            "shadow_observations": observations,
            "raw_actions": raw_actions,
            "executed_actions": executed_actions,
            "old_token_logprobs": token_logprobs,
            "reference_sha256": str(replay.reference_sha256),
        },
    )


def _required_directory(config: Mapping[str, Any], name: str) -> Path:
    raw = config.get(name)
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"bundle_config.{name} is required")
    path = Path(raw).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"bundle_config.{name} directory not found: {path}")
    return path


def _parse_scene_fingerprints(config: Mapping[str, Any]) -> dict[str, str]:
    raw = config.get("expected_scene_fingerprints_json")
    if not isinstance(raw, str) or not raw:
        raise ValueError("bundle_config.expected_scene_fingerprints_json is required")
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "bundle_config.expected_scene_fingerprints_json is invalid JSON"
        ) from exc
    if not isinstance(decoded, dict) or not decoded:
        raise ValueError(
            "bundle_config.expected_scene_fingerprints_json must be a non-empty object"
        )
    fingerprints: dict[str, str] = {}
    for scene_id, digest in decoded.items():
        if (
            not isinstance(scene_id, str)
            or not scene_id
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise ValueError(
                "bundle_config.expected_scene_fingerprints_json must map non-empty "
                "scene IDs to lowercase SHA256 digests"
            )
        fingerprints[scene_id] = digest
    canonical = json.dumps(fingerprints, sort_keys=True, separators=(",", ":"))
    if raw != canonical:
        raise ValueError(
            "bundle_config.expected_scene_fingerprints_json must be canonical JSON"
        )
    return fingerprints


@dataclass(frozen=True)
class _HumanoidSupport:
    VideoMimicMotionPolicy: Any
    ShadowActionSample: Any
    PolicyObservation: Any
    RobotKinematicState: Any
    ResolvedSimContext: Any
    PHYSICS_TIMESTEP_S: float


def _import_humanoid_support(repo_path: Path) -> _HumanoidSupport:
    required = (
        repo_path / "policies" / "planners" / "videomimic.py",
        repo_path / "sim" / "videomimic_shadow.py",
        repo_path / "sim" / "scene_runtime.py",
        repo_path / "twin_scene",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"humanoid support repo is incomplete: {missing}")
    repo_text = str(repo_path)
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)
    planner_module = importlib.import_module("policies.planners.videomimic")
    shadow_module = importlib.import_module("sim.videomimic_shadow")
    policy_api = importlib.import_module("policies.api")
    scene_runtime = importlib.import_module("sim.scene_runtime")
    runtime_contract = importlib.import_module("sim.runtime_contract")
    for module in (
        planner_module,
        shadow_module,
        policy_api,
        scene_runtime,
        runtime_contract,
    ):
        module_file = module.__file__
        if module_file is None:
            raise ImportError(
                f"humanoid support module {module.__name__} has no source file"
            )
        module_path = Path(module_file).resolve()
        if repo_path not in module_path.parents:
            raise ImportError(
                f"humanoid support module {module.__name__} resolved outside {repo_path}"
            )
    return _HumanoidSupport(
        VideoMimicMotionPolicy=planner_module.VideoMimicMotionPolicy,
        ShadowActionSample=shadow_module.ShadowActionSample,
        PolicyObservation=policy_api.PolicyObservation,
        RobotKinematicState=policy_api.RobotKinematicState,
        ResolvedSimContext=scene_runtime.ResolvedSimContext,
        PHYSICS_TIMESTEP_S=float(runtime_contract.PHYSICS_TIMESTEP_S),
    )
