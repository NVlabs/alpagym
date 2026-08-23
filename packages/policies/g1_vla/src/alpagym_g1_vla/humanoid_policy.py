# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VLA callback using AlpaGym's existing motion-reference path."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, TypedDict, cast

import torch
from alpasim_grpc.v0.humanoid_contracts import (
    HUMANOID_FULL_ROTATION_LOCAL_XY_DECODE_CONTEXT_SCHEMA,
)
from alpagym_runtime.alpasim.humanoid_policy_server import (
    HUMANOID_EXECUTION_MODE_MOTION_REFERENCE,
    HUMANOID_VISUAL_INPUT_MANIFEST_SCHEMA,
    MOTION_REFERENCE_JOINT_NAMES,
    HumanoidCameraFrame,
    HumanoidMotionReference,
    HumanoidMotionReferenceFrame,
    HumanoidReferenceDecodeContext,
    HumanoidPolicyInput,
    HumanoidPolicyStepOutput,
    HumanoidRealizedFeedbackTrace,
    humanoid_model_input_tensor_sha256,
    humanoid_visual_input_manifest_sha256,
)
from alpagym_runtime.replay import ActionSelection, PolicyReplayData

from alpagym_g1_vla.bundle import (
    FLOW_SDE_TRAINING,
    MODEL_FAMILY,
    NATIVE_ODE_QUALIFICATION,
    QUALIFICATION_REPLAY_SCHEMA,
    REPLAY_SCHEMA,
    vla_sampling_mode,
)
from alpagym_g1_vla.history import (
    VlaImageHistory,
    VlaImageSourceIdentity,
    decode_legacy_d455_letterbox,
    decode_native_d435,
    stable_episode_int,
)
from alpagym_g1_vla.flow import (
    VLA_FLOW_IGNORE_LAST,
    VLA_FLOW_NOISE_LEVEL,
)
from alpagym_g1_vla.inference_model import unwrap_actor_critic
from alpagym_g1_vla.model import (
    VlaPolicySample,
    VlaPsiActorCritic,
    VlaQualificationSample,
)
from alpagym_g1_vla.provenance import (
    VlaBundleProfile,
    vla_bundle_profile_for_model_root,
)
from alpagym_g1_vla.reference_adapter import (
    H50_FRAME_COUNT,
    REFERENCE_PERIOD_US,
    SUPPORTED_REPLAN_CONTROLLER_TICKS,
    VlaMotionReferenceAdapter,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_H50_SCHEMA = "g1_motion_reference_29d_50hz_h50.v1"
_VLA_ACTION_ROWS = 30
_VLA_ACTION_WIDTH = 38
# Async training starts with the native RealTimeChunkController's conservative
# latency prior.  Synchronous qualification freezes the simulator during model
# inference, so it must start from measured zero delay instead.
_RTC_ASYNC_INITIAL_DELAY_ROWS = 6
_RTC_SYNC_INITIAL_DELAY_ROWS = 0
_RTC_DELAY_WINDOW_SIZE = 6
_RTC_PREFIX_MODE_LATENCY_PREDICTOR = "latency_predictor/v1"
_RTC_PREFIX_MODE_FIXED_CONTINUITY = "fixed_continuity/v1"
_VLA_CONTROL_HZ = 30
_REFERENCE_CONTROL_HZ = 50


@dataclass(frozen=True)
class _PreviousRtcChunk:
    """One validated sampled chunk and the source shot it is encoded against."""

    reference_id: int
    source_timestamp_us: int
    normalized_actions: torch.Tensor
    denormalized_joint_positions: torch.Tensor
    base_quaternion_wxyz: torch.Tensor
    base_xy: torch.Tensor


@dataclass(frozen=True)
class _RtcLaneState:
    """RTC state replaced as one object only after an output fully validates."""

    delay_rows: tuple[int, ...]
    previous_chunk: _PreviousRtcChunk | None


@dataclass(frozen=True)
class _RtcInputs:
    """Pure RTC inputs plus the state to commit with a successful sample."""

    normalized_actions: torch.Tensor
    mask: torch.Tensor
    source_cursor_h50: int
    source_start_row_h30: int
    velocity_seed_joint_position: torch.Tensor | None
    next_delay_rows: tuple[int, ...]
    observed_delay_rows: int
    configured_continuity_prefix_rows: int
    prefix_mode: str


@dataclass
class _Lane:
    """Per-env deterministic RNG and image-history ownership."""

    generator: torch.Generator
    history: VlaImageHistory
    reset_episode_id: int
    initial_rtc_delay_rows: int
    qualification_continuity_prefix_rows: int
    next_reference_sequence: int = 1
    rtc_state: _RtcLaneState | None = None

    def __post_init__(self) -> None:
        """Seed RTC with the execution mode's explicit latency prior."""
        if self.rtc_state is None:
            self.rtc_state = _RtcLaneState(
                delay_rows=(self.initial_rtc_delay_rows,), previous_chunk=None
            )


class _SessionModelLookup(Protocol):
    """Inference-engine surface used by humanoid callbacks."""

    def get_model_for_session(self, session_uuid: str) -> torch.nn.Module:
        """Return the immutable model owner for one session."""
        ...


class _VisualInputs(TypedDict):
    """Exact kwargs shared by action sampling and value-only inference."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor
    sequence_lengths: torch.Tensor
    image_counts: torch.Tensor
    image_offsets: torch.Tensor
    image_patch_counts: torch.Tensor
    patch_counts: torch.Tensor
    patch_offsets: torch.Tensor
    physical_states: torch.Tensor
    effective_image_grid_thw: torch.Tensor | None
    visual_pool_factors: torch.Tensor | None


class G1VlaHumanoidPolicy:
    """One VLA session sharing an immutable behavior-model lease."""

    def __init__(
        self,
        inference_engine: _SessionModelLookup,
        *,
        session_uuid: str,
        request: Any,
        image_format: str,
        expected_schedule_sha256: str,
        humanoid_repo_path: Path,
        humanoid_vla_reference_adapter_sha256: str,
        humanoid_motion_reference_sha256: str,
        language_instruction: str,
        run_config_sha256: str,
        bundle_profile: VlaBundleProfile,
        sampling_mode: str,
        replan_controller_ticks: int,
    ) -> None:
        """Bind the exact request identity and create lane state lazily."""
        if not session_uuid:
            raise ValueError("VLA policy requires a non-empty session UUID")
        self._inference_engine = inference_engine
        self._session_uuid = session_uuid
        self._request = request
        self._image_format = image_format
        self._expected_schedule_sha256 = expected_schedule_sha256
        self._language_instruction = language_instruction
        self._instruction_sha256 = hashlib.sha256(
            language_instruction.encode("utf-8")
        ).hexdigest()
        if _SHA256.fullmatch(run_config_sha256) is None:
            raise ValueError("VLA run_config_sha256 must be a lowercase SHA-256")
        self._run_config_sha256 = run_config_sha256
        if not isinstance(bundle_profile, VlaBundleProfile):
            raise TypeError("VLA policy requires an attested bundle profile")
        self._bundle_profile = bundle_profile
        if sampling_mode not in {FLOW_SDE_TRAINING, NATIVE_ODE_QUALIFICATION}:
            raise ValueError("VLA policy requires an explicit supported sampling mode")
        self._sampling_mode = sampling_mode
        self._initial_rtc_delay_rows = (
            _RTC_SYNC_INITIAL_DELAY_ROWS
            if sampling_mode == NATIVE_ODE_QUALIFICATION
            else _RTC_ASYNC_INITIAL_DELAY_ROWS
        )
        self._qualification_continuity_prefix_rows = (
            bundle_profile.native_qualification_continuity_prefix_rows
            if sampling_mode == NATIVE_ODE_QUALIFICATION
            else 0
        )
        if replan_controller_ticks not in SUPPORTED_REPLAN_CONTROLLER_TICKS:
            raise ValueError("VLA policy received an unsupported replan cadence")
        self._replan_controller_ticks = int(replan_controller_ticks)
        self._reference_adapter = VlaMotionReferenceAdapter(
            humanoid_repo_path,
            reference_adapter_sha256=humanoid_vla_reference_adapter_sha256,
            motion_reference_sha256=humanoid_motion_reference_sha256,
        )
        if self._reference_adapter.joint_names != MOTION_REFERENCE_JOINT_NAMES:
            raise ValueError(
                "VLA reference adapter joint order differs from AlpaGym's G1 contract"
            )
        scenario_id = str(request.scenario_id)
        if not scenario_id.strip():
            raise ValueError("VLA policy requires a non-empty scenario_id")
        self._bats_episode_index = stable_episode_int(scenario_id)
        self._lanes: dict[int, _Lane] = {}

    def step(
        self,
        policy_inputs: tuple[HumanoidPolicyInput, ...],
        *,
        sample_actions: bool = True,
    ) -> tuple[HumanoidPolicyStepOutput, ...]:
        """Sample chunks or evaluate FINALIZE bootstrap values per env lane."""
        outputs: list[HumanoidPolicyStepOutput] = []
        model_owner = self._inference_engine.get_model_for_session(self._session_uuid)
        actor_critic = unwrap_actor_critic(model_owner)
        if (
            actor_critic.rtc_max_delay_exclusive
            != self._reference_adapter.rtc_max_delay_exclusive
        ):
            raise ValueError(
                "attested VLA checkpoint and native reference adapter disagree "
                "on RTC max_delay"
            )
        if self._initial_rtc_delay_rows >= actor_critic.rtc_max_delay_exclusive:
            raise ValueError(
                "VLA initial RTC delay is outside the attested checkpoint contract"
            )
        if (
            self._qualification_continuity_prefix_rows
            >= actor_critic.rtc_max_delay_exclusive
        ):
            raise ValueError(
                "VLA qualification continuity prefix is outside the attested "
                "checkpoint contract"
            )
        if actor_critic.schedule.sha256 != self._expected_schedule_sha256:
            raise ValueError("VLA leased model flow schedule identity changed")
        if (
            actor_critic.qualification_schedule.sha256
            != self._bundle_profile.native_qualification_schedule_sha256
        ):
            raise ValueError(
                "VLA leased model native qualification schedule identity changed"
            )
        if (
            actor_critic.qualification_clip_normalized_actions
            is not self._bundle_profile.native_qualification_clip_normalized_actions
        ):
            raise ValueError(
                "VLA leased model native qualification wire contract changed"
            )
        if actor_critic.noise_level != VLA_FLOW_NOISE_LEVEL:
            raise ValueError(
                f"VLA profile requires Flow-SDE noise_level={VLA_FLOW_NOISE_LEVEL}"
            )
        if actor_critic.ignore_last is not VLA_FLOW_IGNORE_LAST:
            raise ValueError("VLA profile requires Flow-SDE ignore_last=true")

        for policy_input in policy_inputs:
            lane = self._lane(policy_input, actor_critic)
            self._validate_lifecycle(policy_input, sample_actions=sample_actions)
            visual_inputs, replay_visual, visual_input_manifest = (
                self._observation_inputs(
                    lane=lane,
                    policy_input=policy_input,
                    actor_critic=actor_critic,
                )
            )
            rtc = self._rtc_inputs(
                lane=lane,
                policy_input=policy_input,
                actor_critic=actor_critic,
            )
            if not sample_actions:
                value: torch.Tensor | None = None
                if policy_input.bootstrap_requested:
                    with self._autocast(actor_critic):
                        values = actor_critic.forward_values(
                            **visual_inputs,
                            rtc_prefix_normalized_actions=rtc.normalized_actions,
                            rtc_prefix_mask=rtc.mask,
                        )
                    value = values[0].detach().to(device="cpu", dtype=torch.float32)
                outputs.append(
                    HumanoidPolicyStepOutput(env_id=policy_input.env_id, value=value)
                )
                continue

            with self._autocast(actor_critic):
                if self._sampling_mode == FLOW_SDE_TRAINING:
                    sample = actor_critic.sample_actions(
                        **visual_inputs,
                        rtc_prefix_normalized_actions=rtc.normalized_actions,
                        rtc_prefix_mask=rtc.mask,
                        generator=lane.generator,
                    )
                elif self._sampling_mode == NATIVE_ODE_QUALIFICATION:
                    sample = actor_critic.sample_actions_native_ode(
                        **visual_inputs,
                        rtc_prefix_normalized_actions=rtc.normalized_actions,
                        rtc_prefix_mask=rtc.mask,
                        generator=lane.generator,
                    )
                else:
                    raise AssertionError("VLA sampling mode changed after construction")
            output = self._sample_output(
                lane=lane,
                policy_input=policy_input,
                sample=sample,
                replay_visual=replay_visual,
                visual_input_manifest=visual_input_manifest,
                rtc=rtc,
                actor_critic=actor_critic,
            )
            outputs.append(output)
        return tuple(outputs)

    def close(self) -> None:
        """Release decoded history images owned by this session."""
        for lane in self._lanes.values():
            lane.history.close()
        self._lanes.clear()

    def _lane(
        self,
        policy_input: HumanoidPolicyInput,
        actor_critic: VlaPsiActorCritic,
    ) -> _Lane:
        """Return or create one lane with request-seed plus env-id RNG."""
        lane = self._lanes.get(policy_input.env_id)
        if lane is not None:
            if lane.reset_episode_id != int(policy_input.episode_id):
                raise ValueError("VLA lane crossed reset IDs inside one session")
            return lane
        device = actor_critic.normalizer.state_low.device
        generator = torch.Generator(device=device)
        generator.manual_seed(int(self._request.random_seed) + int(policy_input.env_id))
        lane = _Lane(
            generator=generator,
            history=VlaImageHistory(episode_index=self._bats_episode_index),
            reset_episode_id=int(policy_input.episode_id),
            initial_rtc_delay_rows=self._initial_rtc_delay_rows,
            qualification_continuity_prefix_rows=(
                self._qualification_continuity_prefix_rows
            ),
        )
        self._lanes[policy_input.env_id] = lane
        return lane

    @staticmethod
    def _validate_lifecycle(
        policy_input: HumanoidPolicyInput,
        *,
        sample_actions: bool,
    ) -> None:
        """Require motion-reference initial/replan/final inputs to stay distinct."""
        if policy_input.step_index == 0 and policy_input.feedback_trace is not None:
            raise ValueError("initial VLA input unexpectedly carries feedback")
        if policy_input.step_index > 0 and policy_input.feedback_trace is None:
            raise ValueError("VLA replan/FINALIZE input is missing feedback")
        if sample_actions and policy_input.bootstrap_requested:
            raise ValueError("VLA bootstrap value cannot be requested while sampling")

    @staticmethod
    def _rtc_inputs(
        *,
        lane: _Lane,
        policy_input: HumanoidPolicyInput,
        actor_critic: VlaPsiActorCritic,
    ) -> _RtcInputs:
        """Build native hard-prefix RTC conditioning for one inference.

        A regular launch is H50 cursor 25 / H30 row 15, but slow inference can
        install a plan late enough that its next launch is already beyond that
        row.  Source timestamps therefore select the actual remaining H30
        suffix; the delay predictor only decides how much of that real suffix
        becomes deterministic density-free prefix.
        """
        device = actor_critic.normalizer.state_low.device
        actions = torch.zeros(
            (1, _VLA_ACTION_ROWS, _VLA_ACTION_WIDTH),
            dtype=torch.float32,
            device=device,
        )
        mask = torch.zeros((1, _VLA_ACTION_ROWS), dtype=torch.bool, device=device)
        rtc_state = lane.rtc_state
        if rtc_state is None or not rtc_state.delay_rows:
            raise AssertionError("VLA RTC lane state is missing")
        if len(rtc_state.delay_rows) > _RTC_DELAY_WINDOW_SIZE or any(
            value < 0 for value in rtc_state.delay_rows
        ):
            raise AssertionError("VLA RTC delay history is invalid")
        previous = rtc_state.previous_chunk
        if previous is None:
            return _RtcInputs(
                normalized_actions=actions,
                mask=mask,
                source_cursor_h50=0,
                source_start_row_h30=0,
                velocity_seed_joint_position=None,
                next_delay_rows=rtc_state.delay_rows,
                observed_delay_rows=rtc_state.delay_rows[-1],
                configured_continuity_prefix_rows=(
                    lane.qualification_continuity_prefix_rows
                ),
                prefix_mode=(
                    _RTC_PREFIX_MODE_FIXED_CONTINUITY
                    if lane.qualification_continuity_prefix_rows > 0
                    else _RTC_PREFIX_MODE_LATENCY_PREDICTOR
                ),
            )

        previous_chunk = previous.normalized_actions
        if (
            tuple(previous_chunk.shape)
            != (
                _VLA_ACTION_ROWS,
                _VLA_ACTION_WIDTH,
            )
            or not torch.isfinite(previous_chunk).all()
        ):
            raise AssertionError("VLA RTC previous chunk is invalid")
        if (
            tuple(previous.denormalized_joint_positions.shape)
            != (
                _VLA_ACTION_ROWS,
                29,
            )
            or not torch.isfinite(previous.denormalized_joint_positions).all()
        ):
            raise AssertionError("VLA RTC previous joint targets are invalid")

        source_cursor_h50, source_start_row_h30 = _rtc_source_cursor(
            source_timestamp_us=previous.source_timestamp_us,
            current_timestamp_us=policy_input.timestamp_us,
        )

        trace = policy_input.feedback_trace
        if trace is None:
            raise ValueError("VLA RTC replan is missing realized feedback")
        if int(trace.env_id) != int(policy_input.env_id):
            raise ValueError("VLA RTC feedback targets a different env lane")
        predecessor_tick_count = _initial_predecessor_tick_count(
            trace,
            installed_reference_id=previous.reference_id,
        )
        # Feedback is on the reference controller's 50 Hz clock.  Ceil rather
        # than round so the
        # delay predictor never exposes a row that may still be executing.
        observed_delay_rows = (
            predecessor_tick_count * _VLA_CONTROL_HZ + _REFERENCE_CONTROL_HZ - 1
        ) // _REFERENCE_CONTROL_HZ
        next_delay_rows = (
            *rtc_state.delay_rows,
            observed_delay_rows,
        )[-_RTC_DELAY_WINDOW_SIZE:]

        source_suffix = previous_chunk[source_start_row_h30:]
        configured_continuity_prefix_rows = lane.qualification_continuity_prefix_rows
        if configured_continuity_prefix_rows > 0:
            requested_prefix_rows = configured_continuity_prefix_rows
            prefix_mode = _RTC_PREFIX_MODE_FIXED_CONTINUITY
        else:
            requested_prefix_rows = max(next_delay_rows)
            prefix_mode = _RTC_PREFIX_MODE_LATENCY_PREDICTOR
        prefix_rows = min(
            requested_prefix_rows,
            int(source_suffix.shape[0]),
            actor_critic.rtc_max_delay_exclusive - 1,
        )
        if source_suffix.shape[0] > 0:
            qpos = torch.as_tensor(policy_input.qpos, dtype=torch.float64).reshape(-1)
            if tuple(qpos.shape) != (36,) or not torch.isfinite(qpos).all():
                raise ValueError("VLA RTC current qpos must be finite shape (36,)")
            rebased_suffix = _rebase_normalized_chunk_relative_dims(
                source_suffix,
                previous_quaternion_wxyz=previous.base_quaternion_wxyz,
                previous_xy=previous.base_xy,
                current_quaternion_wxyz=qpos[3:7],
                current_xy=qpos[:2],
                normalizer=actor_critic.normalizer,
            )
            # Native prev_actions is the real suffix followed by zero padding.
            # Only the first predictor-selected rows are frozen by the mask.
            actions[0, : rebased_suffix.shape[0]] = rebased_suffix.to(
                device=device, dtype=torch.float32
            )
            mask[0, :prefix_rows] = True

        predecessor_row = max(0, source_start_row_h30 - 1)
        velocity_seed = previous.denormalized_joint_positions[predecessor_row].clone()
        return _RtcInputs(
            normalized_actions=actions,
            mask=mask,
            source_cursor_h50=source_cursor_h50,
            source_start_row_h30=source_start_row_h30,
            velocity_seed_joint_position=velocity_seed,
            next_delay_rows=next_delay_rows,
            observed_delay_rows=observed_delay_rows,
            configured_continuity_prefix_rows=configured_continuity_prefix_rows,
            prefix_mode=prefix_mode,
        )

    def _observation_inputs(
        self,
        *,
        lane: _Lane,
        policy_input: HumanoidPolicyInput,
        actor_critic: VlaPsiActorCritic,
    ) -> tuple[_VisualInputs, dict[str, torch.Tensor], dict[str, object]]:
        """Build and freeze the exact native Psi visual/proprio condition."""
        if len(policy_input.camera_frames) != 1:
            raise ValueError("VLA policy requires exactly one current D435 frame")
        frame = policy_input.camera_frames[0]
        self._validate_same_shot_state(policy_input, frame)
        if hashlib.sha256(frame.image_bytes).hexdigest() != frame.image_sha256:
            raise ValueError("VLA source JPEG digest changed after camera routing")
        source_identity = VlaImageSourceIdentity(
            env_id=frame.env_id,
            frame_start_us=frame.frame_start_us,
            frame_end_us=frame.frame_end_us,
            logical_id=frame.logical_id,
            byte_length=len(frame.image_bytes),
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
        preprocess_profile = self._bundle_profile.camera_preprocess_profile
        if preprocess_profile == (
            "psi_resize_nearest_224x224_center_crop_224x224_no_letterbox.v1"
        ):
            current = decode_native_d435(
                frame.image_bytes, image_format=self._image_format
            )
        elif preprocess_profile == "legacy_d455_letterbox_224x140_to_224x224.v1":
            current = decode_legacy_d455_letterbox(
                frame.image_bytes, image_format=self._image_format
            )
        else:
            raise ValueError("VLA bundle uses an unsupported camera preprocess ABI")
        selected = lane.history.select_with_current(
            current,
            source_identity=source_identity,
        )
        prepared: tuple[Any, ...] = ()
        try:
            psi = cast(Any, actor_critic.psi_model)
            if preprocess_profile.startswith("psi_resize_nearest_"):
                prepared = self._apply_checkpoint_eval_image_transform(
                    psi,
                    selected,
                )
            else:
                prepared = selected
            built = self._build_training_compatible_vlm_batch(
                psi,
                [list(prepared)],
                [self._language_instruction],
            )
        finally:
            # Torchvision currently allocates transformed PIL images, but close
            # by object identity so a future identity transform remains safe.
            closed: set[int] = set()
            for image in (current, *selected, *prepared):
                identity = id(image)
                if identity in closed:
                    continue
                image.close()
                closed.add(identity)
        if not isinstance(built, tuple) or len(built) != 6:
            raise TypeError("Psi _build_vlm_batch returned an unexpected contract")
        input_ids, attention_mask, pixel_values, grid, effective_grid, pool_factors = (
            built
        )
        tensors = tuple(
            torch.as_tensor(value) if value is not None else None for value in built
        )
        input_ids, attention_mask, pixel_values, grid, effective_grid, pool_factors = (
            tensors
        )
        assert input_ids is not None and attention_mask is not None
        assert pixel_values is not None and grid is not None
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("Psi input_ids must have shape [1, S]")
        if attention_mask.shape != input_ids.shape:
            raise ValueError("Psi attention_mask differs from input_ids")
        if pixel_values.ndim != 2 or grid.ndim != 2 or grid.shape[1] != 3:
            raise ValueError("Psi visual patch tensors have an invalid shape")
        image_patch_counts = grid.to(dtype=torch.int64).prod(dim=1)
        if int(image_patch_counts.sum().item()) != int(pixel_values.shape[0]):
            raise ValueError("Psi grid metadata differs from visual patch rows")
        image_count = int(grid.shape[0])
        patch_count = int(pixel_values.shape[0])
        device = input_ids.device
        if (effective_grid is None) != (pool_factors is None):
            raise ValueError(
                "Psi effective visual grids and pool factors must be present together"
            )
        if effective_grid is None:
            # Psi treats this pair as optional for an uncompressed history.  Use
            # its documented identity defaults explicitly so behavior sampling
            # and trainer replay execute the exact same visual-conditioning
            # branch even when later BATS decisions carry real pooling metadata.
            effective_grid = grid.clone()
            pool_factors = torch.ones(
                image_count,
                dtype=torch.int64,
                device=grid.device,
            )
        assert pool_factors is not None
        physical_state = torch.tensor(
            frame.policy_joint_position,
            dtype=torch.float32,
            device=actor_critic.normalizer.state_low.device,
        ).reshape(1, 1, 29)
        model_inputs: _VisualInputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": grid,
            "sequence_lengths": attention_mask.to(dtype=torch.int64).sum(dim=1),
            "image_counts": torch.tensor(
                [image_count], dtype=torch.int64, device=device
            ),
            "image_offsets": torch.tensor(
                [0, image_count], dtype=torch.int64, device=device
            ),
            "image_patch_counts": image_patch_counts,
            "patch_counts": torch.tensor(
                [patch_count], dtype=torch.int64, device=device
            ),
            "patch_offsets": torch.tensor(
                [0, patch_count], dtype=torch.int64, device=device
            ),
            "physical_states": physical_state,
            "effective_image_grid_thw": effective_grid,
            "visual_pool_factors": pool_factors,
        }
        replay_visual: dict[str, torch.Tensor] = {
            "input_ids": _cpu_clone(input_ids[0]),
            "attention_mask": _cpu_clone(attention_mask[0]),
            "pixel_values": _cpu_clone(pixel_values),
            "image_grid_thw": _cpu_clone(grid),
            "physical_states": _cpu_clone(physical_state[0]),
            "selected_history_indices": torch.tensor(
                lane.history.last_selected_indices, dtype=torch.int64
            ),
        }
        replay_visual["effective_image_grid_thw"] = _cpu_clone(effective_grid)
        replay_visual["visual_pool_factors"] = _cpu_clone(pool_factors)
        selected_identities = lane.history.last_selected_source_identities
        if len(selected_identities) != image_count:
            raise AssertionError(
                "VLA selected image identities differ from processor input"
            )
        source_frames = [
            identity.manifest_entry(
                role="current" if index == image_count - 1 else "history"
            )
            for index, identity in enumerate(selected_identities)
        ]
        visual_input_manifest: dict[str, object] = {
            "schema": HUMANOID_VISUAL_INPUT_MANIFEST_SCHEMA,
            "session_uuid": policy_input.session_uuid,
            "episode_id": int(policy_input.episode_id),
            "step_index": int(policy_input.step_index),
            "timestamp_us": int(policy_input.timestamp_us),
            "env_id": int(policy_input.env_id),
            "decision_id": int(policy_input.decision_id),
            "camera_logical_id": frame.logical_id,
            "image_format": self._image_format,
            "preprocess_profile": preprocess_profile,
            "instruction_sha256": self._instruction_sha256,
            "source_frames": source_frames,
            "pixel_values": {
                "dtype": str(replay_visual["pixel_values"].dtype),
                "shape": list(replay_visual["pixel_values"].shape),
                "sha256": humanoid_model_input_tensor_sha256(
                    replay_visual["pixel_values"]
                ),
            },
        }
        return model_inputs, replay_visual, visual_input_manifest

    @staticmethod
    def _apply_checkpoint_eval_image_transform(
        psi: Any,
        observations: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        """Apply ckpt_1500's exact eval geometry before the serving builder.

        Psi's serving ``_build_vlm_batch`` calls ``build_qwenvl_inputs``
        directly and therefore skips ``Psi0ModelTransform.__call__``.  Training
        first applied ``ResizeImage((224, 224), NEAREST)`` and then
        ``CenterCrop((224, 224))`` to each native 640x480 D435 frame.  Reusing
        the checkpoint-owned transform objects keeps that ABI exact while
        intentionally omitting training-only color jitter at evaluation time.
        """
        model_transform = getattr(psi, "model_transform", None)
        if model_transform is None:
            raise TypeError("Psi checkpoint is missing its model transform")
        if bool(getattr(model_transform, "adaptive_resize", False)):
            raise ValueError("VLA checkpoint must disable adaptive image resize")
        resize_spec = getattr(model_transform, "resize", None)
        crop_spec = getattr(model_transform, "center_crop", None)
        if resize_spec is None or crop_spec is None:
            raise TypeError("Psi checkpoint is missing resize or center-crop config")

        def _size(value: Any, *, label: str) -> tuple[int, int]:
            raw = getattr(value, "size", None)
            if isinstance(raw, int):
                result = (raw, raw)
            elif isinstance(raw, (list, tuple)) and len(raw) == 2:
                result = (int(raw[0]), int(raw[1]))
            else:
                raise TypeError(f"Psi {label} size has an unexpected contract")
            if result != (224, 224):
                raise ValueError(f"Psi {label} must be exactly 224x224")
            return result

        _size(resize_spec, label="resize")
        _size(crop_spec, label="center crop")
        if not callable(resize_spec) or not callable(crop_spec):
            raise TypeError("Psi resize and center-crop configs must be callable")
        resizer = resize_spec()
        center_crop = crop_spec()
        if not callable(resizer) or not callable(center_crop):
            raise TypeError("Psi resize and center-crop transforms must be callable")
        interpolation = getattr(resizer, "interpolation", None)
        if str(getattr(interpolation, "name", "")).upper() != "NEAREST":
            raise ValueError("Psi resize interpolation must be NEAREST")

        transformed: list[Any] = []
        try:
            for image in observations:
                if (
                    getattr(image, "size", None) != (640, 480)
                    or getattr(image, "mode", None) != "RGB"
                ):
                    raise ValueError("Psi source image must be native 640x480 RGB")
                result = center_crop(resizer(image))
                if (
                    getattr(result, "size", None) != (224, 224)
                    or getattr(result, "mode", None) != "RGB"
                ):
                    raise ValueError(
                        "Psi checkpoint image transform did not produce 224x224 RGB"
                    )
                transformed.append(result)
        except BaseException:
            for image in transformed:
                image.close()
            raise
        return tuple(transformed)

    @staticmethod
    def _build_training_compatible_vlm_batch(
        psi: Any,
        observations: list[list[Any]],
        instructions: list[str],
    ) -> tuple[Any, ...]:
        """Bridge both attested Psi serving contracts without changing training semantics.

        The legacy Psi snapshot returns the six tensors needed for pooled BATS
        history.  The stairs-blocks snapshot predates that serving fix and
        returns only four tensors even though its training transform enables
        ``compress_history_visual_tokens``.  In that case, rebuilding through
        the checkpoint's own model transform is required: merely appending
        identity pooling metadata would retain unpooled history tokens that the
        checkpoint did not see during training.
        """
        built = psi._build_vlm_batch(observations, instructions)
        if not isinstance(built, tuple):
            raise TypeError("Psi _build_vlm_batch returned an unexpected contract")
        if len(built) == 6:
            return built
        if len(built) != 4:
            raise TypeError("Psi _build_vlm_batch returned an unexpected contract")

        model_transform = getattr(psi, "model_transform", None)
        compress_history = bool(
            getattr(model_transform, "compress_history_visual_tokens", False)
        )
        if not compress_history:
            return (*built, None, None)
        if len(observations) != 1 or len(instructions) != 1:
            raise ValueError(
                "pooled Psi serving compatibility requires one policy sample"
            )
        if model_transform is None or not hasattr(
            model_transform, "build_qwenvl_inputs"
        ):
            raise TypeError(
                "Psi checkpoint requests visual-history compression but its "
                "training transform is unavailable"
            )
        images = observations[0]
        if not images:
            raise ValueError("Psi visual condition must contain a current image")
        pool_factor = int(
            getattr(model_transform, "history_visual_pool_factor", 1) or 1
        )
        if pool_factor < 1:
            raise ValueError("Psi history_visual_pool_factor must be positive")
        image_pool_factors = [pool_factor for _ in images[:-1]] + [1]
        rebuilt = model_transform.build_qwenvl_inputs(
            psi.vlm_processor,
            images,
            instructions[0],
            image_pool_factors=image_pool_factors,
        )
        device = getattr(psi, "device", None)
        if hasattr(rebuilt, "to"):
            rebuilt = rebuilt.to(device)
        elif isinstance(rebuilt, Mapping):
            rebuilt = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in rebuilt.items()
            }
        else:
            raise TypeError("Psi training transform returned an unexpected contract")
        required = (
            "input_ids",
            "attention_mask",
            "pixel_values",
            "image_grid_thw",
            "effective_image_grid_thw",
            "visual_pool_factors",
        )
        missing = [key for key in required if key not in rebuilt]
        if missing:
            raise TypeError(
                f"Psi pooled training transform omitted required tensors {missing}"
            )
        return tuple(rebuilt[key] for key in required)

    @staticmethod
    def _validate_same_shot_state(
        policy_input: HumanoidPolicyInput,
        frame: HumanoidCameraFrame,
    ) -> None:
        """Require VLA proprio to come from the image's render snapshot."""
        if frame.env_id != policy_input.env_id:
            raise ValueError("VLA D435 frame targets a different env lane")
        qpos = torch.as_tensor(policy_input.qpos, dtype=torch.float32)
        render_qpos = torch.tensor(frame.render_qpos, dtype=torch.float32)
        if tuple(qpos.shape) != (36,) or tuple(render_qpos.shape) != (36,):
            raise ValueError("VLA G1 qpos and render_qpos must both be 36-D")
        if not torch.equal(qpos, render_qpos):
            raise ValueError("VLA image and policy qpos are not the same shot")
        if tuple(frame.policy_joint_position) != tuple(frame.render_qpos[7:]):
            raise AssertionError("camera policy_joint_position changed semantics")

    def _sample_output(
        self,
        *,
        lane: _Lane,
        policy_input: HumanoidPolicyInput,
        sample: VlaPolicySample | VlaQualificationSample | Any,
        replay_visual: dict[str, torch.Tensor],
        visual_input_manifest: dict[str, object],
        rtc: _RtcInputs,
        actor_critic: VlaPsiActorCritic,
    ) -> HumanoidPolicyStepOutput:
        """Materialize the H50 buffer while retaining the raw Flow action."""
        denormalized_wire = torch.as_tensor(sample.denormalized_wire)
        if tuple(denormalized_wire.shape) != (1, 30, 38):
            raise ValueError("VLA sampled wire chunk must have shape [1,30,38]")
        rows = _cpu_clone(denormalized_wire[0]).to(dtype=torch.float32)
        built_reference = self._reference_adapter.build(
            rows.numpy(),
            qpos=_cpu_clone(policy_input.qpos).numpy(),
            qvel=_cpu_clone(policy_input.qvel).numpy(),
            timestamp_us=policy_input.timestamp_us,
            velocity_seed_joint_position=(
                None
                if rtc.velocity_seed_joint_position is None
                else rtc.velocity_seed_joint_position.numpy()
            ),
        )
        reference_id = (
            (int(policy_input.env_id) + 1) << 32
        ) | lane.next_reference_sequence
        reference = _wire_reference(
            built_reference.reference,
            decode_context=built_reference.decode_context,
            reference_id=reference_id,
            source_decision_id=policy_input.decision_id,
        )
        sampled_clipped = getattr(sample, "clipped_normalized", None)
        if sampled_clipped is None:
            sampled_clipped = actor_critic.normalizer.normalize_action(
                denormalized_wire
            )
        sampled_clipped = torch.as_tensor(sampled_clipped)
        if (
            tuple(sampled_clipped.shape)
            != (
                1,
                _VLA_ACTION_ROWS,
                _VLA_ACTION_WIDTH,
            )
            or not torch.isfinite(sampled_clipped).all()
        ):
            raise ValueError(
                "VLA sampled clipped-normalized chunk must have shape [1,30,38]"
            )
        if torch.any((sampled_clipped < -1.0) | (sampled_clipped > 1.0)):
            raise ValueError("VLA sampled clipped-normalized chunk left [-1,1]")
        sampled_density = getattr(sample, "density_latent", None)
        if sampled_density is not None:
            sampled_density = torch.as_tensor(sampled_density)
            if (
                tuple(sampled_density.shape) != (1, _VLA_ACTION_ROWS, _VLA_ACTION_WIDTH)
                or not torch.isfinite(sampled_density).all()
            ):
                raise ValueError("VLA sampled density latent must have shape [1,30,38]")
        qualification_schedule_sha256: str | None = None
        qualification_clip_normalized_actions: bool | None = None
        if self._sampling_mode == NATIVE_ODE_QUALIFICATION:
            if sampled_density is None:
                raise ValueError(
                    "VLA native qualification sample is missing its raw ODE latent"
                )
            qualification_schedule_sha256 = str(getattr(sample, "schedule_sha256", ""))
            if (
                qualification_schedule_sha256
                != actor_critic.qualification_schedule.sha256
            ):
                raise ValueError(
                    "VLA native qualification sample changed its serving schedule"
                )
            sample_clip = getattr(sample, "clip_normalized_actions", None)
            if sample_clip is not actor_critic.qualification_clip_normalized_actions:
                raise ValueError(
                    "VLA native qualification sample changed its wire clip contract"
                )
            qualification_clip_normalized_actions = bool(sample_clip)
            rtc_normalized = (
                sampled_clipped
                if actor_critic.qualification_clip_normalized_actions
                else sampled_density
            )
        else:
            rtc_normalized = sampled_clipped
        if bool(rtc.mask.any()) and not torch.equal(
            rtc_normalized.to(
                device=rtc.normalized_actions.device,
                dtype=rtc.normalized_actions.dtype,
            )[rtc.mask[..., None].expand_as(rtc.normalized_actions)],
            rtc.normalized_actions[
                rtc.mask[..., None].expand_as(rtc.normalized_actions)
            ],
        ):
            raise AssertionError("VLA sampler changed the fixed RTC prefix")
        payload: dict[str, Any] = dict(replay_visual)
        visual_input_manifest_sha256 = humanoid_visual_input_manifest_sha256(
            visual_input_manifest
        )
        payload.update(
            {
                "sampling_mode": self._sampling_mode,
                "denormalized_wire_actions": rows.clone(),
                "rtc_prefix_normalized_actions": _cpu_clone(rtc.normalized_actions[0]),
                "rtc_prefix_mask": _cpu_clone(rtc.mask[0]),
                "rtc_prefix_mode": rtc.prefix_mode,
                "rtc_configured_continuity_prefix_rows": (
                    rtc.configured_continuity_prefix_rows
                ),
                "rtc_observed_delay_rows": rtc.observed_delay_rows,
                "instruction_sha256": self._instruction_sha256,
                # This digest is attested together with the checkpoint source
                # bundle. It is the only run-config provenance stamped into
                # replay; no independently configured manifest identity exists.
                "run_config_sha256": self._run_config_sha256,
                "camera_render_receipt_sha256": policy_input.camera_frames[
                    0
                ].render_receipt_sha256,
                "visual_input_manifest_sha256": visual_input_manifest_sha256,
                "humanoid": {"vla_raw_action_rows": rows.clone()},
            }
        )
        trace = getattr(sample, "trace", None)
        if self._sampling_mode == FLOW_SDE_TRAINING and trace is None:
            raise ValueError("VLA Flow-SDE training sample is missing its PPO trace")
        if self._sampling_mode == NATIVE_ODE_QUALIFICATION and trace is not None:
            raise ValueError("VLA native ODE qualification sample carried a PPO trace")
        if self._sampling_mode == NATIVE_ODE_QUALIFICATION:
            forbidden = (
                "old_element_logprobs",
                "old_log_probs",
                "values",
            )
            leaked = [
                name for name in forbidden if getattr(sample, name, None) is not None
            ]
            if leaked:
                raise ValueError(
                    f"VLA native ODE qualification sample carried PPO fields {leaked}"
                )
            if (
                sampled_density is None
                or qualification_schedule_sha256 is None
                or qualification_clip_normalized_actions is None
            ):
                raise AssertionError(
                    "VLA qualification invariants changed after validation"
                )
            payload.update(
                {
                    "density_latent": _cpu_clone(sampled_density[0]),
                    "clipped_normalized_actions": _cpu_clone(sampled_clipped[0]),
                    "qualification_schedule_sha256": qualification_schedule_sha256,
                    "qualification_inference_steps": int(
                        actor_critic.qualification_schedule.num_steps
                    ),
                    "qualification_clip_normalized_actions": (
                        qualification_clip_normalized_actions
                    ),
                }
            )
        old_logprob: torch.Tensor | None = None
        value: torch.Tensor | None = None
        if trace is not None:
            chain = torch.as_tensor(trace.chain)
            if tuple(chain.shape[:2]) != (1, 11):
                raise ValueError("VLA sampled chain must contain 10 denoise steps")
            clipped_normalized_raw = getattr(sample, "clipped_normalized", None)
            old_element_logprobs_raw = getattr(sample, "old_element_logprobs", None)
            old_log_probs_raw = getattr(sample, "old_log_probs", None)
            values_raw = getattr(sample, "values", None)
            required = {
                "clipped_normalized": clipped_normalized_raw,
                "old_element_logprobs": old_element_logprobs_raw,
                "old_log_probs": old_log_probs_raw,
                "values": values_raw,
            }
            missing = [name for name, field in required.items() if field is None]
            if missing:
                raise ValueError(
                    f"VLA trainable sample is missing Flow replay fields {missing}"
                )
            old_element_logprobs = torch.as_tensor(old_element_logprobs_raw)
            if tuple(old_element_logprobs.shape) != (
                1,
                _VLA_ACTION_ROWS * _VLA_ACTION_WIDTH,
            ):
                raise ValueError(
                    "VLA sampled element logprobs must have shape [1,1140]"
                )
            masked_elements = rtc.mask[..., None].expand(-1, -1, _VLA_ACTION_WIDTH)
            if not torch.equal(
                old_element_logprobs.reshape(1, _VLA_ACTION_ROWS, _VLA_ACTION_WIDTH)[
                    masked_elements
                ],
                torch.zeros_like(
                    old_element_logprobs.reshape(
                        1, _VLA_ACTION_ROWS, _VLA_ACTION_WIDTH
                    )[masked_elements]
                ),
            ):
                raise AssertionError("VLA fixed RTC prefix contributed rollout density")
            payload.update(
                {
                    "latent_chain": _cpu_clone(chain[0]),
                    "denoise_index": _cpu_clone(
                        torch.as_tensor(trace.denoise_indices)[0]
                    ),
                    "clipped_normalized_actions": _cpu_clone(
                        torch.as_tensor(clipped_normalized_raw)[0]
                    ),
                    "schedule_sha256": str(trace.schedule_sha256),
                    "flow_noise_level": VLA_FLOW_NOISE_LEVEL,
                    "flow_ignore_last": VLA_FLOW_IGNORE_LAST,
                    "old_element_logprobs": _cpu_clone(old_element_logprobs[0]),
                }
            )
            old_logprob = _cpu_clone(torch.as_tensor(old_log_probs_raw)[0]).reshape(())
            value = _cpu_clone(torch.as_tensor(values_raw)[0]).reshape(())
        replay = PolicyReplayData(
            replay_schema_version=1,
            payload_schema=(
                REPLAY_SCHEMA
                if self._sampling_mode == FLOW_SDE_TRAINING
                else QUALIFICATION_REPLAY_SCHEMA
            ),
            payload_schema_version=1,
            model_family=MODEL_FAMILY,
            action_selection=ActionSelection(set_ix=0, sample_ix=0),
            old_logprob=old_logprob,
            payload=payload,
        )
        # Commit RTC state only after the complete output and replay sample
        # have passed validation; a failed inference must not become the next
        # chunk's predecessor.
        qpos = _cpu_clone(policy_input.qpos).to(dtype=torch.float64).reshape(-1)
        previous_chunk = _PreviousRtcChunk(
            reference_id=reference_id,
            source_timestamp_us=int(policy_input.timestamp_us),
            normalized_actions=_cpu_clone(rtc_normalized[0]).to(dtype=torch.float32),
            denormalized_joint_positions=rows[:, :29].clone(),
            base_quaternion_wxyz=qpos[3:7].clone(),
            base_xy=qpos[:2].clone(),
        )
        lane.rtc_state = _RtcLaneState(
            delay_rows=rtc.next_delay_rows,
            previous_chunk=previous_chunk,
        )
        lane.next_reference_sequence += 1
        return HumanoidPolicyStepOutput(
            env_id=policy_input.env_id,
            motion_reference=reference,
            logprob=old_logprob,
            value=value,
            replay_data=replay,
            model_extra={
                "humanoid_visual_input_manifest": visual_input_manifest,
                "vla_history_frame_count": int(
                    replay_visual["selected_history_indices"].numel()
                ),
                "vla_rtc_delay_rows": int(rtc.mask.sum().item()),
                "vla_rtc_actual_prefix_rows": int(rtc.mask.sum().item()),
                "vla_rtc_configured_continuity_prefix_rows": (
                    rtc.configured_continuity_prefix_rows
                ),
                "vla_rtc_observed_delay_rows": rtc.observed_delay_rows,
                "vla_rtc_prefix_mode": rtc.prefix_mode,
                "vla_rtc_source_cursor_h50": rtc.source_cursor_h50,
                "vla_rtc_source_start_row_h30": rtc.source_start_row_h30,
                "vla_raw_action_rows": 30,
                "vla_reference_frames": H50_FRAME_COUNT,
                "vla_replan_controller_ticks": self._replan_controller_ticks,
            },
        )

    @staticmethod
    def _autocast(actor_critic: VlaPsiActorCritic) -> torch.autocast:
        """Return the checkpoint's CUDA bfloat16 autocast context."""
        device = actor_critic.normalizer.state_low.device
        return torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        )


def build_humanoid_policy_factory(
    run_config: Any,
    inference_engine: _SessionModelLookup,
):
    """Build motion-reference VLA sessions from strict bundle config."""
    sampling_mode = vla_sampling_mode(run_config)
    config = run_config.policy.model.bundle_config
    if not isinstance(config, Mapping):
        raise TypeError("VLA bundle_config must be a mapping")
    expected_schedule = _required_sha(config, "expected_schedule_sha256")
    humanoid_repo_path = (
        Path(_required_text(config, "humanoid_repo_path")).expanduser().resolve()
    )
    humanoid_vla_reference_adapter_sha256 = _required_sha(
        config, "humanoid_vla_reference_adapter_sha256"
    )
    humanoid_motion_reference_sha256 = _required_sha(
        config, "humanoid_motion_reference_sha256"
    )
    language_instruction = _required_text(config, "language_instruction")
    camera_logical_id = _required_text(config, "camera_logical_id")
    camera_format = _required_text(config, "camera_image_format")
    camera_contract_sha256 = _required_sha(config, "camera_contract_sha256")
    model_root = (
        Path(str(run_config.policy.model.path)).expanduser().resolve(strict=False)
    )
    profile = vla_bundle_profile_for_model_root(model_root)
    actual_camera_abi = (
        camera_logical_id,
        camera_format,
        camera_contract_sha256,
        language_instruction,
    )
    expected_camera_abi = (
        profile.camera_logical_id,
        profile.camera_image_format,
        profile.camera_contract_sha256,
        profile.language_instruction,
    )
    if actual_camera_abi != expected_camera_abi:
        raise ValueError(
            "VLA bundle camera/instruction ABI differs from its attested model profile"
        )

    def _factory(session_uuid: str, request: Any) -> G1VlaHumanoidPolicy:
        """Validate one reserved H50 session before any model execution."""
        replan_controller_ticks = _validate_session_request(
            request,
            camera_logical_id=camera_logical_id,
            camera_format=camera_format,
            camera_contract_sha256=camera_contract_sha256,
            camera_resolution=profile.camera_source_resolution,
            camera_profile=profile.camera_profile,
        )
        return G1VlaHumanoidPolicy(
            inference_engine,
            session_uuid=session_uuid,
            request=request,
            image_format=camera_format,
            expected_schedule_sha256=expected_schedule,
            humanoid_repo_path=humanoid_repo_path,
            humanoid_vla_reference_adapter_sha256=humanoid_vla_reference_adapter_sha256,
            humanoid_motion_reference_sha256=humanoid_motion_reference_sha256,
            language_instruction=language_instruction,
            run_config_sha256=profile.run_config_sha256,
            bundle_profile=profile,
            sampling_mode=sampling_mode,
            replan_controller_ticks=replan_controller_ticks,
        )

    return _factory


def _validate_session_request(
    request: Any,
    *,
    camera_logical_id: str,
    camera_format: str,
    camera_contract_sha256: str,
    camera_resolution: tuple[int, int],
    camera_profile: str,
) -> int:
    """Require the H50, canonical-joint, and profile-owned camera ABI."""
    if int(request.execution_mode) != HUMANOID_EXECUTION_MODE_MOTION_REFERENCE:
        raise ValueError("VLA policy requires motion-reference execution mode")
    if tuple(str(name) for name in request.joint_names) != MOTION_REFERENCE_JOINT_NAMES:
        raise ValueError("VLA policy joint order differs from canonical IsaacLab")
    if int(request.action_size) != 0:
        raise ValueError("VLA motion-reference action_size must be zero")
    spec = request.reference_spec
    actual_without_replan = (
        str(spec.schema),
        int(spec.frame_count),
        int(spec.sample_period_us),
        tuple(str(name) for name in spec.joint_names),
    )
    expected_without_replan = (
        _H50_SCHEMA,
        H50_FRAME_COUNT,
        REFERENCE_PERIOD_US,
        MOTION_REFERENCE_JOINT_NAMES,
    )
    replan_controller_ticks = int(spec.control_ticks_per_policy_step)
    if (
        actual_without_replan != expected_without_replan
        or replan_controller_ticks not in SUPPORTED_REPLAN_CONTROLLER_TICKS
    ):
        raise ValueError("VLA H50 motion-reference contract changed")
    if (
        str(spec.decode_context_schema)
        != HUMANOID_FULL_ROTATION_LOCAL_XY_DECODE_CONTEXT_SCHEMA
    ):
        raise ValueError("VLA decode-context session contract changed")
    if str(request.action_schema) != _H50_SCHEMA:
        raise ValueError("VLA action_schema must match the H50 reference schema")
    camera = request.policy_camera_spec
    camera_actual = (
        str(camera.schema),
        str(camera.logical_id),
        int(camera.width),
        int(camera.height),
        str(camera.image_format),
        str(camera.contract_sha256),
    )
    camera_expected = (
        "humanoid_policy_camera_rgb_qpos.v1",
        camera_logical_id,
        int(camera_resolution[0]),
        int(camera_resolution[1]),
        camera_format,
        camera_contract_sha256,
    )
    if camera_actual != camera_expected:
        raise ValueError(f"VLA {camera_profile} camera contract changed")
    for field in ("attempt_id", "scene_id", "scenario_id"):
        if not str(getattr(request, field)):
            raise ValueError(f"VLA session requires non-empty {field}")
    return replan_controller_ticks


def _wire_reference(
    reference: Any,
    *,
    decode_context: Any,
    reference_id: int,
    source_decision_id: int,
) -> HumanoidMotionReference:
    """Convert the canonical repository MotionReference to AlpaGym tensors."""
    frames = tuple(
        HumanoidMotionReferenceFrame(
            timestamp_us=round(float(frame.time_s) * 1_000_000.0),
            joint_position=torch.tensor(frame.joint_position, dtype=torch.float32),
            joint_velocity=torch.tensor(frame.joint_velocity, dtype=torch.float32),
            root_position=torch.tensor(frame.root_position, dtype=torch.float32),
            root_quaternion_wxyz=torch.tensor(
                frame.root_quaternion_wxyz, dtype=torch.float32
            ),
        )
        for frame in reference.frames
    )
    return HumanoidMotionReference(
        reference_id=reference_id,
        source_decision_id=source_decision_id,
        frames=frames,
        root_z_alignment_offset_m=0.0,
        decode_context=HumanoidReferenceDecodeContext(
            schema=str(decode_context.schema),
            chunk_base_quaternion_wxyz=torch.tensor(
                decode_context.chunk_base_quaternion_wxyz,
                dtype=torch.float32,
            ),
            local_xy_from_frame_zero=torch.tensor(
                decode_context.local_xy_from_frame_zero,
                dtype=torch.float32,
            ),
        ),
    )


def _rtc_source_cursor(
    *,
    source_timestamp_us: int,
    current_timestamp_us: int,
) -> tuple[int, int]:
    """Map one exact H50 source clock to the next unconsumed H30 row."""
    timestamps: list[int] = []
    for label, value in (
        ("source", source_timestamp_us),
        ("current", current_timestamp_us),
    ):
        if isinstance(value, bool):
            raise TypeError(f"VLA RTC {label} timestamp must be an integer")
        timestamp = int(value)
        if timestamp != value or timestamp < 0:
            raise ValueError(
                f"VLA RTC {label} timestamp must be a non-negative integer"
            )
        timestamps.append(timestamp)
    source, current = timestamps
    if current <= source:
        raise ValueError("VLA RTC source timestamps must be strictly increasing")
    elapsed_us = current - source
    if elapsed_us % REFERENCE_PERIOD_US != 0:
        raise ValueError("VLA RTC source time is not on the exact 20 ms H50 grid")
    cursor_h50 = elapsed_us // REFERENCE_PERIOD_US
    # H50 target n has source phase 3*n/5.  Ceil selects the first H30 row
    # that has not already begun by the launch instant; row 30 means exhausted.
    start_row_h30 = min(
        (cursor_h50 * _VLA_CONTROL_HZ + _REFERENCE_CONTROL_HZ - 1)
        // _REFERENCE_CONTROL_HZ,
        _VLA_ACTION_ROWS,
    )
    return cursor_h50, start_row_h30


def _initial_predecessor_tick_count(
    trace: HumanoidRealizedFeedbackTrace,
    *,
    installed_reference_id: int,
) -> int:
    """Count the initial 50 Hz ticks preceding one sampled plan install."""
    ticks = tuple(trace.ticks)
    first_installed = next(
        (
            index
            for index, tick in enumerate(ticks)
            if int(tick.active_reference_id) == int(installed_reference_id)
        ),
        len(ticks),
    )
    predecessor_ids = {
        int(tick.active_reference_id) for tick in ticks[:first_installed]
    }
    if len(predecessor_ids) > 1:
        raise ValueError(
            "VLA RTC feedback switched between multiple predecessor references"
        )
    if any(
        int(tick.active_reference_id) != int(installed_reference_id)
        for tick in ticks[first_installed:]
    ):
        raise ValueError("VLA RTC feedback left the newly installed reference")
    return first_installed


def _rebase_normalized_chunk_relative_dims(
    normalized_chunk: torch.Tensor,
    *,
    previous_quaternion_wxyz: torch.Tensor,
    previous_xy: torch.Tensor,
    current_quaternion_wxyz: torch.Tensor,
    current_xy: torch.Tensor,
    normalizer: Any,
) -> torch.Tensor:
    """Re-express VLA dims 29:38 in the current yaw-relative base.

    Joint targets are absolute and remain untouched.  Rotation reconstruction
    follows VLA's yaw-only convention; local translation continues to use
    the full base rotation, matching the captured deployment behavior.
    """
    chunk = (
        torch.as_tensor(normalized_chunk).detach().to(device="cpu", dtype=torch.float64)
    )
    if chunk.ndim != 2 or int(chunk.shape[1]) != _VLA_ACTION_WIDTH:
        raise ValueError("VLA RTC chunk must have shape [H,38]")
    if not torch.isfinite(chunk).all():
        raise ValueError("VLA RTC chunk contains non-finite values")
    low = (
        torch.as_tensor(normalizer.action_low)
        .detach()
        .to(device="cpu", dtype=torch.float64)
    )
    high = (
        torch.as_tensor(normalizer.action_high)
        .detach()
        .to(device="cpu", dtype=torch.float64)
    )
    if tuple(low.shape) != (_VLA_ACTION_WIDTH,) or tuple(high.shape) != (
        _VLA_ACTION_WIDTH,
    ):
        raise ValueError("VLA RTC action bounds must have shape (38,)")
    action_range = high - low
    if not torch.all(action_range > 0):
        raise ValueError("VLA RTC action bounds are degenerate")

    relative = (chunk[:, 29:38] + 1.0) * 0.5 * action_range[29:38] + low[29:38]
    previous_rotation = _quaternion_wxyz_to_rotation(previous_quaternion_wxyz)
    current_rotation = _quaternion_wxyz_to_rotation(current_quaternion_wxyz)
    previous_yaw_rotation = _yaw_rotation(previous_rotation)
    current_yaw_rotation_t = _yaw_rotation(current_rotation).T
    current_rotation_t = current_rotation.T
    previous_position = torch.tensor(
        [float(previous_xy[0]), float(previous_xy[1]), 0.0], dtype=torch.float64
    )
    current_position = torch.tensor(
        [float(current_xy[0]), float(current_xy[1]), 0.0], dtype=torch.float64
    )

    rebased_relative = torch.empty_like(relative)
    for row_index, row in enumerate(relative):
        relative_rotation = _rotation_6d_to_matrix(row[:6])
        world_rotation = previous_yaw_rotation @ relative_rotation
        world_position = previous_position + previous_rotation @ torch.tensor(
            [float(row[6]), float(row[7]), 0.0], dtype=torch.float64
        )
        current_relative_rotation = current_yaw_rotation_t @ world_rotation
        local_delta = current_rotation_t @ (world_position - current_position)
        rebased_relative[row_index, :3] = current_relative_rotation[:, 0]
        rebased_relative[row_index, 3:6] = current_relative_rotation[:, 1]
        rebased_relative[row_index, 6:8] = local_delta[:2]
        rebased_relative[row_index, 8] = math.atan2(
            float(current_relative_rotation[1, 0]),
            float(current_relative_rotation[0, 0]),
        )

    rebased = chunk.clone()
    rebased[:, 29:38] = (
        (rebased_relative - low[29:38]) / action_range[29:38] * 2.0 - 1.0
    ).clamp(-1.0, 1.0)
    return rebased.to(dtype=torch.as_tensor(normalized_chunk).dtype).contiguous()


def _quaternion_wxyz_to_rotation(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert one finite WXYZ quaternion to a normalized rotation matrix."""
    q = torch.as_tensor(quaternion).detach().to(device="cpu", dtype=torch.float64)
    if tuple(q.shape) != (4,) or not torch.isfinite(q).all():
        raise ValueError("VLA RTC base quaternion must be finite shape (4,)")
    norm = torch.linalg.vector_norm(q)
    if float(norm) > 1.0e-9:
        q = q / norm
    else:
        q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    w, x, y, z = (float(value) for value in q)
    return torch.tensor(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=torch.float64,
    )


def _rotation_6d_to_matrix(rotation_6d: torch.Tensor) -> torch.Tensor:
    """Project two rotation columns onto SO(3), matching VLA deployment."""
    value = torch.as_tensor(rotation_6d, dtype=torch.float64).reshape(-1)
    if tuple(value.shape) != (6,) or not torch.isfinite(value).all():
        raise ValueError("VLA RTC rotation-6D value must be finite shape (6,)")
    first = value[:3]
    first_norm = torch.linalg.vector_norm(first)
    column_zero = (
        first / first_norm
        if float(first_norm) > 1.0e-9
        else torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    )
    second = value[3:6] - torch.dot(column_zero, value[3:6]) * column_zero
    second_norm = torch.linalg.vector_norm(second)
    column_one = (
        second / second_norm
        if float(second_norm) > 1.0e-9
        else torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    )
    return torch.column_stack(
        (column_zero, column_one, torch.cross(column_zero, column_one, dim=0))
    )


def _yaw_rotation(rotation: torch.Tensor) -> torch.Tensor:
    """Return the Z-yaw component of a rotation matrix."""
    yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return torch.tensor(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )


def _required_text(config: Mapping[str, Any], name: str) -> str:
    """Read one required non-empty bundle string."""
    value = str(config[name])
    if not value.strip():
        raise ValueError(f"VLA bundle_config.{name} must be non-empty")
    return value


def _required_sha(config: Mapping[str, Any], name: str) -> str:
    """Read one required lowercase SHA-256 bundle identity."""
    value = _required_text(config, name)
    if _SHA256.fullmatch(value) is None:
        raise ValueError(f"VLA bundle_config.{name} must be lowercase SHA-256")
    return value


def _cpu_clone(value: torch.Tensor) -> torch.Tensor:
    """Freeze one tensor into independent contiguous CPU replay storage."""
    return value.detach().to(device="cpu").clone().contiguous()
