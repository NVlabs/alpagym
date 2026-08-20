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
from alpagym_runtime.alpasim.humanoid_policy_server import (
    HUMANOID_EXECUTION_MODE_MOTION_REFERENCE,
    MOTION_REFERENCE_JOINT_NAMES,
    HumanoidCameraFrame,
    HumanoidMotionReference,
    HumanoidMotionReferenceFrame,
    HumanoidPolicyInput,
    HumanoidPolicyStepOutput,
    HumanoidRealizedFeedbackTrace,
)
from alpagym_runtime.replay import ActionSelection, PolicyReplayData

from alpagym_g1_vla.bundle import MODEL_FAMILY, REPLAY_SCHEMA
from alpagym_g1_vla.history import (
    VlaImageHistory,
    stable_episode_int,
    width_fit_letterbox_d455,
)
from alpagym_g1_vla.flow import (
    VLA_FLOW_IGNORE_LAST,
    VLA_FLOW_NOISE_LEVEL,
)
from alpagym_g1_vla.inference_model import unwrap_actor_critic
from alpagym_g1_vla.model import VlaPolicySample, VlaPsiActorCritic
from alpagym_g1_vla.provenance import RUN_CONFIG_SHA256
from alpagym_g1_vla.reference_adapter import (
    H50_FRAME_COUNT,
    REFERENCE_PERIOD_US,
    REPLAN_CONTROLLER_TICKS,
    VlaMotionReferenceAdapter,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_H50_SCHEMA = "g1_motion_reference_29d_50hz_h50.v1"
_VLA_ACTION_ROWS = 30
_VLA_ACTION_WIDTH = 38
_RTC_INITIAL_DELAY_ROWS = 6
_RTC_DELAY_WINDOW_SIZE = 6
_VLA_CONTROL_HZ = 30
_REFERENCE_CONTROL_HZ = 50


@dataclass(frozen=True)
class _PreviousRtcChunk:
    """One validated sampled chunk and the source shot it is encoded against."""

    reference_id: int
    source_timestamp_us: int
    clipped_normalized_actions: torch.Tensor
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


@dataclass
class _Lane:
    """Per-env deterministic RNG and image-history ownership."""

    generator: torch.Generator
    history: VlaImageHistory
    reset_episode_id: int
    next_reference_sequence: int = 1
    rtc_state: _RtcLaneState | None = None

    def __post_init__(self) -> None:
        """Seed the native conservative delay predictor exactly once."""
        if self.rtc_state is None:
            self.rtc_state = _RtcLaneState(
                delay_rows=(_RTC_INITIAL_DELAY_ROWS,), previous_chunk=None
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
        if _RTC_INITIAL_DELAY_ROWS >= actor_critic.rtc_max_delay_exclusive:
            raise ValueError(
                "VLA initial RTC delay is outside the attested checkpoint contract"
            )
        if actor_critic.schedule.sha256 != self._expected_schedule_sha256:
            raise ValueError("VLA leased model flow schedule identity changed")
        if actor_critic.noise_level != VLA_FLOW_NOISE_LEVEL:
            raise ValueError(
                f"VLA profile requires Flow-SDE noise_level={VLA_FLOW_NOISE_LEVEL}"
            )
        if actor_critic.ignore_last is not VLA_FLOW_IGNORE_LAST:
            raise ValueError("VLA profile requires Flow-SDE ignore_last=true")

        for policy_input in policy_inputs:
            lane = self._lane(policy_input, actor_critic)
            self._validate_lifecycle(policy_input, sample_actions=sample_actions)
            visual_inputs, replay_visual = self._observation_inputs(
                lane=lane,
                policy_input=policy_input,
                actor_critic=actor_critic,
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
                sample = actor_critic.sample_actions(
                    **visual_inputs,
                    rtc_prefix_normalized_actions=rtc.normalized_actions,
                    rtc_prefix_mask=rtc.mask,
                    generator=lane.generator,
                )
            output = self._sample_output(
                lane=lane,
                policy_input=policy_input,
                sample=sample,
                replay_visual=replay_visual,
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
            )

        previous_chunk = previous.clipped_normalized_actions
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
        prefix_rows = min(
            max(next_delay_rows),
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
        )

    def _observation_inputs(
        self,
        *,
        lane: _Lane,
        policy_input: HumanoidPolicyInput,
        actor_critic: VlaPsiActorCritic,
    ) -> tuple[_VisualInputs, dict[str, torch.Tensor]]:
        """Build and freeze the exact native Psi visual/proprio condition."""
        if len(policy_input.camera_frames) != 1:
            raise ValueError("VLA policy requires exactly one current D455 frame")
        frame = policy_input.camera_frames[0]
        self._validate_same_shot_state(policy_input, frame)
        current = width_fit_letterbox_d455(
            frame.image_bytes, image_format=self._image_format
        )
        selected = lane.history.select_with_current(
            current,
            timestamp_us=frame.render_timestamp_us,
            capture_receipt_sha256=frame.render_receipt_sha256,
        )
        try:
            psi = cast(Any, actor_critic.psi_model)
            built = psi._build_vlm_batch(
                [list(selected)],
                [self._language_instruction],
            )
        finally:
            current.close()
            for image in selected:
                image.close()
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
        return model_inputs, replay_visual

    @staticmethod
    def _validate_same_shot_state(
        policy_input: HumanoidPolicyInput,
        frame: HumanoidCameraFrame,
    ) -> None:
        """Require VLA proprio to come from the image's render snapshot."""
        if frame.env_id != policy_input.env_id:
            raise ValueError("VLA D455 frame targets a different env lane")
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
        sample: VlaPolicySample | Any,
        replay_visual: dict[str, torch.Tensor],
        rtc: _RtcInputs,
        actor_critic: VlaPsiActorCritic,
    ) -> HumanoidPolicyStepOutput:
        """Materialize the H50 buffer while retaining the raw Flow action."""
        denormalized_wire = torch.as_tensor(sample.denormalized_wire)
        if tuple(denormalized_wire.shape) != (1, 30, 38):
            raise ValueError("VLA sampled wire chunk must have shape [1,30,38]")
        rows = _cpu_clone(denormalized_wire[0]).to(dtype=torch.float32)
        canonical_reference = self._reference_adapter.build(
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
            canonical_reference,
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
        if bool(rtc.mask.any()) and not torch.equal(
            sampled_clipped.to(
                device=rtc.normalized_actions.device,
                dtype=rtc.normalized_actions.dtype,
            )[rtc.mask[..., None].expand_as(rtc.normalized_actions)],
            rtc.normalized_actions[
                rtc.mask[..., None].expand_as(rtc.normalized_actions)
            ],
        ):
            raise AssertionError("VLA Flow sampler changed the fixed RTC prefix")
        payload: dict[str, Any] = dict(replay_visual)
        payload.update(
            {
                "denormalized_wire_actions": rows.clone(),
                "rtc_prefix_normalized_actions": _cpu_clone(rtc.normalized_actions[0]),
                "rtc_prefix_mask": _cpu_clone(rtc.mask[0]),
                "instruction_sha256": self._instruction_sha256,
                # This digest is attested together with the checkpoint source
                # bundle. It is the only run-config provenance stamped into
                # replay; no independently configured manifest identity exists.
                "run_config_sha256": RUN_CONFIG_SHA256,
                "camera_render_receipt_sha256": policy_input.camera_frames[
                    0
                ].render_receipt_sha256,
                "humanoid": {"vla_raw_action_rows": rows.clone()},
            }
        )
        trace = getattr(sample, "trace", None)
        old_logprob: torch.Tensor | None = None
        value: torch.Tensor | None = None
        if trace is not None:
            chain = torch.as_tensor(trace.chain)
            if tuple(chain.shape[:2]) != (1, 11):
                raise ValueError("VLA sampled chain must contain 10 denoise steps")
            required = (
                "clipped_normalized",
                "old_element_logprobs",
                "old_log_probs",
                "values",
            )
            missing = [name for name in required if getattr(sample, name, None) is None]
            if missing:
                raise ValueError(
                    f"VLA trainable sample is missing Flow replay fields {missing}"
                )
            old_element_logprobs = torch.as_tensor(sample.old_element_logprobs)
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
                        torch.as_tensor(sample.clipped_normalized)[0]
                    ),
                    "schedule_sha256": str(trace.schedule_sha256),
                    "flow_noise_level": VLA_FLOW_NOISE_LEVEL,
                    "flow_ignore_last": VLA_FLOW_IGNORE_LAST,
                    "old_element_logprobs": _cpu_clone(old_element_logprobs[0]),
                }
            )
            old_logprob = _cpu_clone(torch.as_tensor(sample.old_log_probs)[0]).reshape(
                ()
            )
            value = _cpu_clone(torch.as_tensor(sample.values)[0]).reshape(())
        replay = PolicyReplayData(
            replay_schema_version=1,
            payload_schema=REPLAY_SCHEMA,
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
            clipped_normalized_actions=_cpu_clone(sampled_clipped[0]).to(
                dtype=torch.float32
            ),
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
                "vla_history_frame_count": int(
                    replay_visual["selected_history_indices"].numel()
                ),
                "vla_rtc_delay_rows": int(rtc.mask.sum().item()),
                "vla_rtc_source_cursor_h50": rtc.source_cursor_h50,
                "vla_rtc_source_start_row_h30": rtc.source_start_row_h30,
                "vla_raw_action_rows": 30,
                "vla_reference_frames": H50_FRAME_COUNT,
                "vla_replan_controller_ticks": REPLAN_CONTROLLER_TICKS,
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
    if camera_format not in {"png", "jpeg"}:
        raise ValueError("VLA camera_image_format must be png or jpeg")
    camera_contract_sha256 = _required_sha(config, "camera_contract_sha256")

    def _factory(session_uuid: str, request: Any) -> G1VlaHumanoidPolicy:
        """Validate one reserved H50 session before any model execution."""
        _validate_session_request(
            request,
            camera_logical_id=camera_logical_id,
            camera_format=camera_format,
            camera_contract_sha256=camera_contract_sha256,
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
        )

    return _factory


def _validate_session_request(
    request: Any,
    *,
    camera_logical_id: str,
    camera_format: str,
    camera_contract_sha256: str,
) -> None:
    """Require the one-second H50, canonical-joint, and D455 ABI."""
    if int(request.execution_mode) != HUMANOID_EXECUTION_MODE_MOTION_REFERENCE:
        raise ValueError("VLA policy requires motion-reference execution mode")
    if tuple(str(name) for name in request.joint_names) != MOTION_REFERENCE_JOINT_NAMES:
        raise ValueError("VLA policy joint order differs from canonical IsaacLab")
    if int(request.action_size) != 0:
        raise ValueError("VLA motion-reference action_size must be zero")
    spec = request.reference_spec
    actual = (
        str(spec.schema),
        int(spec.frame_count),
        int(spec.sample_period_us),
        int(spec.control_ticks_per_policy_step),
        tuple(str(name) for name in spec.joint_names),
    )
    expected = (
        _H50_SCHEMA,
        H50_FRAME_COUNT,
        REFERENCE_PERIOD_US,
        REPLAN_CONTROLLER_TICKS,
        MOTION_REFERENCE_JOINT_NAMES,
    )
    if actual != expected:
        raise ValueError("VLA H50 motion-reference contract changed")
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
        224,
        140,
        camera_format,
        camera_contract_sha256,
    )
    if camera_actual != camera_expected:
        raise ValueError("VLA D455 camera contract changed")
    for field in ("attempt_id", "scene_id", "scenario_id"):
        if not str(getattr(request, field)):
            raise ValueError(f"VLA session requires non-empty {field}")


def _wire_reference(
    reference: Any,
    *,
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
        reference_sha256=str(reference.sha256),
        root_z_alignment_offset_m=0.0,
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
