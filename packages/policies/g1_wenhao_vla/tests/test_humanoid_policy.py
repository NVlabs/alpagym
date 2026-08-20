"""Closed-loop contract tests for the H50 Wenhao rollout policy."""

from __future__ import annotations

import hashlib
import io
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from alpagym_runtime.alpasim.humanoid_policy_server import (
    HUMANOID_EXECUTION_MODE_MOTION_REFERENCE,
    MOTION_REFERENCE_JOINT_NAMES,
    HumanoidCameraFrame,
    HumanoidPolicyInput,
    HumanoidRealizedControlTick,
    HumanoidRealizedFeedbackTrace,
)

from alpagym_g1_wenhao_vla.flow import WenhaoFlowSchedule
from alpagym_g1_wenhao_vla.history import (
    WenhaoImageHistory,
    image_array,
    select_bats_history_indices,
    stable_episode_int,
    width_fit_letterbox_d455,
)
from alpagym_g1_wenhao_vla.humanoid_policy import (
    G1WenhaoVlaHumanoidPolicy,
    _rebase_normalized_chunk_relative_dims,
    build_humanoid_policy_factory,
)
from alpagym_g1_wenhao_vla.inference_model import WenhaoNativeInferenceModel
from alpagym_g1_wenhao_vla.model import WenhaoPsiActorCritic
from alpagym_g1_wenhao_vla.normalization import WenhaoQ99Normalizer
from alpagym_g1_wenhao_vla.provenance import RUN_CONFIG_SHA256
import alpagym_g1_wenhao_vla.reference_adapter as reference_adapter_module
from alpagym_g1_wenhao_vla.reference_adapter import (
    H50_FRAME_COUNT,
    WenhaoH50ReferenceAdapter,
    _load_source_module,
)

_HUMANOID_REPO = Path("/test/humanoid-repo")
_HUMANOID_PLANNER_SHA256 = "a" * 64
_HUMANOID_MOTION_REFERENCE_SHA256 = "b" * 64


@dataclass(frozen=True)
class _FakeActionChunk:
    actions: np.ndarray
    chunk_base_quat_wxyz: np.ndarray
    chunk_base_xy: np.ndarray


@dataclass(frozen=True)
class _FakeVelocitySeed:
    joint_position: np.ndarray

    @classmethod
    def episode_reset_hold(cls, joint_position: object) -> _FakeVelocitySeed:
        return cls(np.asarray(joint_position, dtype=np.float64))

    @classmethod
    def previous_policy_target(cls, joint_position: object) -> _FakeVelocitySeed:
        return cls(np.asarray(joint_position, dtype=np.float64))


@dataclass(frozen=True)
class _FakeMotionFrame:
    time_s: float
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    root_position: np.ndarray
    root_quaternion_wxyz: np.ndarray


@dataclass(frozen=True)
class _FakeMotionReference:
    frames: tuple[_FakeMotionFrame, ...]
    joint_names: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.frames)

    def uniform_period_s(self) -> float | None:
        if len(self.frames) < 2:
            return None
        periods = np.diff([frame.time_s for frame in self.frames])
        return float(periods[0]) if np.allclose(periods, periods[0]) else None

    @property
    def sha256(self) -> str:
        digest = hashlib.sha256()
        for frame in self.frames:
            digest.update(np.float64(frame.time_s).tobytes())
            for value in (
                frame.joint_position,
                frame.joint_velocity,
                frame.root_position,
                frame.root_quaternion_wxyz,
            ):
                digest.update(np.asarray(value, dtype=np.float64).tobytes())
        return digest.hexdigest()


def _fake_convert_wenhao_chunk(
    chunk: _FakeActionChunk,
    *,
    source_cursor: int,
    velocity_seed: _FakeVelocitySeed,
) -> tuple[SimpleNamespace, ...]:
    """Mirror the exact target clock needed by adapter unit tests."""
    assert source_cursor == 0
    actions = np.asarray(chunk.actions, dtype=np.float64)
    previous = np.asarray(velocity_seed.joint_position, dtype=np.float64)
    targets = []
    for target_index in range(50):
        phase = min(target_index * 3.0 / 5.0, 29.0)
        lower = int(math.floor(phase))
        upper = min(lower + 1, 29)
        fraction = phase - lower
        position = (
            actions[lower, :29] * (1.0 - fraction) + actions[upper, :29] * fraction
        )
        velocity_hz = 30.0 if target_index == 0 else 50.0
        velocity = (position - previous) * velocity_hz
        targets.append(
            SimpleNamespace(
                joint_position=position,
                joint_velocity=velocity,
                root_quaternion_wxyz=np.asarray(
                    chunk.chunk_base_quat_wxyz, dtype=np.float64
                ),
            )
        )
        previous = position
    return tuple(targets)


@pytest.fixture(autouse=True)
def _install_isolated_humanoid_support(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unit tests independent of any sibling workstation checkout."""
    planner = SimpleNamespace(
        WENHAO_G1_JOINT_NAMES=MOTION_REFERENCE_JOINT_NAMES,
        WENHAO_RTC_PREFIX_LENGTH_MAX=7,
        WenhaoServerActionChunk=_FakeActionChunk,
        WenhaoVelocitySeed=_FakeVelocitySeed,
        convert_wenhao_chunk_to_sonic_targets=_fake_convert_wenhao_chunk,
    )
    motion = SimpleNamespace(
        MotionFrame=_FakeMotionFrame,
        MotionReference=_FakeMotionReference,
    )
    support = reference_adapter_module._HumanoidModules(
        planner=planner,
        motion=motion,
    )

    def _load_support(*args: object, **kwargs: object) -> object:
        del args, kwargs
        return support

    monkeypatch.setattr(
        reference_adapter_module,
        "_load_humanoid_modules",
        _load_support,
    )


class _FakeVlm(torch.nn.Module):
    """Tiny frozen Qwen stand-in."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(64, 8)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed prompt ids."""
        return self.embedding(input_ids)


class _FakeActionHead(torch.nn.Module):
    """Tiny trainable flow velocity stand-in."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.05))

    def forward(
        self,
        *,
        hidden_states: None,
        timestep: torch.Tensor,
        joint_attention_kwargs: dict[str, torch.Tensor | None],
        vlm_attn_mask: torch.Tensor,
        return_dict: bool,
    ) -> SimpleNamespace:
        """Return one finite Psi-shaped velocity."""
        del hidden_states, vlm_attn_mask, return_dict
        latent = joint_attention_kwargs["action_hidden_embeds"]
        assert latent is not None
        time = timestep[:, None, None] if timestep.ndim == 1 else timestep[..., None]
        return SimpleNamespace(
            action=latent * self.scale + time.to(latent.dtype) / 100_000.0
        )


class _FakePsi(torch.nn.Module):
    """Expose native preprocessing and condition methods used by rollout."""

    def __init__(self) -> None:
        super().__init__()
        self.vlm_model = _FakeVlm()
        self.action_header = _FakeActionHead()
        self.seen_images: list[list[np.ndarray]] = []

    def _build_vlm_batch(
        self,
        observations: list[list[Image.Image]],
        instructions: list[str],
    ) -> tuple[torch.Tensor, ...]:
        """Build deterministic tensors while retaining input chronology."""
        assert len(observations) == len(instructions) == 1
        self.seen_images.append([np.asarray(image).copy() for image in observations[0]])
        count = len(observations[0])
        ids = torch.arange(1, count + 3, dtype=torch.int64)[None]
        mask = torch.ones_like(ids)
        pixels = torch.arange(count * 4, dtype=torch.float32).reshape(count, 4)
        grid = torch.ones((count, 3), dtype=torch.int64)
        if count == 1:
            return ids, mask, pixels, grid, None, None
        return (
            ids,
            mask,
            pixels,
            grid,
            grid.clone(),
            torch.tensor([2] * (count - 1) + [1], dtype=torch.int64),
        )

    def _vlm_hidden_states(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        effective_image_grid_thw: torch.Tensor | None,
        visual_pool_factors: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return frozen prompt embeddings."""
        del (
            attention_mask,
            pixel_values,
            image_grid_thw,
            effective_image_grid_thw,
            visual_pool_factors,
        )
        return self.vlm_model(input_ids)

    def _select_vlm_planner_hidden(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Keep every prompt token."""
        return hidden, attention_mask


class _Engine:
    """Minimal session-model lookup fake."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    def get_model_for_session(self, session_uuid: str) -> torch.nn.Module:
        """Return the fixed model after checking the session key."""
        assert session_uuid == "session"
        return self.model


def _model() -> WenhaoPsiActorCritic:
    psi = _FakePsi()
    psi.vlm_model.requires_grad_(False)
    schedule = WenhaoFlowSchedule(
        model_timesteps=torch.tensor(
            [1000, 889, 778, 667, 556, 445, 334, 223, 112, 1],
            dtype=torch.float32,
        ),
        sigmas=torch.tensor(
            [1, 0.889, 0.778, 0.667, 0.556, 0.445, 0.334, 0.223, 0.112, 0.001, 0],
            dtype=torch.float32,
        ),
    )
    normalizer = WenhaoQ99Normalizer(
        state_q01=torch.full((29,), -2.0),
        state_q99=torch.full((29,), 2.0),
        action_q01=torch.full((38,), -2.0),
        action_q99=torch.full((38,), 2.0),
    )
    return WenhaoPsiActorCritic(
        psi_model=psi,
        schedule=schedule,
        noise_level=0.4,
        normalizer=normalizer,
        vlm_hidden_dim=8,
        rtc_max_delay_exclusive=8,
        critic_hidden_sizes=(16,),
    ).eval()


def _png(color: tuple[int, int, int]) -> bytes:
    image = Image.new("RGB", (224, 140), color=color)
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    image.close()
    return stream.getvalue()


def _input(
    *,
    step: int,
    color: tuple[int, int, int],
    bootstrap: bool = False,
    active_reference_ids: tuple[int, ...] = (),
    timestamp_us: int | None = None,
) -> HumanoidPolicyInput:
    logical_timestamp_us = step * 500_000 if timestamp_us is None else timestamp_us
    qpos = torch.zeros(36, dtype=torch.float32)
    qpos[:3] = torch.tensor([0.1, -0.2, 0.8]) + step * 0.01
    qpos[3] = 1.0
    qpos[7:] = torch.linspace(-0.3, 0.4, 29) + step * 0.01
    frame = HumanoidCameraFrame(
        env_id=0,
        frame_start_us=logical_timestamp_us,
        frame_end_us=logical_timestamp_us,
        logical_id="d455_rgb",
        image_bytes=_png(color),
        render_timestamp_us=logical_timestamp_us,
        observation_decision_id=step + 1,
        render_qpos=tuple(float(value) for value in qpos),
        render_state_sha256="5" * 64,
        camera_contract_sha256="6" * 64,
        image_sha256="7" * 64,
        render_receipt_sha256=f"{step + 8:064x}",
    )
    return HumanoidPolicyInput(
        session_uuid="session",
        episode_id=17,
        step_index=step,
        timestamp_us=logical_timestamp_us,
        env_id=0,
        qpos=qpos,
        qvel=torch.zeros(35),
        observation=torch.zeros(1),
        scalars={},
        camera_frames=(frame,),
        feedback_trace=(
            None
            if step == 0
            else HumanoidRealizedFeedbackTrace(
                env_id=0,
                source_decision_id=step,
                ticks=tuple(
                    HumanoidRealizedControlTick(
                        control_tick_offset=tick_index + 1,
                        qpos=qpos.clone(),
                        qvel=torch.zeros(35),
                        observation=torch.zeros(1),
                        scalars={},
                        timestamp_us=(step - 1) * 500_000 + (tick_index + 1) * 20_000,
                        active_reference_id=reference_id,
                        reference_action_index=tick_index,
                        active_reference_sha256="8" * 64,
                        applied_reference_sha256="9" * 64,
                        root_z_alignment_offset_m=0.0,
                        reward=0.0,
                        terminated=False,
                        truncated=False,
                        metrics={},
                        control_episode_step=tick_index + 1,
                    )
                    for tick_index, reference_id in enumerate(active_reference_ids)
                ),
            )
        ),
        decision_id=step + 1,
        bootstrap_requested=bootstrap,
    )


def test_letterbox_preserves_native_pixels_and_black_padding() -> None:
    image = width_fit_letterbox_d455(_png((12, 34, 56)), image_format="png")
    array = image_array(image)
    image.close()

    assert not bool(array[:42].any())
    assert not bool(array[182:].any())
    assert np.all(array[42:182] == np.asarray([12, 34, 56], dtype=np.uint8))


def test_history_is_oldest_to_current_and_bats_is_frozen() -> None:
    history = WenhaoImageHistory(episode_index=17)
    colors = ((10, 0, 0), (20, 0, 0), (30, 0, 0))
    selected_colors: list[list[int]] = []
    for index, color in enumerate(colors):
        current = Image.new("RGB", (224, 224), color=color)
        selected = history.select_with_current(
            current,
            timestamp_us=index * 500_000,
            capture_receipt_sha256=f"receipt-{index}",
        )
        selected_colors.append(
            [int(np.asarray(image)[100, 100, 0]) for image in selected]
        )
        current.close()
        for image in selected:
            image.close()
    history.close()

    assert selected_colors == [[10], [10, 20], [10, 20, 30]]
    assert select_bats_history_indices(100, 17) == (
        0,
        9,
        11,
        20,
        32,
        35,
        39,
        53,
        61,
        64,
        66,
        67,
        75,
        76,
        78,
        81,
        82,
        83,
        85,
        90,
        91,
        92,
        93,
        95,
        96,
        97,
        98,
        99,
    )
    assert stable_episode_int("ascend") == 1_412_517_340


def test_history_defers_duplicate_current_capture_receipt() -> None:
    """A stale producer receipt never appears as both history and current."""
    history = WenhaoImageHistory(episode_index=stable_episode_int("ascend"))
    first = Image.new("RGB", (224, 224), color=(10, 0, 0))
    stale = Image.new("RGB", (224, 224), color=(10, 0, 0))
    fresh = Image.new("RGB", (224, 224), color=(30, 0, 0))
    selected_first = history.select_with_current(
        first, timestamp_us=0, capture_receipt_sha256="receipt-a"
    )
    selected_stale = history.select_with_current(
        stale, timestamp_us=500_000, capture_receipt_sha256="receipt-a"
    )
    selected_fresh = history.select_with_current(
        fresh, timestamp_us=1_000_000, capture_receipt_sha256="receipt-b"
    )
    try:
        assert len(selected_first) == 1
        assert len(selected_stale) == 1
        assert [int(np.asarray(image)[0, 0, 0]) for image in selected_fresh] == [10, 30]
    finally:
        first.close()
        stale.close()
        fresh.close()
        for selected in (selected_first, selected_stale, selected_fresh):
            for image in selected:
                image.close()
        history.close()


def _policy(
    model: WenhaoPsiActorCritic, *, seed: int = 41
) -> G1WenhaoVlaHumanoidPolicy:
    return G1WenhaoVlaHumanoidPolicy(
        _Engine(model),
        session_uuid="session",
        request=SimpleNamespace(random_seed=seed, scenario_id="ascend"),
        image_format="png",
        expected_schedule_sha256=model.schedule.sha256,
        humanoid_repo_path=_HUMANOID_REPO,
        humanoid_planner_sha256=_HUMANOID_PLANNER_SHA256,
        humanoid_motion_reference_sha256=_HUMANOID_MOTION_REFERENCE_SHA256,
        language_instruction="walk up the stairs",
    )


def test_reference_adapter_preserves_policy_target_zero_and_exact_target_clock() -> (
    None
):
    rows = np.zeros((30, 38), dtype=np.float32)
    rows[:, :29] = np.arange(30, dtype=np.float32)[:, None] * 0.01
    rows[:, 29:35] = np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32)
    qpos = np.zeros(36, dtype=np.float32)
    qpos[:3] = [0.1, -0.2, 0.8]
    qpos[3] = 1.0
    qpos[7:] = -0.1
    qvel = np.zeros(35, dtype=np.float32)

    adapter = WenhaoH50ReferenceAdapter(
        _HUMANOID_REPO,
        planner_sha256=_HUMANOID_PLANNER_SHA256,
        motion_reference_sha256=_HUMANOID_MOTION_REFERENCE_SHA256,
    )
    assert adapter.rtc_max_delay_exclusive == 8
    reference = adapter.build(rows, qpos=qpos, qvel=qvel, timestamp_us=500_000)

    assert len(reference.frames) == H50_FRAME_COUNT
    np.testing.assert_array_equal(reference.frames[0].joint_position, rows[0, :29])
    np.testing.assert_allclose(
        reference.frames[0].joint_velocity,
        (rows[0, :29] - qpos[7:]) / (1.0 / 30.0),
        rtol=0,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        reference.frames[1].joint_position,
        np.full(29, 0.006),
        rtol=0,
        atol=1.0e-7,
    )
    for previous, current in zip(reference.frames[:-1], reference.frames[1:]):
        np.testing.assert_allclose(
            current.joint_velocity,
            (current.joint_position - previous.joint_position) / 0.02,
            rtol=0,
            atol=1.0e-6,
        )
    assert [round(frame.time_s * 1_000_000) for frame in reference.frames] == [
        500_000 + index * 20_000 for index in range(50)
    ]


def test_dynamic_humanoid_module_is_content_pinned_and_content_cached(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "support.py"
    source_v1 = b"VALUE = 1\n"
    source_path.write_bytes(source_v1)
    digest_v1 = hashlib.sha256(source_v1).hexdigest()

    first = _load_source_module(
        source_path, role="test_support", expected_sha256=digest_v1
    )
    again = _load_source_module(
        source_path, role="test_support", expected_sha256=digest_v1
    )
    assert first is again
    assert first.VALUE == 1

    source_v2 = b"VALUE = 2\n"
    source_path.write_bytes(source_v2)
    digest_v2 = hashlib.sha256(source_v2).hexdigest()
    second = _load_source_module(
        source_path, role="test_support", expected_sha256=digest_v2
    )
    assert second is not first
    assert second.VALUE == 2
    with pytest.raises(ValueError, match="content SHA256 changed"):
        _load_source_module(source_path, role="test_support", expected_sha256=digest_v1)


def test_flow_sample_emits_h50_and_preserves_exact_raw_action_replay() -> None:
    model = _model()
    assert model.noise_level == 0.4
    policy = _policy(model)
    policy_input = _input(step=0, color=(4, 5, 6))
    output = policy.step((policy_input,))[0]
    assert output.motion_reference is not None
    assert output.replay_data is not None
    assert output.motion_reference.reference_id == (1 << 32) | 1
    assert output.motion_reference.source_decision_id == policy_input.decision_id
    assert len(output.motion_reference.frames) == 50
    assert output.replay_data.old_logprob is not None
    payload = output.replay_data.payload
    assert payload["run_config_sha256"] == RUN_CONFIG_SHA256
    assert "inference_manifest_sha256" not in payload
    torch.testing.assert_close(
        payload["physical_states"][0],
        torch.tensor(policy_input.camera_frames[0].policy_joint_position),
    )
    torch.testing.assert_close(
        payload["old_element_logprobs"].sum(), output.replay_data.old_logprob
    )
    assert payload["flow_noise_level"] == 0.4
    assert payload["flow_ignore_last"] is True
    torch.testing.assert_close(
        payload["effective_image_grid_thw"], payload["image_grid_thw"]
    )
    torch.testing.assert_close(
        payload["visual_pool_factors"], torch.ones(1, dtype=torch.int64)
    )
    torch.testing.assert_close(
        payload["denormalized_wire_actions"],
        payload["humanoid"]["wenhao_raw_action_rows"],
    )
    assert payload["latent_chain"].shape == (11, 30, 38)
    assert not bool(payload["rtc_prefix_mask"].any())
    assert not bool(payload["rtc_prefix_normalized_actions"].any())
    assert set(payload["humanoid"]) == {"wenhao_raw_action_rows"}
    assert "transition" not in payload
    for value in payload.values():
        if isinstance(value, torch.Tensor):
            assert value.device.type == "cpu"
    policy.close()


def test_second_sample_hard_inpaints_rebased_predecessor_suffix() -> None:
    """RTC freezes predicted overlap rows and gives them zero Flow density."""
    model = _model()
    policy = _policy(model, seed=17)
    first_input = _input(step=0, color=(4, 5, 6))
    first = policy.step((first_input,))[0]
    assert first.motion_reference is not None
    assert first.replay_data is not None
    first_chunk = first.replay_data.payload["clipped_normalized_actions"]
    first_lane = policy._lanes[0]
    assert first_lane.rtc_state is not None
    previous = first_lane.rtc_state.previous_chunk
    assert previous is not None
    assert previous.reference_id == first.motion_reference.reference_id
    assert previous.source_timestamp_us == first_input.timestamp_us
    torch.testing.assert_close(previous.clipped_normalized_actions, first_chunk)
    torch.testing.assert_close(
        previous.base_quaternion_wxyz,
        first_input.qpos[3:7],
        check_dtype=False,
    )
    torch.testing.assert_close(
        previous.base_xy,
        first_input.qpos[:2],
        check_dtype=False,
    )

    # Nine 50 Hz predecessor ticks conservatively ceil to six 30 Hz source rows.
    predecessor_id = first.motion_reference.reference_id - 1
    second_input = _input(
        step=1,
        color=(7, 8, 9),
        active_reference_ids=(predecessor_id,) * 9
        + (first.motion_reference.reference_id,) * 16,
    )
    second = policy.step((second_input,))[0]
    assert second.replay_data is not None
    payload = second.replay_data.payload
    mask = payload["rtc_prefix_mask"]
    assert torch.equal(
        mask,
        torch.tensor([True] * 6 + [False] * 24, dtype=torch.bool),
    )
    expected = _rebase_normalized_chunk_relative_dims(
        first_chunk[15:],
        previous_quaternion_wxyz=first_input.qpos[3:7],
        previous_xy=first_input.qpos[:2],
        current_quaternion_wxyz=second_input.qpos[3:7],
        current_xy=second_input.qpos[:2],
        normalizer=model.normalizer,
    )
    torch.testing.assert_close(
        payload["rtc_prefix_normalized_actions"][:6], expected[:6]
    )
    torch.testing.assert_close(payload["clipped_normalized_actions"][:6], expected[:6])
    element_density = payload["old_element_logprobs"].reshape(30, 38)
    assert torch.equal(element_density[:6], torch.zeros_like(element_density[:6]))
    replay_result = model.forward(
        input_ids=payload["input_ids"][None],
        attention_mask=payload["attention_mask"][None],
        pixel_values=payload["pixel_values"],
        image_grid_thw=payload["image_grid_thw"],
        sequence_lengths=payload["attention_mask"].to(torch.int64).sum()[None],
        image_counts=torch.tensor([payload["image_grid_thw"].shape[0]]),
        image_offsets=torch.tensor([0, payload["image_grid_thw"].shape[0]]),
        image_patch_counts=payload["image_grid_thw"].to(torch.int64).prod(dim=1),
        patch_counts=torch.tensor([payload["pixel_values"].shape[0]]),
        patch_offsets=torch.tensor([0, payload["pixel_values"].shape[0]]),
        physical_states=payload["physical_states"][None],
        latent_chain=payload["latent_chain"][None],
        denoise_indices=payload["denoise_index"].reshape(1),
        flow_schedule_sha256=payload["schedule_sha256"],
        clipped_normalized_actions=payload["clipped_normalized_actions"][None],
        denormalized_wire_actions=payload["denormalized_wire_actions"][None],
        rtc_prefix_normalized_actions=payload["rtc_prefix_normalized_actions"][None],
        rtc_prefix_mask=payload["rtc_prefix_mask"][None],
        effective_image_grid_thw=payload.get("effective_image_grid_thw"),
        visual_pool_factors=payload.get("visual_pool_factors"),
    )
    new_element_density = replay_result["element_log_probs"]
    assert new_element_density is not None
    new_element_density = new_element_density.reshape(30, 38)
    assert torch.equal(
        new_element_density[:6], torch.zeros_like(new_element_density[:6])
    )
    assert second.model_extra is not None
    assert second.model_extra["wenhao_rtc_delay_rows"] == 6
    assert second.model_extra["wenhao_rtc_source_cursor_h50"] == 25
    assert second.model_extra["wenhao_rtc_source_start_row_h30"] == 15
    first_wire = first.replay_data.payload["denormalized_wire_actions"]
    expected_velocity = (first_wire[15, :29] - first_wire[14, :29]) / (1.0 / 30.0)
    assert second.motion_reference is not None
    torch.testing.assert_close(
        second.motion_reference.frames[0].joint_velocity,
        expected_velocity,
        rtol=0,
        atol=2.0e-5,
    )
    policy.close()


def test_late_cursor_28_uses_row_17_suffix_and_row_16_seam_seed() -> None:
    """A late launch never freezes rows already consumed on the H50 clock."""
    model = _model()
    policy = _policy(model, seed=19)
    first_input = _input(step=0, color=(1, 3, 5))
    first = policy.step((first_input,))[0]
    assert first.motion_reference is not None
    assert first.replay_data is not None
    first_payload = first.replay_data.payload
    second_input = _input(
        step=1,
        color=(2, 4, 6),
        timestamp_us=28 * 20_000,
        active_reference_ids=(first.motion_reference.reference_id,) * 28,
    )
    second = policy.step((second_input,))[0]
    assert second.motion_reference is not None
    assert second.replay_data is not None
    second_payload = second.replay_data.payload
    assert second.model_extra is not None
    assert second.model_extra["wenhao_rtc_source_cursor_h50"] == 28
    assert second.model_extra["wenhao_rtc_source_start_row_h30"] == 17
    assert int(second_payload["rtc_prefix_mask"].sum().item()) == 6

    expected_suffix = _rebase_normalized_chunk_relative_dims(
        first_payload["clipped_normalized_actions"][17:],
        previous_quaternion_wxyz=first_input.qpos[3:7],
        previous_xy=first_input.qpos[:2],
        current_quaternion_wxyz=second_input.qpos[3:7],
        current_xy=second_input.qpos[:2],
        normalizer=model.normalizer,
    )
    rtc_buffer = second_payload["rtc_prefix_normalized_actions"]
    torch.testing.assert_close(rtc_buffer[:13], expected_suffix)
    assert torch.equal(rtc_buffer[13:], torch.zeros_like(rtc_buffer[13:]))
    first_wire = first_payload["denormalized_wire_actions"]
    expected_velocity = (first_wire[17, :29] - first_wire[16, :29]) / (1.0 / 30.0)
    torch.testing.assert_close(
        second.motion_reference.frames[0].joint_velocity,
        expected_velocity,
        rtol=0,
        atol=2.0e-5,
    )
    policy.close()


def test_exhausted_cursor_50_has_no_prefix_and_uses_terminal_seam_seed() -> None:
    """Once H50 is exhausted, RTC predicts fresh rows from terminal row 29."""
    model = _model()
    policy = _policy(model, seed=29)
    first = policy.step((_input(step=0, color=(2, 3, 4)),))[0]
    assert first.motion_reference is not None
    assert first.replay_data is not None
    second = policy.step(
        (
            _input(
                step=1,
                color=(5, 6, 7),
                timestamp_us=50 * 20_000,
                active_reference_ids=(first.motion_reference.reference_id,) * 50,
            ),
        )
    )[0]
    assert second.motion_reference is not None
    assert second.replay_data is not None
    assert second.model_extra is not None
    assert second.model_extra["wenhao_rtc_source_cursor_h50"] == 50
    assert second.model_extra["wenhao_rtc_source_start_row_h30"] == 30
    rtc_buffer = second.replay_data.payload["rtc_prefix_normalized_actions"]
    rtc_mask = second.replay_data.payload["rtc_prefix_mask"]
    assert torch.equal(rtc_buffer, torch.zeros_like(rtc_buffer))
    assert not bool(rtc_mask.any())
    first_wire = first.replay_data.payload["denormalized_wire_actions"]
    second_wire = second.replay_data.payload["denormalized_wire_actions"]
    expected_velocity = (second_wire[0, :29] - first_wire[29, :29]) / (1.0 / 30.0)
    torch.testing.assert_close(
        second.motion_reference.frames[0].joint_velocity,
        expected_velocity,
        rtol=0,
        atol=2.0e-5,
    )
    policy.close()


@pytest.mark.parametrize(
    ("timestamp_us", "message"),
    ((510_000, "exact 20 ms"), (0, "strictly increasing")),
)
def test_rtc_source_time_must_be_monotonic_on_h50_grid(
    timestamp_us: int,
    message: str,
) -> None:
    model = _model()
    policy = _policy(model, seed=31)
    first = policy.step((_input(step=0, color=(3, 2, 1)),))[0]
    assert first.motion_reference is not None
    lane = policy._lanes[0]
    state_before = lane.rtc_state
    sequence_before = lane.next_reference_sequence
    with pytest.raises(ValueError, match=message):
        policy.step(
            (
                _input(
                    step=1,
                    color=(6, 5, 4),
                    timestamp_us=timestamp_us,
                    active_reference_ids=(first.motion_reference.reference_id,),
                ),
            )
        )
    assert lane.rtc_state is state_before
    assert lane.next_reference_sequence == sequence_before
    policy.close()


def test_rtc_delay_predictor_clamps_to_real_previous_suffix() -> None:
    """A long observed overlap cannot freeze zero-padded or out-of-bound rows."""
    model = _model()
    policy = _policy(model, seed=23)
    first = policy.step((_input(step=0, color=(1, 2, 3)),))[0]
    assert first.motion_reference is not None
    predecessor_id = first.motion_reference.reference_id - 1
    second = policy.step(
        (
            _input(
                step=1,
                color=(4, 5, 6),
                # Forty predecessor ticks plus one installed-plan tick put
                # this observation at H50 cursor 41 / H30 row 25.  The delay
                # estimate is 24, but only five real source rows remain.
                timestamp_us=41 * 20_000,
                active_reference_ids=(predecessor_id,) * 40
                + (first.motion_reference.reference_id,),
            ),
        )
    )[0]
    assert second.replay_data is not None
    assert int(second.replay_data.payload["rtc_prefix_mask"].sum().item()) == 5
    lane = policy._lanes[0]
    assert lane.rtc_state is not None
    assert lane.rtc_state.delay_rows == (6, 24)
    assert second.model_extra is not None
    assert second.model_extra["wenhao_rtc_delay_rows"] == 5
    assert second.model_extra["wenhao_rtc_source_cursor_h50"] == 41
    assert second.model_extra["wenhao_rtc_source_start_row_h30"] == 25
    policy.close()


def test_rtc_delay_predictor_clamps_to_attested_checkpoint_max_delay() -> None:
    """A historical observed delay of 24 still freezes at most rows 0..6."""

    model = _model()
    policy = _policy(model, seed=37)
    first = policy.step((_input(step=0, color=(1, 2, 3)),))[0]
    assert first.motion_reference is not None
    predecessor_id = first.motion_reference.reference_id - 1
    late_second = policy.step(
        (
            _input(
                step=1,
                color=(4, 5, 6),
                # Forty predecessor ticks conservatively map to 24 H30 rows;
                # one installed tick keeps the transition lifecycle realistic.
                timestamp_us=820_000,
                active_reference_ids=(predecessor_id,) * 40
                + (first.motion_reference.reference_id,),
            ),
        )
    )[0]
    assert late_second.motion_reference is not None
    # The late source has only five rows left, so first prove delay 24 entered
    # the six-sample history before testing that history on a regular H50 launch.
    assert late_second.replay_data is not None
    assert int(late_second.replay_data.payload["rtc_prefix_mask"].sum()) == 5

    third = policy.step(
        (
            _input(
                step=2,
                color=(7, 8, 9),
                timestamp_us=1_320_000,
                active_reference_ids=(late_second.motion_reference.reference_id,) * 25,
            ),
        )
    )[0]
    assert third.replay_data is not None
    mask = third.replay_data.payload["rtc_prefix_mask"]
    assert torch.equal(
        mask,
        torch.tensor([True] * 7 + [False] * 23, dtype=torch.bool),
    )
    lane = policy._lanes[0]
    assert lane.rtc_state is not None
    assert lane.rtc_state.delay_rows == (6, 24, 0)
    assert third.model_extra is not None
    assert third.model_extra["wenhao_rtc_delay_rows"] == 7
    policy.close()


def test_yaw_only_rtc_rebase_preserves_world_target() -> None:
    """Relative action pose is re-expressed under a changed chunk base."""
    normalizer = WenhaoQ99Normalizer(
        state_q01=torch.full((29,), -2.0),
        state_q99=torch.full((29,), 2.0),
        action_q01=torch.full((38,), -2.0),
        action_q99=torch.full((38,), 2.0),
    )
    physical = torch.zeros((1, 38), dtype=torch.float32)
    physical[0, 29:35] = torch.tensor([1, 0, 0, 0, 1, 0])
    physical[0, 35:37] = torch.tensor([1, 0])
    normalized = normalizer.normalize_action(physical)
    half_yaw = np.pi / 4.0
    rebased = _rebase_normalized_chunk_relative_dims(
        normalized,
        previous_quaternion_wxyz=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        previous_xy=torch.tensor([0.0, 0.0]),
        current_quaternion_wxyz=torch.tensor(
            [np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)]
        ),
        current_xy=torch.tensor([0.0, 0.0]),
        normalizer=normalizer,
    )
    rebased_physical = normalizer.denormalize_action(rebased)
    torch.testing.assert_close(rebased[:, :29], normalized[:, :29])
    torch.testing.assert_close(
        rebased_physical[0, 29:38],
        torch.tensor([0, -1, 0, 1, 0, 0, 0, -1, -np.pi / 2]),
        rtol=0,
        atol=1.0e-5,
    )


def test_finalize_reuses_observation_path_and_returns_only_bootstrap_value() -> None:
    model = _model()
    policy = _policy(model, seed=9)
    first = policy.step((_input(step=0, color=(1, 0, 0)),))[0]
    final = policy.step(
        (_input(step=1, color=(2, 0, 0), bootstrap=True),),
        sample_actions=False,
    )[0]
    assert first.motion_reference is not None
    assert final.value is not None
    assert final.motion_reference is None
    assert final.logprob is None
    assert final.replay_data is None
    psi = model.psi_model
    assert isinstance(psi, _FakePsi)
    assert len(psi.seen_images) == 2
    assert len(psi.seen_images[-1]) == 2
    policy.close()


def test_inference_only_sample_needs_no_flow_trace_logprob_or_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    rows = torch.zeros((1, 30, 38), dtype=torch.float32)
    rows[..., 29] = 1.0
    rows[..., 33] = 1.0
    monkeypatch.setattr(
        model,
        "sample_actions",
        lambda **kwargs: SimpleNamespace(denormalized_wire=rows),
    )
    policy = _policy(model)

    output = policy.step((_input(step=0, color=(3, 4, 5)),))[0]

    assert output.motion_reference is not None
    assert output.logprob is None
    assert output.value is None
    assert output.replay_data is not None
    assert output.replay_data.old_logprob is None
    torch.testing.assert_close(
        output.replay_data.payload["denormalized_wire_actions"], rows[0]
    )
    assert "latent_chain" not in output.replay_data.payload
    policy.close()


def test_factory_and_inference_adapter_fail_closed_on_wrong_contracts() -> None:
    model = _model()
    adapter = WenhaoNativeInferenceModel(model)
    assert adapter.get_model() is model
    with pytest.raises(TypeError, match="expected Wenhao"):
        adapter.set_model(torch.nn.Linear(2, 2))

    config = SimpleNamespace(
        policy=SimpleNamespace(
            model=SimpleNamespace(
                bundle_config={
                    "expected_schedule_sha256": model.schedule.sha256,
                    "humanoid_repo_path": str(_HUMANOID_REPO),
                    "humanoid_wenhao_planner_sha256": _HUMANOID_PLANNER_SHA256,
                    "humanoid_motion_reference_sha256": (
                        _HUMANOID_MOTION_REFERENCE_SHA256
                    ),
                    "language_instruction": "walk up the stairs",
                    "camera_logical_id": "d455_rgb",
                    "camera_image_format": "png",
                    "camera_contract_sha256": "6" * 64,
                }
            )
        )
    )
    factory = build_humanoid_policy_factory(config, _Engine(model))
    request = SimpleNamespace(
        execution_mode=HUMANOID_EXECUTION_MODE_MOTION_REFERENCE,
        action_schema="g1_motion_reference_29d_50hz_h50.v1",
        action_size=0,
        joint_names=MOTION_REFERENCE_JOINT_NAMES,
        random_seed=0,
        attempt_id="attempt",
        scene_id="hq_stairs",
        scenario_id="ascend",
        reference_spec=SimpleNamespace(
            schema="g1_motion_reference_29d_50hz_h50.v1",
            frame_count=50,
            sample_period_us=20_000,
            control_ticks_per_policy_step=25,
            joint_names=MOTION_REFERENCE_JOINT_NAMES,
        ),
        policy_camera_spec=SimpleNamespace(
            schema="humanoid_policy_camera_rgb_qpos.v1",
            logical_id="wrong_camera",
            width=224,
            height=140,
            image_format="png",
            contract_sha256="6" * 64,
        ),
    )
    with pytest.raises(ValueError, match="D455 camera contract changed"):
        factory("session", request)
