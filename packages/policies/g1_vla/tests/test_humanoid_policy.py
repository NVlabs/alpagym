"""Closed-loop contract tests for the H50 VLA rollout policy."""

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

from alpagym_g1_vla.bundle import (
    FLOW_SDE_TRAINING,
    NATIVE_ODE_QUALIFICATION,
    QUALIFICATION_REPLAY_SCHEMA,
)
from alpagym_g1_vla.flow import VlaFlowSchedule
from alpagym_g1_vla.history import (
    VlaImageHistory,
    decode_native_d435,
    image_array,
    select_bats_history_indices,
    stable_episode_int,
)
from alpagym_g1_vla.humanoid_policy import (
    G1VlaHumanoidPolicy,
    _rebase_normalized_chunk_relative_dims,
    build_humanoid_policy_factory,
)
from alpagym_g1_vla.inference_model import VlaNativeInferenceModel
from alpagym_g1_vla.model import VlaPsiActorCritic
from alpagym_g1_vla.normalization import VlaQ99Normalizer
from alpagym_g1_vla.provenance import (
    STAIRSBLOCKS_VLA_BUNDLE_PROFILE,
)
import alpagym_g1_vla.reference_adapter as reference_adapter_module
from alpagym_g1_vla.reference_adapter import (
    H50_FRAME_COUNT,
    VlaMotionReferenceAdapter,
    _load_source_module,
)

_HUMANOID_REPO = Path("/test/humanoid-repo")
_HUMANOID_VLA_REFERENCE_ADAPTER_SHA256 = "a" * 64
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


def _fake_convert_vla_chunk(
    chunk: _FakeActionChunk,
    *,
    source_row_cursor: int,
    velocity_seed: _FakeVelocitySeed,
) -> tuple[SimpleNamespace, ...]:
    """Mirror the exact target clock needed by adapter unit tests."""
    assert source_row_cursor == 0
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
                root_xy=np.asarray(chunk.chunk_base_xy, dtype=np.float64)
                + np.asarray([phase * 0.01, -phase * 0.005]),
                root_quaternion_wxyz=np.asarray(
                    chunk.chunk_base_quat_wxyz, dtype=np.float64
                ),
            )
        )
        previous = position
    return tuple(targets)


def _fake_full_rotation_local_xy_context(
    chunk: _FakeActionChunk,
    *,
    source_row_cursor: int,
) -> SimpleNamespace:
    """Mirror the H50 local-XY side channel from the isolated fake chunk."""
    assert source_row_cursor == 0
    actions = np.asarray(chunk.actions, dtype=np.float64)
    local_xy = []
    for target_index in range(50):
        phase = min(target_index * 3.0 / 5.0, 29.0)
        lower = int(math.floor(phase))
        upper = min(lower + 1, 29)
        fraction = phase - lower
        local_xy.append(
            actions[lower, 35:37] * (1.0 - fraction) + actions[upper, 35:37] * fraction
        )
    local_xy_from_frame_zero = np.asarray(local_xy) - local_xy[0]
    local_xy_from_frame_zero[0] = 0.0
    return SimpleNamespace(
        schema="full_pelvis_rotation_local_xy_completed_z/v1",
        chunk_base_quat_wxyz=np.asarray(chunk.chunk_base_quat_wxyz),
        local_xy_from_frame_zero=local_xy_from_frame_zero,
    )


@pytest.fixture(autouse=True)
def _install_isolated_humanoid_support(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unit tests independent of any sibling workstation checkout."""
    reference_adapter = SimpleNamespace(
        VLA_G1_JOINT_NAMES=MOTION_REFERENCE_JOINT_NAMES,
        VLA_RTC_PREFIX_LENGTH_MAX=7,
        VlaServerActionChunk=_FakeActionChunk,
        VlaVelocitySeed=_FakeVelocitySeed,
        convert_vla_chunk_to_reference_targets=_fake_convert_vla_chunk,
        convert_vla_chunk_to_full_rotation_local_xy_context=(
            _fake_full_rotation_local_xy_context
        ),
    )
    motion = SimpleNamespace(
        MotionFrame=_FakeMotionFrame,
        MotionReference=_FakeMotionReference,
    )
    support = reference_adapter_module._HumanoidModules(
        reference_adapter=reference_adapter,
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


class _FakeResizeTransform:
    interpolation = SimpleNamespace(name="NEAREST")

    def __call__(self, image: Image.Image) -> Image.Image:
        return image.resize((224, 224), resample=Image.Resampling.NEAREST)


class _FakeCenterCropTransform:
    def __call__(self, image: Image.Image) -> Image.Image:
        return image.crop((0, 0, 224, 224))


class _FakeResizeSpec:
    size = (224, 224)

    def __call__(self) -> _FakeResizeTransform:
        return _FakeResizeTransform()


class _FakeCenterCropSpec:
    size = (224, 224)

    def __call__(self) -> _FakeCenterCropTransform:
        return _FakeCenterCropTransform()


class _FakeEvalTransform:
    adaptive_resize = False
    compress_history_visual_tokens = False
    resize = _FakeResizeSpec()
    center_crop = _FakeCenterCropSpec()


class _FakePsi(torch.nn.Module):
    """Expose native preprocessing and condition methods used by rollout."""

    def __init__(self) -> None:
        super().__init__()
        self.vlm_model = _FakeVlm()
        self.action_header = _FakeActionHead()
        self.seen_images: list[list[np.ndarray]] = []
        self.model_transform = _FakeEvalTransform()

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


class _FakePooledTransform(_FakeEvalTransform):
    """Stand in for the checkpoint's training-time visual pooling transform."""

    compress_history_visual_tokens = True
    history_visual_pool_factor = 2

    def __init__(self) -> None:
        self.seen_pool_factors: list[list[int]] = []

    def build_qwenvl_inputs(
        self,
        processor: object,
        images: list[Image.Image],
        instruction: str,
        *,
        image_pool_factors: list[int],
    ) -> dict[str, torch.Tensor]:
        del processor, instruction
        self.seen_pool_factors.append(list(image_pool_factors))
        count = len(images)
        grid = torch.ones((count, 3), dtype=torch.int64)
        return {
            "input_ids": torch.arange(1, count + 3, dtype=torch.int64)[None],
            "attention_mask": torch.ones((1, count + 2), dtype=torch.int64),
            "pixel_values": torch.arange(count * 4, dtype=torch.float32).reshape(
                count, 4
            ),
            "image_grid_thw": grid,
            "effective_image_grid_thw": grid.clone(),
            "visual_pool_factors": torch.tensor(image_pool_factors, dtype=torch.int64),
        }


class _FakeFourTuplePsi(_FakePsi):
    """Mirror ckpt_1500's pre-fix four-tensor serving helper."""

    def __init__(self) -> None:
        super().__init__()
        self.model_transform = _FakePooledTransform()
        self.vlm_processor = object()
        self.device = torch.device("cpu")

    def _build_vlm_batch(
        self,
        observations: list[list[Image.Image]],
        instructions: list[str],
    ) -> tuple[torch.Tensor, ...]:
        built = super()._build_vlm_batch(observations, instructions)
        return built[:4]


class _Engine:
    """Minimal session-model lookup fake."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    def get_model_for_session(self, session_uuid: str) -> torch.nn.Module:
        """Return the fixed model after checking the session key."""
        assert session_uuid == "session"
        return self.model


def _model(psi: _FakePsi | None = None) -> VlaPsiActorCritic:
    psi = _FakePsi() if psi is None else psi
    psi.vlm_model.requires_grad_(False)
    schedule = VlaFlowSchedule(
        model_timesteps=torch.tensor(
            [1000, 889, 778, 667, 556, 445, 334, 223, 112, 1],
            dtype=torch.float32,
        ),
        sigmas=torch.tensor(
            [1, 0.889, 0.778, 0.667, 0.556, 0.445, 0.334, 0.223, 0.112, 0.001, 0],
            dtype=torch.float32,
        ),
    )
    qualification_schedule = VlaFlowSchedule(
        model_timesteps=schedule.model_timesteps.clone(),
        sigmas=schedule.sigmas.clone(),
    )
    normalizer = VlaQ99Normalizer(
        state_q01=torch.full((29,), -2.0),
        state_q99=torch.full((29,), 2.0),
        action_q01=torch.full((38,), -2.0),
        action_q99=torch.full((38,), 2.0),
    )
    return VlaPsiActorCritic(
        psi_model=psi,
        schedule=schedule,
        qualification_schedule=qualification_schedule,
        qualification_clip_normalized_actions=False,
        noise_level=0.4,
        normalizer=normalizer,
        vlm_hidden_dim=8,
        rtc_max_delay_exclusive=8,
        critic_hidden_sizes=(16,),
    ).eval()


def _jpeg(
    color: tuple[int, int, int],
    *,
    size: tuple[int, int] = (640, 480),
    quality: int = 95,
) -> bytes:
    image = Image.new("RGB", size, color=color)
    stream = io.BytesIO()
    image.save(stream, format="JPEG", quality=quality)
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
        logical_id="d435_rgb",
        image_bytes=_jpeg(color),
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


def test_native_d435_decoder_preserves_full_raster_without_letterbox() -> None:
    image = decode_native_d435(_jpeg((100, 100, 100)), image_format="jpeg")
    array = image_array(image)
    image.close()

    assert array.shape == (480, 640, 3)
    assert np.all(array == np.asarray([100, 100, 100], dtype=np.uint8))


def test_native_d435_decoder_rejects_legacy_d455_raster() -> None:
    with pytest.raises(ValueError, match="exactly 640x480"):
        decode_native_d435(
            _jpeg((12, 34, 56), size=(224, 140)),
            image_format="jpeg",
        )


def test_native_d435_decoder_rejects_non_training_jpeg_quality() -> None:
    with pytest.raises(ValueError, match="quality=95"):
        decode_native_d435(
            _jpeg((100, 100, 100), quality=90),
            image_format="jpeg",
        )


def test_checkpoint_eval_geometry_is_direct_nearest_resize_without_padding() -> None:
    x = np.arange(640, dtype=np.uint16)[None, :]
    y = np.arange(480, dtype=np.uint16)[:, None]
    source_array = np.empty((480, 640, 3), dtype=np.uint8)
    source_array[..., 0] = x % 256
    source_array[..., 1] = y % 256
    source_array[..., 2] = (x + y) % 256
    source = Image.fromarray(source_array, mode="RGB")
    expected = source.resize((224, 224), resample=Image.Resampling.NEAREST)

    transformed = G1VlaHumanoidPolicy._apply_checkpoint_eval_image_transform(
        SimpleNamespace(model_transform=_FakeEvalTransform()),
        (source,),
    )
    assert len(transformed) == 1
    np.testing.assert_array_equal(np.asarray(transformed[0]), np.asarray(expected))

    transformed[0].close()
    expected.close()
    source.close()


def test_checkpoint_eval_geometry_rejects_adaptive_resize() -> None:
    transform = _FakeEvalTransform()
    transform.adaptive_resize = True
    source = Image.new("RGB", (640, 480))
    with pytest.raises(ValueError, match="adaptive"):
        G1VlaHumanoidPolicy._apply_checkpoint_eval_image_transform(
            SimpleNamespace(model_transform=transform),
            (source,),
        )
    source.close()


def test_checkpoint_eval_geometry_rejects_wrong_resize_size() -> None:
    transform = _FakeEvalTransform()
    transform.resize = SimpleNamespace(size=(224, 168))
    source = Image.new("RGB", (640, 480))
    with pytest.raises(ValueError, match="resize must be exactly 224x224"):
        G1VlaHumanoidPolicy._apply_checkpoint_eval_image_transform(
            SimpleNamespace(model_transform=transform),
            (source,),
        )
    source.close()


def test_checkpoint_eval_geometry_rejects_wrong_interpolation() -> None:
    resize = _FakeResizeTransform()
    resize.interpolation = SimpleNamespace(name="BILINEAR")

    class _BadResizeSpec:
        size = (224, 224)

        def __call__(self) -> _FakeResizeTransform:
            return resize

    transform = _FakeEvalTransform()
    transform.resize = _BadResizeSpec()
    source = Image.new("RGB", (640, 480))
    with pytest.raises(ValueError, match="NEAREST"):
        G1VlaHumanoidPolicy._apply_checkpoint_eval_image_transform(
            SimpleNamespace(model_transform=transform),
            (source,),
        )
    source.close()


def test_history_is_oldest_to_current_and_bats_is_frozen() -> None:
    history = VlaImageHistory(episode_index=17)
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
    history = VlaImageHistory(episode_index=stable_episode_int("ascend"))
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


def test_history_remains_two_hz_when_policy_replans_at_ten_hz() -> None:
    """Fresh 100 ms decisions must not accelerate the BATS history clock."""
    history = VlaImageHistory(episode_index=17)
    selected_batches: list[tuple[Image.Image, ...]] = []
    try:
        for index in range(6):
            current = Image.new("RGB", (224, 224), color=(index, 0, 0))
            selected_batches.append(
                history.select_with_current(
                    current,
                    timestamp_us=index * 100_000,
                    capture_receipt_sha256=f"receipt-{index}",
                )
            )
            current.close()
        assert [len(batch) for batch in selected_batches] == [1, 1, 1, 1, 1, 2]
        assert [int(np.asarray(image)[0, 0, 0]) for image in selected_batches[-1]] == [
            0,
            5,
        ]
    finally:
        for batch in selected_batches:
            for image in batch:
                image.close()
        history.close()


def _policy(
    model: VlaPsiActorCritic,
    *,
    seed: int = 41,
    run_config_sha256: str = STAIRSBLOCKS_VLA_BUNDLE_PROFILE.run_config_sha256,
    sampling_mode: str = FLOW_SDE_TRAINING,
) -> G1VlaHumanoidPolicy:
    profile = STAIRSBLOCKS_VLA_BUNDLE_PROFILE
    return G1VlaHumanoidPolicy(
        _Engine(model),
        session_uuid="session",
        request=SimpleNamespace(random_seed=seed, scenario_id="ascend"),
        image_format="jpeg",
        expected_schedule_sha256=model.schedule.sha256,
        humanoid_repo_path=_HUMANOID_REPO,
        humanoid_vla_reference_adapter_sha256=_HUMANOID_VLA_REFERENCE_ADAPTER_SHA256,
        humanoid_motion_reference_sha256=_HUMANOID_MOTION_REFERENCE_SHA256,
        language_instruction="walk ahead.",
        run_config_sha256=run_config_sha256,
        bundle_profile=profile,
        sampling_mode=sampling_mode,
        replan_controller_ticks=25,
    )


def test_reference_adapter_preserves_policy_target_zero_and_exact_target_clock() -> (
    None
):
    rows = np.zeros((30, 38), dtype=np.float32)
    rows[:, :29] = np.arange(30, dtype=np.float32)[:, None] * 0.01
    rows[:, 29:35] = np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32)
    rows[:, 35] = np.arange(30, dtype=np.float32) * 0.01
    rows[:, 36] = np.arange(30, dtype=np.float32) * -0.005
    qpos = np.zeros(36, dtype=np.float32)
    qpos[:3] = [0.1, -0.2, 0.8]
    qpos[3] = 1.0
    qpos[7:] = -0.1
    qvel = np.zeros(35, dtype=np.float32)

    adapter = VlaMotionReferenceAdapter(
        _HUMANOID_REPO,
        reference_adapter_sha256=_HUMANOID_VLA_REFERENCE_ADAPTER_SHA256,
        motion_reference_sha256=_HUMANOID_MOTION_REFERENCE_SHA256,
    )
    assert adapter.rtc_max_delay_exclusive == 8
    built = adapter.build(rows, qpos=qpos, qvel=qvel, timestamp_us=500_000)
    reference = built.reference

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
    np.testing.assert_array_equal(reference.frames[0].root_position[:2], qpos[:2])
    np.testing.assert_allclose(
        reference.frames[10].root_position[:2],
        qpos[:2] + np.asarray([0.06, -0.03]),
        rtol=0,
        atol=1.0e-8,
    )
    assert all(frame.root_position[2] == qpos[2] for frame in reference.frames)
    assert built.decode_context.schema == "full_pelvis_rotation_local_xy_completed_z/v1"
    np.testing.assert_array_equal(
        built.decode_context.local_xy_from_frame_zero[0],
        np.zeros(2, dtype=np.float32),
    )
    np.testing.assert_allclose(
        built.decode_context.local_xy_from_frame_zero[10],
        np.asarray([0.06, -0.03], dtype=np.float32),
        rtol=0,
        atol=1.0e-8,
    )


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
    assert output.motion_reference.decode_context is not None
    assert output.motion_reference.decode_context.schema == (
        "full_pelvis_rotation_local_xy_completed_z/v1"
    )
    torch.testing.assert_close(
        output.motion_reference.decode_context.chunk_base_quaternion_wxyz,
        policy_input.qpos[3:7],
    )
    assert output.motion_reference.decode_context.local_xy_from_frame_zero.shape == (
        50,
        2,
    )
    assert output.replay_data.old_logprob is not None
    payload = output.replay_data.payload
    assert (
        payload["run_config_sha256"]
        == STAIRSBLOCKS_VLA_BUNDLE_PROFILE.run_config_sha256
    )
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
        payload["humanoid"]["vla_raw_action_rows"],
    )
    assert payload["latent_chain"].shape == (11, 30, 38)
    assert not bool(payload["rtc_prefix_mask"].any())
    assert not bool(payload["rtc_prefix_normalized_actions"].any())
    assert set(payload["humanoid"]) == {"vla_raw_action_rows"}
    assert "transition" not in payload
    for value in payload.values():
        if isinstance(value, torch.Tensor):
            assert value.device.type == "cpu"
    policy.close()


def test_four_tuple_psi_rebuilds_checkpoint_training_visual_pooling() -> None:
    """The ckpt_1500 serving gap must not silently unpool BATS history."""
    psi = _FakeFourTuplePsi()
    policy = _policy(_model(psi), seed=43)

    first = policy.step((_input(step=0, color=(11, 12, 13)),))[0]
    assert first.motion_reference is not None
    assert first.replay_data is not None
    torch.testing.assert_close(
        first.replay_data.payload["visual_pool_factors"],
        torch.tensor([1], dtype=torch.int64),
    )

    second = policy.step(
        (
            _input(
                step=1,
                color=(21, 22, 23),
                active_reference_ids=(first.motion_reference.reference_id,) * 25,
            ),
        )
    )[0]
    assert second.replay_data is not None
    torch.testing.assert_close(
        second.replay_data.payload["visual_pool_factors"],
        torch.tensor([2, 1], dtype=torch.int64),
    )
    assert psi.model_transform.seen_pool_factors == [[1], [2, 1]]
    policy.close()


def test_policy_stamps_the_selected_bundle_run_config_identity() -> None:
    profile = STAIRSBLOCKS_VLA_BUNDLE_PROFILE
    policy = _policy(_model(), run_config_sha256=profile.run_config_sha256)

    output = policy.step((_input(step=0, color=(7, 8, 9)),))[0]

    assert output.replay_data is not None
    assert output.replay_data.payload["run_config_sha256"] == profile.run_config_sha256
    policy.close()


def test_async_training_initial_delay_preserves_suffix_on_first_replan() -> None:
    """Async training keeps the conservative native six-row latency prior."""
    model = _model()
    policy = _policy(model, seed=13)
    first = policy.step((_input(step=0, color=(4, 5, 6)),))[0]
    assert first.motion_reference is not None

    second = policy.step(
        (
            _input(
                step=1,
                color=(7, 8, 9),
                active_reference_ids=(first.motion_reference.reference_id,) * 25,
            ),
        )
    )[0]

    assert second.replay_data is not None
    assert second.model_extra is not None
    assert int(second.replay_data.payload["rtc_prefix_mask"].sum()) == 6
    assert second.model_extra["vla_rtc_delay_rows"] == 6
    assert second.model_extra["vla_rtc_source_cursor_h50"] == 25
    assert second.model_extra["vla_rtc_source_start_row_h30"] == 15
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
    torch.testing.assert_close(previous.normalized_actions, first_chunk)
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
    assert second.model_extra["vla_rtc_delay_rows"] == 6
    assert second.model_extra["vla_rtc_source_cursor_h50"] == 25
    assert second.model_extra["vla_rtc_source_start_row_h30"] == 15
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
    assert second.model_extra["vla_rtc_source_cursor_h50"] == 28
    assert second.model_extra["vla_rtc_source_start_row_h30"] == 17
    assert int(second_payload["rtc_prefix_mask"].sum()) == 6

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
    assert second.model_extra["vla_rtc_source_cursor_h50"] == 50
    assert second.model_extra["vla_rtc_source_start_row_h30"] == 30
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
    assert second.model_extra["vla_rtc_delay_rows"] == 5
    assert second.model_extra["vla_rtc_source_cursor_h50"] == 41
    assert second.model_extra["vla_rtc_source_start_row_h30"] == 25
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
    assert third.model_extra["vla_rtc_delay_rows"] == 7
    policy.close()


def test_yaw_only_rtc_rebase_preserves_world_target() -> None:
    """Relative action pose is re-expressed under a changed chunk base."""
    normalizer = VlaQ99Normalizer(
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
        "sample_actions_native_ode",
        lambda **kwargs: SimpleNamespace(
            density_latent=torch.zeros_like(rows),
            clipped_normalized=torch.zeros_like(rows),
            denormalized_wire=rows,
            schedule_sha256=model.qualification_schedule.sha256,
            clip_normalized_actions=False,
        ),
    )
    monkeypatch.setattr(
        model,
        "sample_actions",
        lambda **kwargs: pytest.fail("qualification called the Flow-SDE sampler"),
    )
    policy = _policy(model, sampling_mode=NATIVE_ODE_QUALIFICATION)

    output = policy.step((_input(step=0, color=(3, 4, 5)),))[0]

    assert output.motion_reference is not None
    assert output.logprob is None
    assert output.value is None
    assert output.replay_data is not None
    assert output.replay_data.payload_schema == QUALIFICATION_REPLAY_SCHEMA
    assert output.replay_data.old_logprob is None
    assert output.replay_data.payload["sampling_mode"] == NATIVE_ODE_QUALIFICATION
    torch.testing.assert_close(
        output.replay_data.payload["denormalized_wire_actions"], rows[0]
    )
    assert "latent_chain" not in output.replay_data.payload
    policy.close()


def test_synchronous_qualification_keeps_fixed_continuity_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Zero simulator-time latency must not erase the action seam overlap."""
    model = _model()

    def _sample_native_ode(**kwargs: torch.Tensor) -> SimpleNamespace:
        prefix = kwargs["rtc_prefix_normalized_actions"]
        mask = kwargs["rtc_prefix_mask"]
        clipped = torch.zeros_like(prefix)
        clipped = torch.where(mask[..., None], prefix, clipped)
        denormalized = model.normalizer.denormalize_action(clipped)
        # Keep the sampler-owned suffix a valid identity rotation. Prefix rows
        # remain the exact denormalization of the previous chunk's suffix.
        suffix = ~mask
        denormalized[:, :, 29] = torch.where(
            suffix, torch.ones_like(denormalized[:, :, 29]), denormalized[:, :, 29]
        )
        denormalized[:, :, 33] = torch.where(
            suffix, torch.ones_like(denormalized[:, :, 33]), denormalized[:, :, 33]
        )
        return SimpleNamespace(
            density_latent=clipped.clone(),
            clipped_normalized=clipped,
            denormalized_wire=denormalized,
            schedule_sha256=model.qualification_schedule.sha256,
            clip_normalized_actions=False,
        )

    monkeypatch.setattr(
        model,
        "sample_actions_native_ode",
        _sample_native_ode,
    )
    policy = _policy(model, sampling_mode=NATIVE_ODE_QUALIFICATION)

    previous = policy.step((_input(step=0, color=(3, 4, 5)),))[0]
    assert previous.motion_reference is not None
    assert previous.replay_data is not None
    lane = policy._lanes[0]
    assert lane.rtc_state is not None
    assert lane.rtc_state.delay_rows == (0,)

    # More replans than the six-entry latency window proves observed zeros can
    # no longer flush the separately configured qualification continuity ABI.
    for step in range(1, 9):
        previous_wire = previous.replay_data.payload["denormalized_wire_actions"]
        current = policy.step(
            (
                _input(
                    step=step,
                    color=(step, step + 1, step + 2),
                    active_reference_ids=(previous.motion_reference.reference_id,) * 25,
                ),
            )
        )[0]
        assert current.motion_reference is not None
        assert current.replay_data is not None
        assert current.model_extra is not None
        payload = current.replay_data.payload
        assert int(payload["rtc_prefix_mask"].sum()) == 6
        assert payload["rtc_prefix_mode"] == "fixed_continuity/v1"
        assert payload["rtc_configured_continuity_prefix_rows"] == 6
        assert payload["rtc_observed_delay_rows"] == 0
        torch.testing.assert_close(
            payload["denormalized_wire_actions"][:6, :29],
            previous_wire[15:21, :29],
            rtol=0,
            atol=0,
        )
        expected_velocity = (previous_wire[15, :29] - previous_wire[14, :29]) * 30.0
        torch.testing.assert_close(
            current.motion_reference.frames[0].joint_velocity,
            expected_velocity,
            rtol=0,
            atol=2.0e-5,
        )
        assert current.model_extra["vla_rtc_delay_rows"] == 6
        assert current.model_extra["vla_rtc_actual_prefix_rows"] == 6
        assert current.model_extra["vla_rtc_observed_delay_rows"] == 0
        assert current.model_extra["vla_rtc_prefix_mode"] == "fixed_continuity/v1"
        previous = current

    assert lane.rtc_state is not None
    assert lane.rtc_state.delay_rows == (0, 0, 0, 0, 0, 0)
    policy.close()


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    (
        ("camera_logical_id", "vla_d455_policy_rgb"),
        ("camera_image_format", "png"),
        ("camera_contract_sha256", "d" * 64),
        ("language_instruction", "Walk ahead."),
    ),
)
def test_ckpt1500_profile_rejects_camera_or_instruction_yaml_override(
    field: str,
    wrong_value: str,
) -> None:
    model = _model()
    profile = STAIRSBLOCKS_VLA_BUNDLE_PROFILE
    bundle_config = {
        "sampling_mode": FLOW_SDE_TRAINING,
        "expected_schedule_sha256": model.schedule.sha256,
        "humanoid_repo_path": str(_HUMANOID_REPO),
        "humanoid_vla_reference_adapter_sha256": (
            _HUMANOID_VLA_REFERENCE_ADAPTER_SHA256
        ),
        "humanoid_motion_reference_sha256": _HUMANOID_MOTION_REFERENCE_SHA256,
        "language_instruction": profile.language_instruction,
        "camera_logical_id": profile.camera_logical_id,
        "camera_image_format": profile.camera_image_format,
        "camera_contract_sha256": profile.camera_contract_sha256,
    }
    bundle_config[field] = wrong_value
    config = SimpleNamespace(
        policy=SimpleNamespace(
            model=SimpleNamespace(
                path=f"/test/models/{profile.model_id}",
                bundle_config=bundle_config,
            ),
            inference=SimpleNamespace(return_trace_for_rl=True),
        )
    )

    with pytest.raises(ValueError, match="attested model profile"):
        build_humanoid_policy_factory(config, _Engine(model))


def test_ckpt1500_profile_records_exact_training_eval_camera_abi() -> None:
    profile = STAIRSBLOCKS_VLA_BUNDLE_PROFILE
    assert profile.camera_profile == "vla_d435_native"
    assert profile.camera_logical_id == "vla_d435_policy_rgb"
    assert profile.camera_image_format == "jpeg"
    assert profile.camera_source_resolution == (640, 480)
    assert profile.camera_contract_sha256 == (
        "dc01fdaaac67036fb69044ea8ff66fb554131dc89bd708a66b8b955c3471a297"
    )
    assert profile.camera_preprocess_profile == (
        "psi_resize_nearest_224x224_center_crop_224x224_no_letterbox.v1"
    )
    # Dataset task is ``Walk ahead.``; VlnverseRepackTransform lowercases it.
    assert profile.language_instruction == "walk ahead."
    assert profile.native_qualification_clip_normalized_actions is False


def test_factory_and_inference_adapter_fail_closed_on_wrong_contracts() -> None:
    model = _model()
    adapter = VlaNativeInferenceModel(model)
    assert adapter.get_model() is model
    with pytest.raises(TypeError, match="expected VLA"):
        adapter.set_model(torch.nn.Linear(2, 2))

    config = SimpleNamespace(
        policy=SimpleNamespace(
            model=SimpleNamespace(
                path=(f"/test/models/{STAIRSBLOCKS_VLA_BUNDLE_PROFILE.model_id}"),
                bundle_config={
                    "sampling_mode": FLOW_SDE_TRAINING,
                    "expected_schedule_sha256": model.schedule.sha256,
                    "humanoid_repo_path": str(_HUMANOID_REPO),
                    "humanoid_vla_reference_adapter_sha256": _HUMANOID_VLA_REFERENCE_ADAPTER_SHA256,
                    "humanoid_motion_reference_sha256": (
                        _HUMANOID_MOTION_REFERENCE_SHA256
                    ),
                    "language_instruction": "walk ahead.",
                    "camera_logical_id": (
                        STAIRSBLOCKS_VLA_BUNDLE_PROFILE.camera_logical_id
                    ),
                    "camera_image_format": "jpeg",
                    "camera_contract_sha256": (
                        STAIRSBLOCKS_VLA_BUNDLE_PROFILE.camera_contract_sha256
                    ),
                },
            ),
            inference=SimpleNamespace(return_trace_for_rl=True),
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
            decode_context_schema="full_pelvis_rotation_local_xy_completed_z/v1",
        ),
        policy_camera_spec=SimpleNamespace(
            schema="humanoid_policy_camera_rgb_qpos.v1",
            logical_id="wrong_camera",
            width=640,
            height=480,
            image_format="jpeg",
            contract_sha256=(STAIRSBLOCKS_VLA_BUNDLE_PROFILE.camera_contract_sha256),
        ),
    )
    with pytest.raises(ValueError, match="vla_d435_native camera contract changed"):
        factory("session", request)

    request.policy_camera_spec.logical_id = (
        STAIRSBLOCKS_VLA_BUNDLE_PROFILE.camera_logical_id
    )
    request.reference_spec.control_ticks_per_policy_step = 25
    policy = factory("session-visual-k25", request)
    assert policy._replan_controller_ticks == 25
    policy.close()

    request.reference_spec.decode_context_schema = ""
    with pytest.raises(ValueError, match="decode-context session contract"):
        factory("session-missing-decode-context", request)
