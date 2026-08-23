# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training-compatible native-D435 decoding and BATS history selection."""

from __future__ import annotations

import hashlib
import io
import math
from functools import lru_cache
from dataclasses import dataclass, field
from typing import cast

import numpy as np
from PIL import Image, JpegImagePlugin, UnidentifiedImageError


def _jpeg_quantization_signature(
    image: Image.Image,
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    tables = getattr(image, "quantization", None)
    if not isinstance(tables, dict):
        raise ValueError("VLA D435 JPEG is missing quantization tables")
    typed_tables = cast(dict[int, list[int]], tables)
    return tuple(
        (int(table_id), tuple(int(value) for value in table))
        for table_id, table in sorted(typed_tables.items())
    )


@lru_cache(maxsize=1)
def _pil_jpeg95_contract_signature() -> tuple[
    int, tuple[tuple[int, tuple[int, ...]], ...]
]:
    """Return Pillow's declared quality=95/default-subsampling JPEG signature."""
    stream = io.BytesIO()
    reference = Image.new("RGB", (16, 16), color=(127, 91, 43))
    try:
        reference.save(stream, format="JPEG", quality=95)
    finally:
        reference.close()
    with Image.open(io.BytesIO(stream.getvalue())) as decoded:
        decoded.load()
        return (
            int(JpegImagePlugin.get_sampling(decoded)),
            _jpeg_quantization_signature(decoded),
        )


def decode_native_d435(image_bytes: bytes, *, image_format: str) -> Image.Image:
    """Decode one native 640x480 RGB D435 JPEG policy observation.

    The stairs-blocks checkpoint was trained from the republished native D435
    dataset, not the older 224x140 D455 policy raster.  This boundary therefore
    owns decoding only.  The checkpoint's ``ResizeImage((224, 224))`` and
    ``CenterCrop((224, 224))`` transforms are applied later, immediately before
    the Psi VLM builder; letterboxing here would silently change the model ABI.

    Args:
        image_bytes: One complete PNG or JPEG frame.
        image_format: Session-declared encoding; ckpt_1500 requires ``"jpeg"``.

    Returns:
        An owned 640x480 uint8 RGB PIL image.
    """
    if image_format != "jpeg":
        raise ValueError("VLA D435 frame encoding must be JPEG")
    expected_format = "JPEG"
    try:
        with Image.open(io.BytesIO(image_bytes)) as decoded:
            decoded.load()
            if decoded.format != expected_format or decoded.mode != "RGB":
                raise ValueError("VLA D435 frame encoding or RGB mode changed")
            if decoded.size != (640, 480):
                raise ValueError("VLA D435 frame must be exactly 640x480")
            signature = (
                int(JpegImagePlugin.get_sampling(decoded)),
                _jpeg_quantization_signature(decoded),
            )
            if signature != _pil_jpeg95_contract_signature():
                raise ValueError(
                    "VLA D435 frame must use Pillow JPEG quality=95 default 4:2:0"
                )
            frame = decoded.copy()
    except (OSError, SyntaxError, UnidentifiedImageError) as exc:
        raise ValueError("VLA D435 frame cannot be decoded") from exc
    return frame


def decode_legacy_d455_letterbox(
    image_bytes: bytes,
    *,
    image_format: str,
) -> Image.Image:
    """Decode the separately attested legacy 224x140 D455 PNG profile."""
    if image_format != "png":
        raise ValueError("legacy VLA D455 frame encoding must be PNG")
    try:
        with Image.open(io.BytesIO(image_bytes)) as decoded:
            decoded.load()
            if decoded.format != "PNG" or decoded.mode != "RGB":
                raise ValueError("legacy VLA D455 encoding or RGB mode changed")
            if decoded.size != (224, 140):
                raise ValueError("legacy VLA D455 frame must be exactly 224x140")
            frame = decoded.copy()
    except (OSError, SyntaxError, UnidentifiedImageError) as exc:
        raise ValueError("legacy VLA D455 frame cannot be decoded") from exc
    canvas = Image.new("RGB", (224, 224), color=(0, 0, 0))
    canvas.paste(frame, (0, 42))
    frame.close()
    return canvas


def _expected_history_frames(num_frames: int, eps: float, decay: float) -> float:
    """Return BATS's expected Bernoulli history count for one decay."""
    if num_frames <= 0:
        return 0.0
    if decay <= 0.0:
        return float(num_frames)
    exp_sum = -np.expm1(-decay) / np.expm1(decay / float(num_frames))
    return float(eps * num_frames + (1.0 - eps) * exp_sum)


def stable_episode_int(episode_id: str) -> int:
    """Map one authoritative episode identity to BATS's deterministic seed.

    This is the exact ``_stable_episode_int`` algorithm used by VLA's
    authoritative evaluation client. Python's process-randomized ``hash`` must
    never be used for this identity.
    """
    if not isinstance(episode_id, str):
        raise TypeError("VLA episode_id must be a string")
    return int.from_bytes(
        hashlib.sha1(episode_id.encode("utf-8")).digest()[:8],
        "big",
        signed=False,
    ) % (1 << 31)


def select_bats_history_indices(
    current_index: int,
    episode_index: int,
) -> tuple[int, ...]:
    """Select prior frames with the checkpoint's exact BATS semantics.

    This is a literal semantic port of ``benchmark/vla_eval_client.py`` in the
    authoritative ``alpa_policy_eval`` delivery.  The pinned source function
    digests are ``14a4d77e...528a`` (expectation) and
    ``72ac7ade...0b99`` (selection).  Selection intentionally uses the nominal
    checkpoint budget 512/16/64; the measured 36/169 visual token counts belong
    to Qwen pooling and must never be substituted into this sampler.

    Args:
        current_index: Global index assigned to the current, non-history frame.
        episode_index: Stable simulator reset/episode identifier.

    Returns:
        Strictly ascending indices in ``[0, current_index)``.
    """
    candidates = list(range(max(int(current_index), 0)))
    if not candidates:
        return ()

    eps = float(np.clip(0.1, 0.0, 1.0))
    available = 512.0 - 64.0
    target_frames = min(float(len(candidates)), available / 16.0)
    if target_frames >= float(len(candidates)):
        return tuple(candidates)

    minimum_expected = eps * float(len(candidates))
    if minimum_expected >= target_frames:
        eps = max(0.0, min(eps, 0.99 * target_frames / float(len(candidates))))
    low = 0.0
    high = 4.0
    while _expected_history_frames(len(candidates), eps, high) > target_frames:
        high *= 2.0
        if high > 1_000_000.0:
            break
    for _ in range(48):
        middle = (low + high) * 0.5
        if _expected_history_frames(len(candidates), eps, middle) > target_frames:
            low = middle
        else:
            high = middle

    relative_time = (
        np.asarray(candidates, dtype=np.float64) - float(current_index)
    ) / max(float(current_index), 1.0)
    probabilities = (1.0 - eps) * np.exp(high * relative_time) + eps
    probabilities = np.clip(probabilities, 0.0, 1.0)
    rng = np.random.default_rng(
        292285 + int(episode_index) * 1_000_003 + int(current_index)
    )
    keep = rng.random(len(candidates)) < probabilities
    return tuple(
        index for index, retain in zip(candidates, keep, strict=True) if bool(retain)
    )


@dataclass
class _HistoryCapture:
    """One owned camera capture and its strong producer receipt identity."""

    source_identity: VlaImageSourceIdentity
    image: Image.Image


@dataclass(frozen=True)
class VlaImageSourceIdentity:
    """Immutable renderer evidence for one decoded VLA source JPEG."""

    env_id: int
    frame_start_us: int
    frame_end_us: int
    logical_id: str
    byte_length: int
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

    def __post_init__(self) -> None:
        """Reject incomplete or non-canonical source evidence."""

        if (
            self.env_id < 0
            or self.frame_start_us < 0
            or self.frame_end_us < 0
            or self.render_timestamp_us < 0
            or self.observation_decision_id < 0
            or self.byte_length <= 0
        ):
            raise ValueError("VLA source image identity has an invalid scalar")
        if not self.logical_id:
            raise ValueError("VLA source image logical_id must be non-empty")
        if not (self.frame_start_us == self.frame_end_us == self.render_timestamp_us):
            raise ValueError("VLA source image must be one zero-shutter capture")
        for name, digest in (
            ("render_state_sha256", self.render_state_sha256),
            ("camera_contract_sha256", self.camera_contract_sha256),
            ("image_sha256", self.image_sha256),
            ("render_receipt_sha256", self.render_receipt_sha256),
            ("scene_fingerprint", self.scene_fingerprint),
            ("model_signature_sha256", self.model_signature_sha256),
            ("camera_to_world_sha256", self.camera_to_world_sha256),
            ("renderer_binding_sha256", self.renderer_binding_sha256),
        ):
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"VLA source image {name} must be lowercase SHA256")

    def manifest_entry(self, *, role: str) -> dict[str, object]:
        """Return one JSON-compatible ordered visual-manifest entry."""

        if role not in {"history", "current"}:
            raise ValueError("VLA source image role must be history or current")
        return {
            "role": role,
            "env_id": self.env_id,
            "frame_start_us": self.frame_start_us,
            "frame_end_us": self.frame_end_us,
            "logical_id": self.logical_id,
            "byte_length": self.byte_length,
            "render_timestamp_us": self.render_timestamp_us,
            "observation_decision_id": self.observation_decision_id,
            "render_state_sha256": self.render_state_sha256,
            "camera_contract_sha256": self.camera_contract_sha256,
            "image_sha256": self.image_sha256,
            "render_receipt_sha256": self.render_receipt_sha256,
            "scene_fingerprint": self.scene_fingerprint,
            "model_signature_sha256": self.model_signature_sha256,
            "camera_to_world_sha256": self.camera_to_world_sha256,
            "renderer_binding_sha256": self.renderer_binding_sha256,
        }


@dataclass
class VlaImageHistory:
    """One lane's 2 Hz pending-capture history in global chronology."""

    episode_index: int
    _frames: list[_HistoryCapture] = field(default_factory=list)
    _pending: _HistoryCapture | None = None
    _last_timestamp_us: int | None = None
    last_selected_indices: tuple[int, ...] = ()
    last_selected_source_identities: tuple[VlaImageSourceIdentity, ...] = ()

    def select_with_current(
        self,
        current: Image.Image,
        *,
        source_identity: VlaImageSourceIdentity,
    ) -> tuple[Image.Image, ...]:
        """Commit one fresh elapsed capture and append the current image last.

        Multiple elapsed boundaries still commit only the single pending image,
        matching the authoritative rollout client. If the producer receipt has
        not advanced, the pending frame is deferred so the same capture never
        appears in both history and current. Returned images are owned copies
        ordered oldest-to-current.
        """
        timestamp_us = source_identity.render_timestamp_us
        capture_receipt_sha256 = source_identity.render_receipt_sha256
        if (
            self._last_timestamp_us is not None
            and timestamp_us < self._last_timestamp_us
        ):
            raise ValueError("VLA image history timestamps moved backwards")
        if self._pending is None:
            self._pending = _HistoryCapture(
                source_identity=source_identity,
                image=current.copy(),
            )
        elif (
            timestamp_us - self._pending.source_identity.render_timestamp_us >= 500_000
        ):
            if (
                self._pending.source_identity.render_receipt_sha256
                != capture_receipt_sha256
            ):
                # Transfer ownership of the old pending image into committed
                # history, then own one copy of the new boundary capture.
                self._frames.append(self._pending)
                self._pending = _HistoryCapture(
                    source_identity=source_identity,
                    image=current.copy(),
                )

        selected = select_bats_history_indices(len(self._frames), self.episode_index)
        selected = tuple(
            index
            for index in selected
            if self._frames[index].source_identity.render_receipt_sha256
            != capture_receipt_sha256
        )
        if any(index >= len(self._frames) for index in selected):
            raise AssertionError("BATS selected outside retained history")
        result = tuple(self._frames[index].image.copy() for index in selected) + (
            current.copy(),
        )
        self.last_selected_indices = selected
        self.last_selected_source_identities = tuple(
            self._frames[index].source_identity for index in selected
        ) + (source_identity,)
        self._last_timestamp_us = timestamp_us
        return result

    def close(self) -> None:
        """Release all owned decoded frames."""
        for capture in self._frames:
            capture.image.close()
        self._frames.clear()
        if self._pending is not None:
            self._pending.image.close()
        self._pending = None
        self._last_timestamp_us = None
        self.last_selected_indices = ()
        self.last_selected_source_identities = ()


def image_array(image: Image.Image) -> np.ndarray:
    """Return an owned HWC uint8 RGB array for diagnostics and tests."""
    array = np.asarray(image, dtype=np.uint8)
    if array.shape not in {(480, 640, 3), (224, 224, 3)} or not math.isfinite(
        float(array.mean())
    ):
        raise ValueError("VLA image has an invalid native or transformed raster")
    return array.copy()
