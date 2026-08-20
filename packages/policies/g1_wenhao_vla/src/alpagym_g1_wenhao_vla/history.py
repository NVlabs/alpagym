# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training-compatible D455 image geometry and BATS history selection."""

from __future__ import annotations

import hashlib
import io
import math
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, UnidentifiedImageError


def width_fit_letterbox_d455(image_bytes: bytes, *, image_format: str) -> Image.Image:
    """Decode a 224x140 RGB D455 frame and letterbox it to 224x224.

    The training parquet used a width-fit resize followed by symmetric black
    padding.  A native 224x140 frame is therefore preserved pixel-for-pixel and
    pasted at vertical offset 42; it is never stretched to a square.

    Args:
        image_bytes: One complete PNG or JPEG frame.
        image_format: Session-declared encoding, either ``"png"`` or ``"jpeg"``.

    Returns:
        An owned 224x224 uint8 RGB PIL image.
    """
    expected_format = {"png": "PNG", "jpeg": "JPEG"}[image_format]
    try:
        with Image.open(io.BytesIO(image_bytes)) as decoded:
            decoded.load()
            if decoded.format != expected_format or decoded.mode != "RGB":
                raise ValueError("Wenhao D455 frame encoding or RGB mode changed")
            if decoded.size != (224, 140):
                raise ValueError("Wenhao D455 frame must be exactly 224x140")
            frame = decoded.copy()
    except (OSError, SyntaxError, UnidentifiedImageError) as exc:
        raise ValueError("Wenhao D455 frame cannot be decoded") from exc

    scaled_height = round(frame.height * 224 / frame.width)
    if scaled_height != 140:
        raise ValueError("Wenhao D455 width-fit geometry changed")
    canvas = Image.new("RGB", (224, 224), color=(0, 0, 0))
    canvas.paste(frame, (0, (224 - scaled_height) // 2))
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

    This is the exact ``_stable_episode_int`` algorithm used by Wenhao's
    authoritative evaluation client. Python's process-randomized ``hash`` must
    never be used for this identity.
    """
    if not isinstance(episode_id, str):
        raise TypeError("Wenhao episode_id must be a string")
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

    timestamp_us: int
    receipt_sha256: str
    image: Image.Image


@dataclass
class WenhaoImageHistory:
    """One lane's 2 Hz pending-capture history in global chronology."""

    episode_index: int
    _frames: list[_HistoryCapture] = field(default_factory=list)
    _pending: _HistoryCapture | None = None
    _last_timestamp_us: int | None = None
    last_selected_indices: tuple[int, ...] = ()

    def select_with_current(
        self,
        current: Image.Image,
        *,
        timestamp_us: int,
        capture_receipt_sha256: str,
    ) -> tuple[Image.Image, ...]:
        """Commit one fresh elapsed capture and append the current image last.

        Multiple elapsed boundaries still commit only the single pending image,
        matching the authoritative rollout client. If the producer receipt has
        not advanced, the pending frame is deferred so the same capture never
        appears in both history and current. Returned images are owned copies
        ordered oldest-to-current.
        """
        timestamp_us = int(timestamp_us)
        if timestamp_us < 0:
            raise ValueError("Wenhao image timestamp must be non-negative")
        if not isinstance(capture_receipt_sha256, str) or not capture_receipt_sha256:
            raise ValueError("Wenhao image capture receipt must be non-empty")
        if (
            self._last_timestamp_us is not None
            and timestamp_us < self._last_timestamp_us
        ):
            raise ValueError("Wenhao image history timestamps moved backwards")
        if self._pending is None:
            self._pending = _HistoryCapture(
                timestamp_us=timestamp_us,
                receipt_sha256=capture_receipt_sha256,
                image=current.copy(),
            )
        elif timestamp_us - self._pending.timestamp_us >= 500_000:
            if self._pending.receipt_sha256 != capture_receipt_sha256:
                # Transfer ownership of the old pending image into committed
                # history, then own one copy of the new boundary capture.
                self._frames.append(self._pending)
                self._pending = _HistoryCapture(
                    timestamp_us=timestamp_us,
                    receipt_sha256=capture_receipt_sha256,
                    image=current.copy(),
                )

        selected = select_bats_history_indices(len(self._frames), self.episode_index)
        selected = tuple(
            index
            for index in selected
            if self._frames[index].receipt_sha256 != capture_receipt_sha256
        )
        if any(index >= len(self._frames) for index in selected):
            raise AssertionError("BATS selected outside retained history")
        result = tuple(self._frames[index].image.copy() for index in selected) + (
            current.copy(),
        )
        self.last_selected_indices = selected
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


def image_array(image: Image.Image) -> np.ndarray:
    """Return an owned HWC uint8 RGB array for diagnostics and tests."""
    array = np.asarray(image, dtype=np.uint8)
    if array.shape != (224, 224, 3) or not math.isfinite(float(array.mean())):
        raise ValueError("Wenhao letterboxed image has an invalid raster")
    return array.copy()
