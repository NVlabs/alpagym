# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure VLA action-chunk to controller-neutral motion-reference adapter.

The 30x38 tensor remains the policy action.  This module only builds the
dispatch representation consumed by the released G1 motion-reference path;
it does not own a controller or a simulator.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import DTypeLike
from alpasim_grpc.v0.humanoid_contracts import (
    HUMANOID_FULL_ROTATION_LOCAL_XY_DECODE_CONTEXT_SCHEMA,
)


H50_FRAME_COUNT = 50
REFERENCE_PERIOD_US = 20_000
# Native evaluation executes 15 rows on the 30 Hz action clock before a new
# inference.  That is 0.5 s, or 25 ticks on the 50 Hz controller clock.  Visual
# Sonic's future window is maintained by the controller-owned rolling queue;
# it is not a second VLA inference cadence.
NATIVE_REPLAN_CONTROLLER_TICKS = 25
SUPPORTED_REPLAN_CONTROLLER_TICKS = frozenset({NATIVE_REPLAN_CONTROLLER_TICKS})
REPLAN_SOURCE_ROWS = 15
_VLA_ACTION_ROWS = 30
_VLA_ACTION_WIDTH = 38
_G1_QPOS = 36
_G1_QVEL = 35


@dataclass(frozen=True)
class _HumanoidModules:
    reference_adapter: ModuleType
    motion: ModuleType


@dataclass(frozen=True)
class VlaReferenceDecodeContext:
    """Full-rotation local-XY information retained for physics completion."""

    schema: str
    chunk_base_quaternion_wxyz: np.ndarray
    local_xy_from_frame_zero: np.ndarray

    def __post_init__(self) -> None:
        """Own finite immutable copies of the pinned adapter output."""
        if self.schema != HUMANOID_FULL_ROTATION_LOCAL_XY_DECODE_CONTEXT_SCHEMA:
            raise ValueError("VLA reference adapter returned an unsupported schema")
        quaternion = _finite_array(
            self.chunk_base_quaternion_wxyz,
            name="VLA decode-context chunk-base quaternion",
            shape=(4,),
            dtype=np.float32,
        )
        norm = float(np.linalg.norm(quaternion))
        if not np.isclose(norm, 1.0, rtol=0.0, atol=1.0e-5):
            raise ValueError(
                "VLA decode-context chunk-base quaternion must be normalized"
            )
        local_xy = _finite_array(
            self.local_xy_from_frame_zero,
            name="VLA decode-context local XY",
            shape=(H50_FRAME_COUNT, 2),
            dtype=np.float32,
        )
        if not np.array_equal(local_xy[0], np.zeros(2, dtype=np.float32)):
            raise ValueError("VLA decode-context local XY row zero must be exact zero")
        quaternion.setflags(write=False)
        local_xy.setflags(write=False)
        object.__setattr__(self, "chunk_base_quaternion_wxyz", quaternion)
        object.__setattr__(self, "local_xy_from_frame_zero", local_xy)


@dataclass(frozen=True)
class VlaReferenceBuild:
    """One H50 reference and the context required to finish its root path."""

    reference: Any
    decode_context: VlaReferenceDecodeContext


class VlaMotionReferenceAdapter:
    """Encode a native VLA action as a one-second 50 Hz reference buffer."""

    def __init__(
        self,
        humanoid_repo_path: Path | str,
        *,
        reference_adapter_sha256: str,
        motion_reference_sha256: str,
    ) -> None:
        """Load the two content-pinned native reference modules."""
        self._repo_path = Path(humanoid_repo_path).expanduser().resolve()
        self._support = _load_humanoid_modules(
            self._repo_path,
            reference_adapter_sha256=_require_sha256(
                "reference_adapter_sha256", reference_adapter_sha256
            ),
            motion_reference_sha256=_require_sha256(
                "motion_reference_sha256", motion_reference_sha256
            ),
        )
        joint_names = tuple(self._support.reference_adapter.VLA_G1_JOINT_NAMES)
        if len(joint_names) != 29 or len(set(joint_names)) != 29:
            raise ValueError("VLA reference adapter has an invalid G1 joint contract")
        self.joint_names = joint_names
        rtc_prefix_length_max = (
            self._support.reference_adapter.VLA_RTC_PREFIX_LENGTH_MAX
        )
        if (
            isinstance(rtc_prefix_length_max, bool)
            or not isinstance(rtc_prefix_length_max, int)
            or not 1 <= rtc_prefix_length_max < _VLA_ACTION_ROWS
        ):
            raise ValueError("VLA reference adapter has an invalid RTC prefix contract")
        self.rtc_max_delay_exclusive = rtc_prefix_length_max + 1

    def build(
        self,
        rows: object,
        *,
        qpos: object,
        qvel: object,
        timestamp_us: int,
        velocity_seed_joint_position: object | None = None,
    ) -> VlaReferenceBuild:
        """Return the H50 reference plus its physics decode context.

        Target zero is a policy target, not the measured robot pose.  On the
        first dispatch its velocity predecessor is the reset/hold pose.  A
        continuous caller supplies the preceding 30 Hz policy target on later
        dispatches, matching PosePump's sequence-boundary derivative.

        This adapter never manufactures an H70 tail.  The runtime owns buffer
        lifetime: after frame 49 it emits a zero-velocity terminal hold until a
        fresh H50 plan atomically replaces the active buffer.
        """

        actions = _finite_array(
            rows,
            name="VLA action rows",
            shape=(_VLA_ACTION_ROWS, _VLA_ACTION_WIDTH),
            dtype=np.float32,
        )
        live_qpos = _finite_array(
            qpos, name="G1 qpos", shape=(_G1_QPOS,), dtype=np.float64
        )
        _finite_array(qvel, name="G1 qvel", shape=(_G1_QVEL,), dtype=np.float64)
        timestamp = int(timestamp_us)
        if timestamp < 0 or timestamp != timestamp_us:
            raise ValueError(
                "VLA reference timestamp_us must be a non-negative integer"
            )
        root_quaternion = live_qpos[3:7]
        quaternion_norm = float(np.linalg.norm(root_quaternion))
        if not np.isclose(quaternion_norm, 1.0, rtol=0.0, atol=1.0e-5):
            raise ValueError(
                "VLA live root quaternion must be normalized; "
                f"found norm {quaternion_norm:.9g}"
            )

        adapter = self._support.reference_adapter
        chunk = adapter.VlaServerActionChunk(
            actions=actions,
            chunk_base_quat_wxyz=live_qpos[3:7],
            chunk_base_xy=live_qpos[:2],
        )
        if velocity_seed_joint_position is None:
            velocity_seed = adapter.VlaVelocitySeed.episode_reset_hold(live_qpos[7:])
        else:
            seed_position = _finite_array(
                velocity_seed_joint_position,
                name="previous VLA policy target",
                shape=(29,),
                dtype=np.float32,
            )
            velocity_seed = adapter.VlaVelocitySeed.previous_policy_target(
                seed_position
            )
        targets = adapter.convert_vla_chunk_to_reference_targets(
            chunk,
            source_row_cursor=0,
            velocity_seed=velocity_seed,
        )
        native_decode_context = (
            adapter.convert_vla_chunk_to_full_rotation_local_xy_context(
                chunk,
                source_row_cursor=0,
            )
        )
        if len(targets) != 50:
            raise AssertionError("VLA reference adapter did not return 50 targets")

        motion = self._support.motion
        start_s = timestamp / 1_000_000.0
        # The VLA does not predict pelvis Z, but it does predict a root-XY path.
        # Preserve and rebase that path so frame zero is exactly the live root
        # XY.  The Z value below is explicitly only a placeholder: the current
        # Visual SONIC qualification lane replaces reference[580:590] per
        # controller tick with measured live pelvis Z.  A production lane must
        # extend the policy/root-trajectory contract with true future Z.
        target_zero_xy = np.asarray(targets[0].root_xy, dtype=np.float64)
        root_xy_translation = live_qpos[:2] - target_zero_xy
        frames = []
        for frame_index, target in enumerate(targets):
            root_position = live_qpos[:3].copy()
            root_position[:2] = (
                np.asarray(target.root_xy, dtype=np.float64) + root_xy_translation
            )
            frames.append(
                motion.MotionFrame(
                    time_s=start_s + frame_index * REFERENCE_PERIOD_US / 1_000_000.0,
                    joint_position=target.joint_position,
                    joint_velocity=target.joint_velocity,
                    # Root Z remains an explicit placeholder because it is
                    # absent from the 38-D action.  It must not be described as
                    # a completed source-motion reference.
                    root_position=root_position,
                    root_quaternion_wxyz=target.root_quaternion_wxyz,
                )
            )
        reference = motion.MotionReference(
            frames=tuple(frames),
            joint_names=self.joint_names,
        )
        if len(reference) != H50_FRAME_COUNT:
            raise AssertionError("VLA adapter emitted a non-H50 reference")
        period_s = reference.uniform_period_s()
        if period_s is None or not np.isclose(period_s, 0.02, rtol=0.0, atol=1.0e-9):
            raise AssertionError("VLA reference is not on the 50 Hz grid")

        return VlaReferenceBuild(
            reference=reference,
            decode_context=VlaReferenceDecodeContext(
                schema=str(native_decode_context.schema),
                chunk_base_quaternion_wxyz=np.asarray(
                    native_decode_context.chunk_base_quat_wxyz,
                    dtype=np.float32,
                ),
                local_xy_from_frame_zero=np.asarray(
                    native_decode_context.local_xy_from_frame_zero,
                    dtype=np.float32,
                ),
            ),
        )


def _load_humanoid_modules(
    repo_path: Path,
    *,
    reference_adapter_sha256: str,
    motion_reference_sha256: str,
) -> _HumanoidModules:
    """Load the exact converter and motion-reference source revisions."""
    required = {
        "reference_adapter": (
            repo_path / "policies" / "reference_adapters" / "vla.py",
            reference_adapter_sha256,
        ),
        "motion": (
            repo_path / "policies" / "motion_reference.py",
            motion_reference_sha256,
        ),
    }
    missing = [str(path) for path, _digest in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"humanoid VLA support is incomplete: {missing}")
    return _HumanoidModules(
        reference_adapter=_load_source_module(
            required["reference_adapter"][0],
            role="vla_motion_reference_adapter",
            expected_sha256=required["reference_adapter"][1],
        ),
        motion=_load_source_module(
            required["motion"][0],
            role="motion_reference",
            expected_sha256=required["motion"][1],
        ),
    )


def _load_source_module(
    path: Path,
    *,
    role: str,
    expected_sha256: str,
) -> ModuleType:
    """Execute one verified source snapshot under a content-derived key."""
    source = path.read_bytes()
    content_sha256 = hashlib.sha256(source).hexdigest()
    if content_sha256 != expected_sha256:
        raise ValueError(
            f"humanoid {role} content SHA256 changed: "
            f"expected {expected_sha256}, found {content_sha256}"
        )
    module_name = f"_alpagym_vla_{role}_{content_sha256}"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load humanoid support module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        code = compile(source, str(path), "exec")
        exec(code, module.__dict__)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    module_file = Path(str(module.__file__)).resolve()
    if module_file != path.resolve():
        raise ImportError(f"humanoid support module resolved outside {path}")
    return module


def _require_sha256(name: str, value: str) -> str:
    """Require one lowercase SHA-256 content identity."""
    digest = str(value)
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"VLA {name} must be a lowercase SHA256 digest")
    return digest


def _finite_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: DTypeLike,
) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite shape {shape}, got {array.shape}")
    return np.ascontiguousarray(array)


__all__ = (
    "H50_FRAME_COUNT",
    "REFERENCE_PERIOD_US",
    "NATIVE_REPLAN_CONTROLLER_TICKS",
    "SUPPORTED_REPLAN_CONTROLLER_TICKS",
    "REPLAN_SOURCE_ROWS",
    "VlaMotionReferenceAdapter",
)
