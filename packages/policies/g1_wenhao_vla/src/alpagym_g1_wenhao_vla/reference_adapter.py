# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure Wenhao 30 Hz chunk to vanilla SONIC's one-second reference buffer.

The 30x38 tensor remains the policy action.  This module only builds the
dispatch representation consumed by the already-qualified GRAIL/SONIC
motion-reference path; it does not own a controller or a simulator.
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


H50_FRAME_COUNT = 50
REFERENCE_PERIOD_US = 20_000
# Wenhao starts the next inference after 15 source rows.  In the synchronous
# AlpaGym rollout this is exactly 25 SONIC ticks; it is a replan trigger, not a
# truncation of the one-second reference buffer.
REPLAN_CONTROLLER_TICKS = 25
REPLAN_SOURCE_ROWS = 15
_WENHAO_ROWS = 30
_WENHAO_WIDTH = 38
_G1_QPOS = 36
_G1_QVEL = 35


@dataclass(frozen=True)
class _HumanoidModules:
    planner: ModuleType
    motion: ModuleType


class WenhaoH50ReferenceAdapter:
    """Encode native Wenhao actions as a one-second, 50 Hz SONIC buffer."""

    def __init__(
        self,
        humanoid_repo_path: Path | str,
        *,
        planner_sha256: str,
        motion_reference_sha256: str,
    ) -> None:
        """Load the two content-pinned native reference modules."""
        self._repo_path = Path(humanoid_repo_path).expanduser().resolve()
        self._support = _load_humanoid_modules(
            self._repo_path,
            planner_sha256=_require_sha256("planner_sha256", planner_sha256),
            motion_reference_sha256=_require_sha256(
                "motion_reference_sha256", motion_reference_sha256
            ),
        )
        planner_names = tuple(self._support.planner.WENHAO_G1_JOINT_NAMES)
        if len(planner_names) != 29 or len(set(planner_names)) != 29:
            raise ValueError("Wenhao converter has an invalid G1 joint contract")
        self.joint_names = planner_names
        rtc_prefix_length_max = self._support.planner.WENHAO_RTC_PREFIX_LENGTH_MAX
        if (
            isinstance(rtc_prefix_length_max, bool)
            or not isinstance(rtc_prefix_length_max, int)
            or not 1 <= rtc_prefix_length_max < _WENHAO_ROWS
        ):
            raise ValueError("Wenhao converter has an invalid RTC prefix contract")
        self.rtc_max_delay_exclusive = rtc_prefix_length_max + 1

    def build(
        self,
        rows: object,
        *,
        qpos: object,
        qvel: object,
        timestamp_us: int,
        velocity_seed_joint_position: object | None = None,
    ) -> Any:
        """Return the 50 targets produced by the authoritative 30->50 Hz map.

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
            name="Wenhao rows",
            shape=(_WENHAO_ROWS, _WENHAO_WIDTH),
            dtype=np.float32,
        )
        live_qpos = _finite_array(
            qpos, name="G1 qpos", shape=(_G1_QPOS,), dtype=np.float64
        )
        _finite_array(qvel, name="G1 qvel", shape=(_G1_QVEL,), dtype=np.float64)
        timestamp = int(timestamp_us)
        if timestamp < 0 or timestamp != timestamp_us:
            raise ValueError(
                "Wenhao reference timestamp_us must be a non-negative integer"
            )
        root_quaternion = live_qpos[3:7]
        quaternion_norm = float(np.linalg.norm(root_quaternion))
        if not np.isclose(quaternion_norm, 1.0, rtol=0.0, atol=1.0e-5):
            raise ValueError(
                "Wenhao live root quaternion must be normalized; "
                f"found norm {quaternion_norm:.9g}"
            )

        planner = self._support.planner
        chunk = planner.WenhaoServerActionChunk(
            actions=actions,
            chunk_base_quat_wxyz=live_qpos[3:7],
            chunk_base_xy=live_qpos[:2],
        )
        if velocity_seed_joint_position is None:
            velocity_seed = planner.WenhaoVelocitySeed.episode_reset_hold(live_qpos[7:])
        else:
            seed_position = _finite_array(
                velocity_seed_joint_position,
                name="previous Wenhao policy target",
                shape=(29,),
                dtype=np.float32,
            )
            velocity_seed = planner.WenhaoVelocitySeed.previous_policy_target(
                seed_position
            )
        targets = planner.convert_wenhao_chunk_to_sonic_targets(
            chunk,
            source_cursor=0,
            velocity_seed=velocity_seed,
        )
        if len(targets) != 50:
            raise AssertionError("Wenhao converter did not return 50 SONIC targets")

        motion = self._support.motion
        start_s = timestamp / 1_000_000.0
        root_position = live_qpos[:3]
        frames = []
        for frame_index, target in enumerate(targets):
            frames.append(
                motion.MotionFrame(
                    time_s=start_s + frame_index * REFERENCE_PERIOD_US / 1_000_000.0,
                    joint_position=target.joint_position,
                    joint_velocity=target.joint_velocity,
                    # Wenhao has no root-Z action and GRAIL's streamed-motion
                    # observation does not consume root XYZ.  Preserve the
                    # realized root position instead of inventing a trajectory.
                    root_position=root_position,
                    root_quaternion_wxyz=target.root_quaternion_wxyz,
                )
            )
        reference = motion.MotionReference(
            frames=tuple(frames),
            joint_names=self.joint_names,
        )
        if len(reference) != H50_FRAME_COUNT:
            raise AssertionError("Wenhao adapter emitted a non-H50 reference")
        period_s = reference.uniform_period_s()
        if period_s is None or not np.isclose(period_s, 0.02, rtol=0.0, atol=1.0e-9):
            raise AssertionError("Wenhao H50 reference is not on the 50 Hz grid")

        return reference


def _load_humanoid_modules(
    repo_path: Path,
    *,
    planner_sha256: str,
    motion_reference_sha256: str,
) -> _HumanoidModules:
    """Load the exact planner and motion-reference source revisions."""
    required = {
        "planner": (
            repo_path / "policies" / "planners" / "wenhao_vla.py",
            planner_sha256,
        ),
        "motion": (
            repo_path / "policies" / "motion_reference.py",
            motion_reference_sha256,
        ),
    }
    missing = [str(path) for path, _digest in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"humanoid Wenhao support is incomplete: {missing}")
    return _HumanoidModules(
        planner=_load_source_module(
            required["planner"][0],
            role="wenhao_planner",
            expected_sha256=required["planner"][1],
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
    module_name = f"_alpagym_wenhao_{role}_{content_sha256}"
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
        raise ValueError(f"Wenhao {name} must be a lowercase SHA256 digest")
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
    "REPLAN_CONTROLLER_TICKS",
    "REPLAN_SOURCE_ROWS",
    "WenhaoH50ReferenceAdapter",
)
