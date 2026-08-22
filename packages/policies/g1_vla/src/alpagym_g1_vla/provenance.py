# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Content-addressed loader boundary for the attested VLA Psi bundle."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, NamedTuple

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class VlaBundleProfile:
    """Immutable content pins for one admitted VLA inference bundle."""

    model_id: str
    checkpoint_step: int
    model_sha256: str
    run_config_sha256: str
    argv_sha256: str
    stats_sha256: str
    base_vlm_tree_sha256: str
    psi_source_tree_sha256: str
    camera_profile: str
    camera_logical_id: str
    camera_image_format: str
    camera_contract_sha256: str
    camera_source_resolution: tuple[int, int]
    camera_preprocess_profile: str
    language_instruction: str
    native_qualification_inference_steps: int
    native_qualification_schedule_sha256: str
    native_qualification_clip_normalized_actions: bool
    native_qualification_continuity_prefix_rows: int
    stats_id: str = "stats_psi0.json"

    def __post_init__(self) -> None:
        """Reject malformed registry entries at import time."""
        if not self.model_id or "/" in self.model_id:
            raise ValueError("VLA bundle model_id must be one path component")
        if self.checkpoint_step < 0:
            raise ValueError("VLA checkpoint_step must be non-negative")
        for label in (
            "model_sha256",
            "run_config_sha256",
            "argv_sha256",
            "stats_sha256",
            "base_vlm_tree_sha256",
            "psi_source_tree_sha256",
            "camera_contract_sha256",
            "native_qualification_schedule_sha256",
        ):
            if _SHA256.fullmatch(str(getattr(self, label))) is None:
                raise ValueError(f"VLA profile {label} must be a lowercase SHA-256")
        if not self.camera_profile or not self.camera_logical_id:
            raise ValueError("VLA profile camera identity must be non-empty")
        if self.camera_image_format not in {"png", "jpeg"}:
            raise ValueError("VLA profile camera_image_format must be png or jpeg")
        if len(self.camera_source_resolution) != 2 or any(
            value <= 0 for value in self.camera_source_resolution
        ):
            raise ValueError("VLA profile camera source resolution must be positive")
        if not self.camera_preprocess_profile:
            raise ValueError("VLA profile camera preprocess identity must be non-empty")
        if not self.language_instruction:
            raise ValueError("VLA profile language instruction must be non-empty")
        if (
            isinstance(self.native_qualification_inference_steps, bool)
            or not isinstance(self.native_qualification_inference_steps, int)
            or self.native_qualification_inference_steps < 1
        ):
            raise ValueError(
                "VLA native qualification inference steps must be positive"
            )
        if not isinstance(self.native_qualification_clip_normalized_actions, bool):
            raise TypeError("VLA native qualification clip flag must be boolean")
        if (
            isinstance(self.native_qualification_continuity_prefix_rows, bool)
            or not isinstance(self.native_qualification_continuity_prefix_rows, int)
            or self.native_qualification_continuity_prefix_rows < 0
        ):
            raise ValueError(
                "VLA native qualification continuity prefix must be a "
                "non-negative row count"
            )


LEGACY_VLA_BUNDLE_PROFILE = VlaBundleProfile(
    model_id="qwen3vl-wenhao2b-s1-step3203",
    checkpoint_step=3203,
    model_sha256="5cfc977203c29e9495bf66d79b7a1c2e3862ad4ffdd9569ae616c5b4121cd1c1",
    run_config_sha256="9feafeb8d43ff840bb74b51d762c6aadea750532328f2e51a42057e737721ce0",
    argv_sha256="8896251cced8d97af6ea9c18bf84b82be9dd78184c4f854af1d56b1fac97311e",
    stats_sha256="9db767e3a82fbb9ab990e7102bca96c44a278122329f7d5a8574124435333e37",
    base_vlm_tree_sha256=(
        "3b8ae84cbc35e9f802414820b4d00a5b1ad5533278c423739c9f274d3e302ded"
    ),
    psi_source_tree_sha256=(
        "f060bfde41b883c92dafc9c13c82dc6d83d1af86a3b3559cacc63d3f874388c4"
    ),
    camera_profile="vla_d455",
    camera_logical_id="vla_d455_policy_rgb",
    camera_image_format="png",
    camera_contract_sha256=(
        "dcc2f3b03d4ab10a8fdd47b699aceddac500f04307085dd36c0236a9f2e238bd"
    ),
    camera_source_resolution=(224, 140),
    camera_preprocess_profile="legacy_d455_letterbox_224x140_to_224x224.v1",
    language_instruction="walk up the stairs.",
    # The original server denormalizes the unbounded ODE output directly.
    native_qualification_inference_steps=10,
    native_qualification_schedule_sha256=(
        "d01fcb068a81712b78f9b72ff95877e332ae03245a2d1aba518c83354cf7ad12"
    ),
    native_qualification_clip_normalized_actions=False,
    native_qualification_continuity_prefix_rows=6,
)

STAIRSBLOCKS_VLA_BUNDLE_PROFILE = VlaBundleProfile(
    model_id="qwen3vl-vla2b-stairsblocks-step1500",
    checkpoint_step=1500,
    model_sha256="7d42e0e1714b901e6e9d7bcd10f4ecab58d66ee7a33575f26bc6f5f063b438d5",
    run_config_sha256="92b1cf52a7dec338b225c12d8a78592edcca49bc60782959368b70d7d31961c3",
    argv_sha256="2ae4545a571ce78768300caf9902d3994de7afc407ca32343f01c50fe647db29",
    stats_sha256="181a9a55918351def894b3dff5aefd1d7f7cea1b0d515f4684ae7468f98b1aa5",
    base_vlm_tree_sha256=(
        "3b8ae84cbc35e9f802414820b4d00a5b1ad5533278c423739c9f274d3e302ded"
    ),
    psi_source_tree_sha256=(
        "f314b603158210bcac2907d242c2c18e51808af79ca6effd22b5b7e0c3c36829"
    ),
    camera_profile="vla_d435_native",
    camera_logical_id="vla_d435_policy_rgb",
    camera_image_format="jpeg",
    camera_contract_sha256=(
        "dc01fdaaac67036fb69044ea8ff66fb554131dc89bd708a66b8b955c3471a297"
    ),
    camera_source_resolution=(640, 480),
    camera_preprocess_profile=(
        "psi_resize_nearest_224x224_center_crop_224x224_no_letterbox.v1"
    ),
    # Lerobot stores ``Walk ahead.``; VlnverseRepackTransform lowercases it
    # before every training example reaches Psi.
    language_instruction="walk ahead.",
    # Native serving runs ten Euler evaluations and applies the q01/q99 affine
    # denormalization directly to the unbounded flow output.
    native_qualification_inference_steps=10,
    native_qualification_schedule_sha256=(
        "d01fcb068a81712b78f9b72ff95877e332ae03245a2d1aba518c83354cf7ad12"
    ),
    native_qualification_clip_normalized_actions=False,
    native_qualification_continuity_prefix_rows=6,
)

VLA_BUNDLE_PROFILES: Mapping[str, VlaBundleProfile] = {
    profile.model_id: profile
    for profile in (
        LEGACY_VLA_BUNDLE_PROFILE,
        STAIRSBLOCKS_VLA_BUNDLE_PROFILE,
    )
}

# Compatibility aliases for callers and tests that still construct the original
# bundle explicitly. Production model selection is path-driven through the
# immutable profile registry above.
MODEL_ID = LEGACY_VLA_BUNDLE_PROFILE.model_id
CHECKPOINT_STEP = LEGACY_VLA_BUNDLE_PROFILE.checkpoint_step
MODEL_SHA256 = LEGACY_VLA_BUNDLE_PROFILE.model_sha256
RUN_CONFIG_SHA256 = LEGACY_VLA_BUNDLE_PROFILE.run_config_sha256
ARGV_SHA256 = LEGACY_VLA_BUNDLE_PROFILE.argv_sha256
STATS_ID = LEGACY_VLA_BUNDLE_PROFILE.stats_id
STATS_SHA256 = LEGACY_VLA_BUNDLE_PROFILE.stats_sha256
BASE_VLM_TREE_SHA256 = LEGACY_VLA_BUNDLE_PROFILE.base_vlm_tree_sha256
PSI_SOURCE_TREE_SHA256 = LEGACY_VLA_BUNDLE_PROFILE.psi_source_tree_sha256


def vla_bundle_profile(model_id: str) -> VlaBundleProfile:
    """Return one externally pinned profile, rejecting unknown model IDs."""
    try:
        return VLA_BUNDLE_PROFILES[model_id]
    except KeyError as error:
        supported = ", ".join(sorted(VLA_BUNDLE_PROFILES))
        raise ValueError(
            f"unattested VLA model ID {model_id!r}; supported: {supported}"
        ) from error


def vla_bundle_profile_for_model_root(model_root: str | Path) -> VlaBundleProfile:
    """Select a profile only from an exact ``models/<model_id>`` path."""
    path = Path(model_root).expanduser()
    if path.parent.name != "models":
        raise ValueError("VLA model root must use the exact models/<model_id> layout")
    return vla_bundle_profile(path.name)


class CanonicalTreeSnapshot(NamedTuple):
    """One canonical tree digest plus the digests of its admitted files."""

    sha256: str
    file_sha256_by_path: Mapping[str, str]


def file_sha256(path: Path) -> str:
    """Hash one regular, non-symlink file without loading it all into memory."""
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"VLA asset is missing or not regular: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_tree_sha256(
    root: Path,
    *,
    format_name: str,
    suffix: str | None = None,
) -> str:
    """Hash a deterministic path/size/content manifest for one asset tree."""
    return canonical_tree_snapshot(
        root,
        format_name=format_name,
        suffix=suffix,
    ).sha256


def canonical_tree_snapshot(
    root: Path,
    *,
    format_name: str,
    suffix: str | None = None,
) -> CanonicalTreeSnapshot:
    """Read one tree into a canonical digest and immutable expected file hashes."""
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    files: list[dict[str, object]] = []
    file_sha256_by_path: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"VLA asset tree must not contain symlinks: {path}")
        if not path.is_file() or (suffix is not None and path.suffix != suffix):
            continue
        relative_path = path.relative_to(root).as_posix()
        content = path.read_bytes()
        content_sha256 = hashlib.sha256(content).hexdigest()
        files.append(
            {
                "path": relative_path,
                "size": len(content),
                "sha256": content_sha256,
            }
        )
        file_sha256_by_path[relative_path] = content_sha256
    if not files:
        raise ValueError(f"VLA asset tree is empty: {root}")
    manifest = json.dumps(
        {"format": format_name, "files": files},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return CanonicalTreeSnapshot(
        sha256=hashlib.sha256(manifest).hexdigest(),
        file_sha256_by_path=file_sha256_by_path,
    )


def _require_digest(path: Path, expected: str, *, label: str) -> None:
    if _SHA256.fullmatch(expected) is None:
        raise ValueError(f"invalid expected SHA256 for {label}")
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA256 mismatch: expected {expected}, got {actual}")


@dataclass(frozen=True)
class VlaSourceBundle:
    """Verified paths for the exact model, processor, stats, and Psi source."""

    policy_eval_root: Path
    model_root: Path
    checkpoint_path: Path
    stats_path: Path
    psi_source_root: Path
    profile: VlaBundleProfile = LEGACY_VLA_BUNDLE_PROFILE

    @classmethod
    def verify(
        cls,
        policy_eval_root: str | Path,
        *,
        model_id: str = MODEL_ID,
    ) -> "VlaSourceBundle":
        """Resolve and attest one registered bundle below a policy-eval root."""
        root = Path(policy_eval_root).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise NotADirectoryError(root)
        profile = vla_bundle_profile(model_id)
        model_root = root / "models" / profile.model_id
        checkpoint = (
            model_root
            / "checkpoints"
            / f"ckpt_{profile.checkpoint_step}"
            / "model.safetensors"
        )
        stats = model_root / "stats" / "meta" / profile.stats_id
        psi_source = root / "src" / "psi"
        _require_digest(checkpoint, profile.model_sha256, label="VLA model checkpoint")
        _require_digest(
            model_root / "run_config.json",
            profile.run_config_sha256,
            label="VLA run config",
        )
        _require_digest(model_root / "argv.txt", profile.argv_sha256, label="VLA argv")
        _require_digest(stats, profile.stats_sha256, label="VLA normalization stats")
        base_vlm_digest = canonical_tree_sha256(
            model_root / "base_vlm",
            format_name="wenhao-base-vlm-tree.v1",
        )
        if base_vlm_digest != profile.base_vlm_tree_sha256:
            raise ValueError(
                "VLA base-VLM processor tree SHA256 mismatch: expected "
                f"{profile.base_vlm_tree_sha256}, got {base_vlm_digest}"
            )
        source_digest = canonical_tree_sha256(
            psi_source,
            format_name="wenhao-psi-source-tree.v1",
            suffix=".py",
        )
        if source_digest != profile.psi_source_tree_sha256:
            raise ValueError(
                "VLA Psi source tree SHA256 mismatch: expected "
                f"{profile.psi_source_tree_sha256}, got {source_digest}"
            )
        return cls(
            policy_eval_root=root,
            model_root=model_root,
            checkpoint_path=checkpoint,
            stats_path=stats,
            psi_source_root=psi_source,
            profile=profile,
        )

    @classmethod
    def verify_model_root(cls, model_root: str | Path) -> "VlaSourceBundle":
        """Attest the exact model root selected by ``policy.model.path``."""
        path = Path(model_root).expanduser().resolve(strict=True)
        profile = vla_bundle_profile_for_model_root(path)
        bundle = cls.verify(path.parent.parent, model_id=profile.model_id)
        if bundle.model_root != path:
            raise ValueError("VLA model path differs from its verified bundle root")
        return bundle
