# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Content-addressed loader boundary for the authoritative Wenhao Psi bundle."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, NamedTuple

MODEL_ID = "qwen3vl-wenhao2b-s1-step3203"
CHECKPOINT_STEP = 3203
MODEL_SHA256 = "5cfc977203c29e9495bf66d79b7a1c2e3862ad4ffdd9569ae616c5b4121cd1c1"
RUN_CONFIG_SHA256 = "9feafeb8d43ff840bb74b51d762c6aadea750532328f2e51a42057e737721ce0"
ARGV_SHA256 = "8896251cced8d97af6ea9c18bf84b82be9dd78184c4f854af1d56b1fac97311e"
STATS_ID = "stats_psi0.json"
STATS_SHA256 = "9db767e3a82fbb9ab990e7102bca96c44a278122329f7d5a8574124435333e37"
BASE_VLM_TREE_SHA256 = (
    "3b8ae84cbc35e9f802414820b4d00a5b1ad5533278c423739c9f274d3e302ded"
)
PSI_SOURCE_TREE_SHA256 = (
    "fc62e7b5ece8318851b6f7638ccc545c617f36e9bc40eca1c3b937f10e939018"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class CanonicalTreeSnapshot(NamedTuple):
    """One canonical tree digest plus the digests of its admitted files."""

    sha256: str
    file_sha256_by_path: Mapping[str, str]


def file_sha256(path: Path) -> str:
    """Hash one regular, non-symlink file without loading it all into memory."""
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Wenhao asset is missing or not regular: {path}")
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
            raise ValueError(f"Wenhao asset tree must not contain symlinks: {path}")
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
        raise ValueError(f"Wenhao asset tree is empty: {root}")
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
class WenhaoSourceBundle:
    """Verified paths for the exact model, processor, stats, and Psi source."""

    policy_eval_root: Path
    model_root: Path
    checkpoint_path: Path
    stats_path: Path
    psi_source_root: Path

    @classmethod
    def verify(cls, policy_eval_root: str | Path) -> "WenhaoSourceBundle":
        """Resolve and attest the current authoritative workstation bundle."""
        root = Path(policy_eval_root).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise NotADirectoryError(root)
        model_root = root / "models" / MODEL_ID
        checkpoint = (
            model_root / "checkpoints" / f"ckpt_{CHECKPOINT_STEP}" / "model.safetensors"
        )
        stats = model_root / "stats" / "meta" / STATS_ID
        psi_source = root / "src" / "psi"
        _require_digest(checkpoint, MODEL_SHA256, label="Wenhao model checkpoint")
        _require_digest(
            model_root / "run_config.json",
            RUN_CONFIG_SHA256,
            label="Wenhao run config",
        )
        _require_digest(model_root / "argv.txt", ARGV_SHA256, label="Wenhao argv")
        _require_digest(stats, STATS_SHA256, label="Wenhao normalization stats")
        base_vlm_digest = canonical_tree_sha256(
            model_root / "base_vlm",
            format_name="wenhao-base-vlm-tree.v1",
        )
        if base_vlm_digest != BASE_VLM_TREE_SHA256:
            raise ValueError(
                "Wenhao base-VLM processor tree SHA256 mismatch: expected "
                f"{BASE_VLM_TREE_SHA256}, got {base_vlm_digest}"
            )
        source_digest = canonical_tree_sha256(
            psi_source,
            format_name="wenhao-psi-source-tree.v1",
            suffix=".py",
        )
        if source_digest != PSI_SOURCE_TREE_SHA256:
            raise ValueError(
                "Wenhao Psi source tree SHA256 mismatch: expected "
                f"{PSI_SOURCE_TREE_SHA256}, got {source_digest}"
            )
        return cls(
            policy_eval_root=root,
            model_root=model_root,
            checkpoint_path=checkpoint,
            stats_path=stats,
            psi_source_root=psi_source,
        )
