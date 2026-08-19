# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free canonical identities for safetensors parameter subsets."""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

ACTOR_STATE_HASH_SCHEMA = "videomimic_planner_actor_state.safetensors.v1"
_PLANNER_ACTOR_ROOTS = frozenset(
    ("actor", "actor_terrain", "actor_attention", "actor_context", "std")
)


def canonical_safetensors_subset_sha256(
    payload: bytes,
    *,
    tensor_names: Collection[str],
) -> str:
    """Hash named tensors independently of safetensors header/layout ordering."""

    header, data = _decode_safetensors(payload)
    names = tuple(sorted(set(tensor_names)))
    if not names:
        raise ValueError("canonical safetensors subset must not be empty")
    missing = tuple(name for name in names if name not in header)
    if missing:
        raise ValueError(
            "safetensors payload is missing identity tensors: " + ", ".join(missing)
        )
    tensors = [_tensor_identity(name, header[name], data) for name in names]
    encoded = json.dumps(
        {"schema": ACTOR_STATE_HASH_SCHEMA, "tensors": tensors},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def planner_actor_state_sha256(path: Path) -> str:
    """Hash the canonical VideoMimic PPO actor subset in one safetensors file."""

    payload = path.read_bytes()
    header, _data = _decode_safetensors(payload)
    actor_names = tuple(
        name
        for name in header
        if name != "__metadata__" and _is_planner_actor_parameter(name)
    )
    return canonical_safetensors_subset_sha256(
        payload,
        tensor_names=actor_names,
    )


def _is_planner_actor_parameter(name: str) -> bool:
    root = name.split(".", 1)[0]
    return root in _PLANNER_ACTOR_ROOTS


def _decode_safetensors(payload: bytes) -> tuple[Mapping[str, Any], memoryview]:
    if len(payload) < 8:
        raise ValueError("safetensors payload is shorter than its header length")
    header_length = struct.unpack("<Q", payload[:8])[0]
    header_end = 8 + header_length
    if header_length == 0 or header_end > len(payload):
        raise ValueError("safetensors header length is invalid")
    try:
        header: Any = json.loads(payload[8:header_end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("safetensors header is not valid JSON") from exc
    if not isinstance(header, Mapping):
        raise ValueError("safetensors header must contain an object")
    return header, memoryview(payload)[header_end:]


def _tensor_identity(
    name: str,
    raw_metadata: Any,
    data: memoryview,
) -> dict[str, Any]:
    if not isinstance(raw_metadata, Mapping):
        raise ValueError(f"safetensors tensor {name!r} has invalid metadata")
    dtype = raw_metadata.get("dtype")
    shape = raw_metadata.get("shape")
    offsets = raw_metadata.get("data_offsets")
    if not isinstance(dtype, str) or not dtype:
        raise ValueError(f"safetensors tensor {name!r} has invalid dtype")
    if not isinstance(shape, list) or any(
        isinstance(size, bool) or not isinstance(size, int) or size < 0
        for size in shape
    ):
        raise ValueError(f"safetensors tensor {name!r} has invalid shape")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or any(
            isinstance(offset, bool) or not isinstance(offset, int)
            for offset in offsets
        )
    ):
        raise ValueError(f"safetensors tensor {name!r} has invalid data offsets")
    start, end = offsets
    if start < 0 or start > end or end > len(data):
        raise ValueError(f"safetensors tensor {name!r} escapes its data section")
    return {
        "name": name,
        "dtype": dtype,
        "shape": shape,
        "data_sha256": hashlib.sha256(data[start:end]).hexdigest(),
    }


__all__ = (
    "ACTOR_STATE_HASH_SCHEMA",
    "canonical_safetensors_subset_sha256",
    "planner_actor_state_sha256",
)
