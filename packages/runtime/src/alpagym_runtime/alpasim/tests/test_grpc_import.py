# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the local AlpaSim gRPC source and humanoid ABI gate."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from alpagym_runtime.alpasim.grpc_import import (
    _validate_descriptor_fields,
    ensure_alpasim_grpc_source,
)


def _message(*fields: str, nested: dict[str, object] | None = None) -> object:
    return SimpleNamespace(
        fields_by_name={field: object() for field in fields},
        nested_types_by_name=nested or {},
    )


def test_no_user_specific_grpc_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unset override leaves the installed package alone."""
    monkeypatch.delenv("ALPASIM_GRPC_ROOT", raising=False)
    ensure_alpasim_grpc_source()


def test_explicit_missing_grpc_root_fails_closed(tmp_path: Path) -> None:
    """A typo in the explicit humanoid proto root cannot fall back to AV stubs."""
    with pytest.raises(RuntimeError, match="humanoid_pb2.py"):
        ensure_alpasim_grpc_source(tmp_path)


def test_descriptor_gate_accepts_required_nested_fields() -> None:
    descriptor = SimpleNamespace(
        message_types_by_name={
            "SimulationReturn": _message(
                nested={"RolloutReturn": _message("behavior_policy_version")}
            )
        }
    )
    _validate_descriptor_fields(
        descriptor,
        {"SimulationReturn.RolloutReturn": frozenset({"behavior_policy_version"})},
        source="runtime_pb2",
    )


def test_descriptor_gate_rejects_missing_bootstrap_field() -> None:
    descriptor = SimpleNamespace(
        message_types_by_name={"HumanoidStepResult": _message("reward", "terminated")}
    )
    with pytest.raises(RuntimeError, match="final_state"):
        _validate_descriptor_fields(
            descriptor,
            {"HumanoidStepResult": frozenset({"reward", "final_state"})},
            source="humanoid_pb2",
        )
