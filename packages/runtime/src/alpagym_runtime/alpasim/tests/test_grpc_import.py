# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the local AlpaSim gRPC source and humanoid ABI gate."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from alpagym_runtime.alpasim.grpc_import import (
    _validate_humanoid_policy_camera_abi,
    _validate_descriptor_fields,
    ensure_alpasim_grpc_source,
    ensure_humanoid_policy_camera_abi,
)


def _message(*fields: str, nested: dict[str, object] | None = None) -> object:
    return SimpleNamespace(
        fields_by_name={field: SimpleNamespace(message_type=None) for field in fields},
        nested_types_by_name=nested or {},
    )


def _strict_camera_abi() -> tuple[object, object]:
    camera_spec = _message(
        "schema",
        "logical_id",
        "width",
        "height",
        "image_format",
        "max_frame_age_us",
        "contract_sha256",
    )
    camera_image = _message(
        "frame_start_us",
        "frame_end_us",
        "image_bytes",
        "logical_id",
        "env_id",
        "render_timestamp_us",
        "observation_decision_id",
        "render_qpos",
        "render_state_sha256",
        "camera_contract_sha256",
        "image_sha256",
        "render_receipt_sha256",
    )
    session = _message("policy_camera_spec")
    observation = _message("camera_images")
    session.fields_by_name["policy_camera_spec"].message_type = camera_spec
    observation.fields_by_name["camera_images"].message_type = camera_image
    descriptor = SimpleNamespace(
        message_types_by_name={
            "HumanoidPolicySessionRequest": session,
            "HumanoidPolicyCameraSpec": camera_spec,
            "HumanoidObservation": observation,
            "HumanoidCameraImage": camera_image,
        }
    )
    contracts = SimpleNamespace(
        HUMANOID_RENDER_STATE_SCHEMA="humanoid_render_state_qpos.v1",
        HUMANOID_RENDER_RECEIPT_SCHEMA="humanoid_render_receipt.v1",
        HumanoidRenderState=lambda **kwargs: kwargs,
        humanoid_image_sha256=lambda image_bytes: image_bytes,
        humanoid_render_receipt_sha256=lambda **kwargs: kwargs,
    )
    return descriptor, contracts


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


def test_strict_policy_camera_abi_accepts_matching_wire_and_shared_contracts() -> None:
    descriptor, contracts = _strict_camera_abi()

    _validate_humanoid_policy_camera_abi(descriptor, contracts)


def test_strict_policy_camera_abi_rejects_missing_image_receipt_field() -> None:
    descriptor, contracts = _strict_camera_abi()
    del descriptor.message_types_by_name["HumanoidCameraImage"].fields_by_name[
        "render_receipt_sha256"
    ]

    with pytest.raises(RuntimeError, match="render_receipt_sha256"):
        _validate_humanoid_policy_camera_abi(descriptor, contracts)


def test_strict_policy_camera_abi_rejects_mismatched_message_link() -> None:
    descriptor, contracts = _strict_camera_abi()
    descriptor.message_types_by_name["HumanoidPolicySessionRequest"].fields_by_name[
        "policy_camera_spec"
    ].message_type = descriptor.message_types_by_name["HumanoidCameraImage"]

    with pytest.raises(RuntimeError, match="matching message type"):
        _validate_humanoid_policy_camera_abi(descriptor, contracts)


def test_strict_policy_camera_gate_explains_matching_package_requirement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor, contracts = _strict_camera_abi()
    del contracts.humanoid_render_receipt_sha256
    humanoid_pb2 = SimpleNamespace(DESCRIPTOR=descriptor)

    def import_module(name: str) -> object:
        if name.endswith("humanoid_pb2"):
            return humanoid_pb2
        if name.endswith("humanoid_contracts"):
            return contracts
        raise AssertionError(name)

    monkeypatch.setattr(
        "alpagym_runtime.alpasim.grpc_import.importlib.import_module",
        import_module,
    )
    with pytest.raises(RuntimeError, match="ALPASIM_GRPC_ROOT"):
        ensure_humanoid_policy_camera_abi()
