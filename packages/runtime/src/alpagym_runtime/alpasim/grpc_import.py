# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for using local AlpaSim generated gRPC sources during prototyping."""

from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path
from typing import Any, Mapping


_HUMANOID_REQUIRED_FIELDS: Mapping[str, frozenset[str]] = {
    "HumanoidPolicySessionRequest": frozenset(
        {
            "joint_names",
            "observation_schema",
            "action_schema",
            "observation_terms",
            "attempt_id",
            "scene_id",
            "scenario_id",
            "random_seed",
            "execution_mode",
            "reference_spec",
        }
    ),
    "HumanoidEnvState": frozenset(
        {"timestamp_us", "named_observations", "observation_schema", "reset_id"}
    ),
    "HumanoidPolicyRequest": frozenset(
        {"bootstrap_only", "bootstrap_env_ids", "request_kind", "observation"}
    ),
    "HumanoidObservation": frozenset(
        {"camera_images", "decision_id", "feedback_traces", "timestamp_us"}
    ),
    "HumanoidPolicyResponse": frozenset(
        {"behavior_policy_version", "value_estimates", "plan_updates"}
    ),
    "HumanoidPlanUpdate": frozenset(
        {
            "reference_id",
            "source_decision_id",
            "frames",
            "reference_sha256",
            "root_z_alignment_offset_m",
        }
    ),
    "HumanoidRealizedControlTick": frozenset(
        {
            "control_tick_offset",
            "state",
            "active_reference_id",
            "reference_action_index",
            "active_reference_sha256",
            "applied_reference_sha256",
            "root_z_alignment_offset_m",
            "reward",
            "terminated",
            "truncated",
            "metrics",
            "control_episode_step",
        }
    ),
    "HumanoidRealizedFeedbackTrace": frozenset(
        {"env_id", "source_decision_id", "ticks"}
    ),
    "HumanoidStepResult": frozenset(
        {
            "state",
            "reward",
            "terminated",
            "truncated",
            "final_state",
            "episode_step",
            "control_ticks",
            "executed_control_ticks",
            "active_reference_sha256",
            "applied_reference_sha256",
        }
    ),
}

_HUMANOID_POLICY_CAMERA_REQUIRED_FIELDS: Mapping[str, frozenset[str]] = {
    "HumanoidPolicySessionRequest": frozenset({"policy_camera_spec"}),
    "HumanoidPolicyCameraSpec": frozenset(
        {
            "schema",
            "logical_id",
            "width",
            "height",
            "image_format",
            "max_frame_age_us",
            "contract_sha256",
        }
    ),
    "HumanoidObservation": frozenset({"camera_images"}),
    "HumanoidCameraImage": frozenset(
        {
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
        }
    ),
}

_HUMANOID_REQUIRED_ENUM_VALUES: Mapping[str, frozenset[str]] = {
    "HumanoidExecutionMode": frozenset(
        {
            "HUMANOID_EXECUTION_MODE_DIRECT_ACTION",
            "HUMANOID_EXECUTION_MODE_MOTION_REFERENCE",
        }
    ),
    "HumanoidPolicyRequestKind": frozenset(
        {
            "HUMANOID_POLICY_REQUEST_KIND_DIRECT_ACTION",
            "HUMANOID_POLICY_REQUEST_KIND_INITIAL_PLAN",
            "HUMANOID_POLICY_REQUEST_KIND_REPLAN_WITH_FEEDBACK",
            "HUMANOID_POLICY_REQUEST_KIND_FINALIZE_WITH_FEEDBACK",
        }
    ),
}

_RUNTIME_REQUIRED_FIELDS: Mapping[str, frozenset[str]] = {
    "RolloutSpec": frozenset(
        {
            "scenario_id",
            "session_uuids",
            "random_seed",
            "attempt_ids",
            "scene_id",
            "expected_behavior_policy_version",
        }
    ),
    "SimulationReturn.RolloutReturn": frozenset({"behavior_policy_version"}),
}


def ensure_alpasim_grpc_source(root: str | Path | None = None) -> None:
    """Optionally prepend an explicit local AlpaSim gRPC development tree.

    Normal runs use the exact ``alpasim-grpc`` revision in ``uv.lock``.
    ``ALPASIM_GRPC_ROOT`` is an opt-in override for testing uncommitted generated
    protobuf sources; the host lifecycle never sets it automatically.
    """
    configured_root = root if root is not None else os.environ.get("ALPASIM_GRPC_ROOT")
    if configured_root is None:
        return
    root_path = Path(configured_root).expanduser().resolve()
    v0_dir = root_path / "alpasim_grpc" / "v0"
    if not (v0_dir / "humanoid_pb2.py").is_file():
        raise RuntimeError(
            "explicit AlpaSim gRPC source is missing "
            f"{v0_dir / 'humanoid_pb2.py'}; point ALPASIM_GRPC_ROOT at "
            "the matching AlpaSim src/grpc directory"
        )

    root_str = str(root_path)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    package_dir = root_path / "alpasim_grpc"
    _ensure_package_path("alpasim_grpc", package_dir)
    _ensure_package_path("alpasim_grpc.v0", v0_dir)
    _validate_humanoid_grpc_abi()


def ensure_humanoid_policy_camera_abi() -> None:
    """Require the exact opt-in wire/shared ABI used by strict VLA RGB input.

    Generic humanoid policies intentionally keep working with the base camera
    packet. A VLA policy server must opt in to this gate at construction time;
    this prevents an older protobuf parser from silently discarding
    ``policy_camera_spec`` and downgrading the session to legacy camera routing.
    """
    try:
        humanoid_pb2 = importlib.import_module("alpasim_grpc.v0.humanoid_pb2")
        humanoid_contracts = importlib.import_module(
            "alpasim_grpc.v0.humanoid_contracts"
        )
        _validate_humanoid_policy_camera_abi(
            humanoid_pb2.DESCRIPTOR,
            humanoid_contracts,
        )
    except (AttributeError, ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "strict humanoid policy camera requires a matching alpasim-grpc "
            "build with HumanoidPolicyCameraSpec, image/render receipt fields, "
            "and alpasim_grpc.v0.humanoid_contracts. Install the matching "
            "AlpaSim gRPC package or set ALPASIM_GRPC_ROOT to its src/grpc "
            "directory."
        ) from exc


def _validate_humanoid_policy_camera_abi(
    descriptor: Any,
    humanoid_contracts: Any,
) -> None:
    """Validate strict policy-camera protobuf links and shared receipt helpers."""
    source = "alpasim_grpc.v0.humanoid_pb2"
    _validate_descriptor_fields(
        descriptor,
        _HUMANOID_POLICY_CAMERA_REQUIRED_FIELDS,
        source=source,
    )
    messages = descriptor.message_types_by_name
    _validate_message_field_type(
        owner=messages["HumanoidPolicySessionRequest"],
        field_name="policy_camera_spec",
        expected=messages["HumanoidPolicyCameraSpec"],
        source=source,
    )
    _validate_message_field_type(
        owner=messages["HumanoidObservation"],
        field_name="camera_images",
        expected=messages["HumanoidCameraImage"],
        source=source,
    )

    expected_schemas = {
        "HUMANOID_RENDER_STATE_SCHEMA": "humanoid_render_state_qpos.v1",
        "HUMANOID_RENDER_RECEIPT_SCHEMA": "humanoid_render_receipt.v1",
    }
    for name, expected in expected_schemas.items():
        if getattr(humanoid_contracts, name, None) != expected:
            raise RuntimeError(
                f"alpasim_grpc.v0.humanoid_contracts has an incompatible {name}"
            )
    for name in (
        "HumanoidRenderState",
        "humanoid_image_sha256",
        "humanoid_render_receipt_sha256",
    ):
        if not callable(getattr(humanoid_contracts, name, None)):
            raise RuntimeError(
                f"alpasim_grpc.v0.humanoid_contracts is missing callable {name}"
            )


def _validate_message_field_type(
    *,
    owner: Any,
    field_name: str,
    expected: Any,
    source: str,
) -> None:
    """Require a protobuf message field to point at the matching message type."""
    field = owner.fields_by_name[field_name]
    actual = getattr(field, "message_type", None)
    if actual is not expected:
        raise RuntimeError(
            f"{source} is incompatible with the strict humanoid policy-camera ABI: "
            f"field {field_name!r} does not reference the matching message type"
        )


def _validate_humanoid_grpc_abi() -> None:
    """Fail closed when generated protos cannot carry correct PPO transitions."""
    humanoid_pb2 = importlib.import_module("alpasim_grpc.v0.humanoid_pb2")
    runtime_pb2 = importlib.import_module("alpasim_grpc.v0.runtime_pb2")
    _validate_descriptor_fields(
        humanoid_pb2.DESCRIPTOR,
        _HUMANOID_REQUIRED_FIELDS,
        source="alpasim_grpc.v0.humanoid_pb2",
    )
    _validate_descriptor_enums(
        humanoid_pb2.DESCRIPTOR,
        _HUMANOID_REQUIRED_ENUM_VALUES,
        source="alpasim_grpc.v0.humanoid_pb2",
    )
    _validate_descriptor_fields(
        runtime_pb2.DESCRIPTOR,
        _RUNTIME_REQUIRED_FIELDS,
        source="alpasim_grpc.v0.runtime_pb2",
    )


def _validate_descriptor_fields(
    descriptor: Any,
    requirements: Mapping[str, frozenset[str]],
    *,
    source: str,
) -> None:
    """Require named protobuf messages and fields, including nested messages."""
    for qualified_name, required_fields in requirements.items():
        parts = qualified_name.split(".")
        message = descriptor.message_types_by_name.get(parts[0])
        for nested_name in parts[1:]:
            message = (
                None
                if message is None
                else message.nested_types_by_name.get(nested_name)
            )
        if message is None:
            raise RuntimeError(
                f"{source} is incompatible with the humanoid PPO ABI: "
                f"missing message {qualified_name!r}. Use the matching AlpaSim checkout "
                "or set ALPASIM_GRPC_ROOT to its src/grpc directory."
            )
        actual_fields = set(message.fields_by_name)
        missing_fields = required_fields - actual_fields
        if missing_fields:
            raise RuntimeError(
                f"{source} is incompatible with the humanoid PPO ABI: message "
                f"{qualified_name!r} is missing fields {sorted(missing_fields)}. "
                "Use the matching AlpaSim checkout or set ALPASIM_GRPC_ROOT to its "
                "src/grpc directory."
            )


def _validate_descriptor_enums(
    descriptor: Any,
    requirements: Mapping[str, frozenset[str]],
    *,
    source: str,
) -> None:
    """Require lifecycle enum values used by the motion-reference RPC."""
    for enum_name, required_values in requirements.items():
        enum = descriptor.enum_types_by_name.get(enum_name)
        if enum is None:
            raise RuntimeError(
                f"{source} is incompatible with the humanoid motion-reference ABI: "
                f"missing enum {enum_name!r}"
            )
        actual = set(enum.values_by_name)
        missing = required_values - actual
        if missing:
            raise RuntimeError(
                f"{source} is incompatible with the humanoid motion-reference ABI: enum "
                f"{enum_name!r} is missing values {sorted(missing)}"
            )


def _ensure_package_path(module_name: str, package_dir: Path) -> None:
    module = sys.modules.get(module_name)
    package_path = str(package_dir)
    if module is None:
        module = types.ModuleType(module_name)
        module.__file__ = str(package_dir / "__init__.py")
        module.__path__ = [package_path]
        sys.modules[module_name] = module
        return
    paths = list(getattr(module, "__path__", []))
    if package_path not in paths:
        paths.insert(0, package_path)
        module.__path__ = paths
