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
    "HumanoidSessionAbortRequest": frozenset({"session_uuid"}),
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
            "decode_context",
        }
    ),
    "HumanoidMotionReferenceSpec": frozenset({"decode_context_schema"}),
    "HumanoidReferenceDecodeContext": frozenset(
        {"schema", "chunk_base_quaternion_wxyz", "local_xy_from_frame_zero"}
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

_HUMANOID_REQUIRED_SERVICE_METHOD_INPUTS: Mapping[str, Mapping[str, str]] = {
    "HumanoidPolicyService": {
        "abort_session": "HumanoidSessionAbortRequest",
    },
    "HumanoidDynamicsService": {
        "abort_session": "HumanoidSessionAbortRequest",
    },
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
            "scene_fingerprint",
            "model_signature_sha256",
            "camera_to_world_sha256",
            "renderer_binding_sha256",
        }
    ),
}

_HUMANOID_POLICY_CAMERA_EVIDENCE_FIELD_NUMBERS: Mapping[str, int] = {
    "scene_fingerprint": 13,
    "model_signature_sha256": 14,
    "camera_to_world_sha256": 15,
    "renderer_binding_sha256": 16,
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

    Portable runs use the exact ``alpasim-grpc`` revision in ``uv.lock``.
    ``ALPASIM_GRPC_ROOT`` is an explicit override for a matching local AlpaSim
    source tree. Formal local-repository runs set it in their recorded launch
    environment and validate the imported module origins and descriptors before
    starting runtime services.
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
            "renderer evidence fields, and alpasim_grpc.v0.humanoid_contracts. "
            "Install the matching "
            "AlpaSim gRPC package or set ALPASIM_GRPC_ROOT to its src/grpc "
            "directory."
        ) from exc


def ensure_humanoid_reference_decode_context_abi() -> None:
    """Require the protobuf links and shared source-hash contract used by plans."""
    try:
        humanoid_pb2 = importlib.import_module("alpasim_grpc.v0.humanoid_pb2")
        humanoid_contracts = importlib.import_module(
            "alpasim_grpc.v0.humanoid_contracts"
        )
        _validate_humanoid_reference_decode_context_abi(
            humanoid_pb2.DESCRIPTOR,
            humanoid_contracts,
        )
    except (AttributeError, ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "humanoid motion-reference planning requires a matching alpasim-grpc "
            "build with decode-context fields and the shared source-hash helper. "
            "Install the matching AlpaSim gRPC package or set ALPASIM_GRPC_ROOT "
            "to its src/grpc directory."
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
    camera_image = messages["HumanoidCameraImage"]
    for (
        field_name,
        expected_number,
    ) in _HUMANOID_POLICY_CAMERA_EVIDENCE_FIELD_NUMBERS.items():
        actual_number = getattr(camera_image.fields_by_name[field_name], "number", None)
        if actual_number != expected_number:
            raise RuntimeError(
                f"{source} is incompatible with the strict humanoid wire ABI: "
                f"field {field_name!r} must use protobuf tag {expected_number}, "
                f"got {actual_number!r}"
            )
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
        "HUMANOID_RENDER_RECEIPT_V2_SCHEMA": "humanoid_render_receipt.v2",
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
        "humanoid_render_receipt_v2_sha256",
    ):
        if not callable(getattr(humanoid_contracts, name, None)):
            raise RuntimeError(
                f"alpasim_grpc.v0.humanoid_contracts is missing callable {name}"
            )


def _validate_humanoid_reference_decode_context_abi(
    descriptor: Any,
    humanoid_contracts: Any,
) -> None:
    """Validate plan decode-context protobuf links and shared hash domains."""
    source = "alpasim_grpc.v0.humanoid_pb2"
    requirements = {
        "HumanoidMotionReferenceSpec": frozenset({"decode_context_schema"}),
        "HumanoidPlanUpdate": frozenset({"decode_context"}),
        "HumanoidReferenceDecodeContext": frozenset(
            {"schema", "chunk_base_quaternion_wxyz", "local_xy_from_frame_zero"}
        ),
    }
    _validate_descriptor_fields(descriptor, requirements, source=source)
    messages = descriptor.message_types_by_name
    _validate_message_field_type(
        owner=messages["HumanoidPlanUpdate"],
        field_name="decode_context",
        expected=messages["HumanoidReferenceDecodeContext"],
        source=source,
    )
    schema = getattr(
        humanoid_contracts,
        "HUMANOID_FULL_ROTATION_LOCAL_XY_DECODE_CONTEXT_SCHEMA",
        None,
    )
    if not isinstance(schema, str) or not schema:
        raise RuntimeError(
            "alpasim_grpc.v0.humanoid_contracts has an incompatible "
            "HUMANOID_FULL_ROTATION_LOCAL_XY_DECODE_CONTEXT_SCHEMA"
        )
    for name in (
        "HUMANOID_REFERENCE_HASH_DOMAIN_V1",
        "HUMANOID_REFERENCE_HASH_DOMAIN_V2",
    ):
        if not isinstance(getattr(humanoid_contracts, name, None), bytes):
            raise RuntimeError(
                f"alpasim_grpc.v0.humanoid_contracts has an incompatible {name}"
            )
    if not callable(getattr(humanoid_contracts, "humanoid_reference_sha256", None)):
        raise RuntimeError(
            "alpasim_grpc.v0.humanoid_contracts is missing callable "
            "humanoid_reference_sha256"
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
            f"{source} is incompatible with the strict humanoid wire ABI: "
            f"field {field_name!r} does not reference the matching message type"
        )


def _validate_humanoid_grpc_abi() -> None:
    """Fail closed when generated protos cannot carry correct PPO transitions."""
    humanoid_pb2 = importlib.import_module("alpasim_grpc.v0.humanoid_pb2")
    humanoid_contracts = importlib.import_module("alpasim_grpc.v0.humanoid_contracts")
    runtime_pb2 = importlib.import_module("alpasim_grpc.v0.runtime_pb2")
    _validate_descriptor_fields(
        humanoid_pb2.DESCRIPTOR,
        _HUMANOID_REQUIRED_FIELDS,
        source="alpasim_grpc.v0.humanoid_pb2",
    )
    _validate_descriptor_service_methods(
        humanoid_pb2.DESCRIPTOR,
        _HUMANOID_REQUIRED_SERVICE_METHOD_INPUTS,
        source="alpasim_grpc.v0.humanoid_pb2",
    )
    _validate_humanoid_reference_decode_context_abi(
        humanoid_pb2.DESCRIPTOR,
        humanoid_contracts,
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


def _validate_descriptor_service_methods(
    descriptor: Any,
    requirements: Mapping[str, Mapping[str, str]],
    *,
    source: str,
) -> None:
    """Require lifecycle RPCs and bind each one to its exact request type."""
    messages = descriptor.message_types_by_name
    services = descriptor.services_by_name
    for service_name, required_methods in requirements.items():
        service = services.get(service_name)
        if service is None:
            raise RuntimeError(
                f"{source} is incompatible with the humanoid PPO ABI: "
                f"missing service {service_name!r}. Use the matching AlpaSim "
                "checkout or set ALPASIM_GRPC_ROOT to its src/grpc directory."
            )
        for method_name, request_name in required_methods.items():
            method = service.methods_by_name.get(method_name)
            if method is None:
                raise RuntimeError(
                    f"{source} is incompatible with the humanoid PPO ABI: service "
                    f"{service_name!r} is missing method {method_name!r}. Use the "
                    "matching AlpaSim checkout or set ALPASIM_GRPC_ROOT to its "
                    "src/grpc directory."
                )
            expected_input = messages.get(request_name)
            if expected_input is None or method.input_type is not expected_input:
                raise RuntimeError(
                    f"{source} is incompatible with the humanoid PPO ABI: service "
                    f"{service_name!r} method {method_name!r} must accept "
                    f"{request_name!r}"
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
