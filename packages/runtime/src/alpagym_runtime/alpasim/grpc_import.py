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
        }
    ),
    "HumanoidEnvState": frozenset(
        {"timestamp_us", "named_observations", "observation_schema", "reset_id"}
    ),
    "HumanoidPolicyRequest": frozenset({"bootstrap_only", "bootstrap_env_ids"}),
    "HumanoidPolicyResponse": frozenset({"behavior_policy_version", "value_estimates"}),
    "HumanoidStepResult": frozenset(
        {"state", "reward", "terminated", "truncated", "final_state", "episode_step"}
    ),
}

_RUNTIME_REQUIRED_FIELDS: Mapping[str, frozenset[str]] = {
    "RolloutSpec": frozenset(
        {"scenario_id", "session_uuids", "random_seed", "attempt_ids", "scene_id"}
    ),
    "SimulationReturn.RolloutReturn": frozenset({"behavior_policy_version"}),
}


def ensure_alpasim_grpc_source(root: str | Path | None = None) -> None:
    """Prepend a local AlpaSim gRPC source tree when it has humanoid protos.

    The released ``alpasim-grpc`` package currently used by AlpaGym can lag the
    local AlpaSim checkout during humanoid prototyping. When
    ``ALPASIM_GRPC_ROOT`` points at generated sources
    containing ``humanoid_pb2.py``, expose that source tree before importing
    ``alpasim_grpc.v0.*`` modules. AV-only environments without the local source
    tree continue to use the installed package.
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


def _validate_humanoid_grpc_abi() -> None:
    """Fail closed when generated protos cannot carry correct PPO transitions."""
    humanoid_pb2 = importlib.import_module("alpasim_grpc.v0.humanoid_pb2")
    runtime_pb2 = importlib.import_module("alpasim_grpc.v0.runtime_pb2")
    _validate_descriptor_fields(
        humanoid_pb2.DESCRIPTOR,
        _HUMANOID_REQUIRED_FIELDS,
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
            message = None if message is None else message.nested_types_by_name.get(nested_name)
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
