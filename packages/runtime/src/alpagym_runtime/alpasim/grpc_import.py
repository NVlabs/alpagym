# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for using local AlpaSim generated gRPC sources during prototyping."""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path


DEFAULT_ALPASIM_GRPC_ROOT = Path("/home/yuxiaoc/repos/alpasim/src/grpc")


def ensure_alpasim_grpc_source(root: str | Path | None = None) -> None:
    """Prepend a local AlpaSim gRPC source tree when it has humanoid protos.

    The released ``alpasim-grpc`` package currently used by AlpaGym can lag the
    local AlpaSim checkout during humanoid prototyping. When
    ``ALPASIM_GRPC_ROOT`` or the workstation default points at generated sources
    containing ``humanoid_pb2.py``, expose that source tree before importing
    ``alpasim_grpc.v0.*`` modules. AV-only environments without the local source
    tree continue to use the installed package.
    """
    root_path = Path(root or os.environ.get("ALPASIM_GRPC_ROOT") or DEFAULT_ALPASIM_GRPC_ROOT)
    v0_dir = root_path / "alpasim_grpc" / "v0"
    if not (v0_dir / "humanoid_pb2.py").is_file():
        return

    root_str = str(root_path)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    package_dir = root_path / "alpasim_grpc"
    _ensure_package_path("alpasim_grpc", package_dir)
    _ensure_package_path("alpasim_grpc.v0", v0_dir)


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
