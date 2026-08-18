# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Safe tensor conversion helpers for planner runtime boundaries."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def as_float32_tensor(
    value: Any,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Convert to float32 without aliasing non-writable NumPy storage."""
    if isinstance(value, np.ndarray) and not value.flags.writeable:
        value = value.copy()
    return torch.as_tensor(value, dtype=torch.float32, device=device)
