# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RLinf value head; callers must explicitly select the activation."""

from collections.abc import Sequence

import torch
import torch.nn as nn


class ValueHead(nn.Module):
    """Small MLP critic head vendored from RLinf."""

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Sequence[int] = (512, 128),
        output_dim: int = 1,
        activation: str = "gelu",
        bias_last: bool = False,
    ) -> None:
        """Build a value MLP with an explicitly selected activation.

        PyTorch's Kaiming initializer does not accept ``"gelu"`` as a gain
        name. GELU therefore uses the standard ReLU Kaiming approximation,
        while the forward activation remains exactly GELU.
        """
        super().__init__()

        layers: list[nn.Module] = []
        in_dim = input_dim

        if activation.lower() == "relu":
            act = nn.ReLU
        elif activation.lower() == "gelu":
            act = nn.GELU
        elif activation.lower() == "tanh":
            act = nn.Tanh
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(act())
            in_dim = h

        layers.append(nn.Linear(in_dim, output_dim, bias=bias_last))

        self.mlp = nn.Sequential(*layers)

        self._init_weights(activation.lower())

    def _init_weights(self, nonlinearity: str = "relu") -> None:
        """Initialize hidden layers and the scalar output head."""
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                if module is self.mlp[-1]:
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                else:
                    if nonlinearity in {"relu", "gelu"}:
                        nn.init.kaiming_normal_(
                            module.weight, mode="fan_out", nonlinearity="relu"
                        )
                    elif nonlinearity == "tanh":
                        nn.init.kaiming_normal_(
                            module.weight, mode="fan_out", nonlinearity="tanh"
                        )
                    else:  # Defensive for direct private-method callers.
                        raise ValueError(f"Unsupported nonlinearity: {nonlinearity}")
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return one value vector for each leading input row."""
        return self.mlp(x)
