"""RMSNorm with an explicit epsilon under the fixed FP32 policy."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class StableRMSNorm(nn.RMSNorm):
    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        super().__init__(normalized_shape, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
