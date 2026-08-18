"""RMSNorm with explicit epsilon and autocast-friendly weights."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class StableRMSNorm(nn.RMSNorm):
    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        super().__init__(normalized_shape, eps=eps)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        weight = None if self.weight is None else self.weight.to(dtype=hidden.dtype)
        return F.rms_norm(hidden, self.normalized_shape, weight, self.eps)
