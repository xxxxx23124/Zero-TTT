"""Per-position full low-rank hypernetwork branch."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from zero_ttt.model.norm import StableRMSNorm


def scale_gradient(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    detached = tensor.detach()
    return detached + scale * (tensor - detached)


class DynamicLowRank(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        rank: int,
        hidden_dim: int,
        context_gradient_scale: float,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.rank = rank
        self.context_gradient_scale = context_gradient_scale
        self.context_norm = StableRMSNorm(d_model)
        self.hidden = nn.Linear(d_model, hidden_dim)
        self.a_head = nn.Linear(hidden_dim, d_model * rank)
        self.b_head = nn.Linear(hidden_dim, rank * d_ff)
        nn.init.zeros_(self.b_head.weight)
        nn.init.zeros_(self.b_head.bias)

    def forward(self, hidden: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Return a dynamic residual for board tokens only."""

        batch = hidden.shape[0]
        context = scale_gradient(context, self.context_gradient_scale)
        encoded = F.silu(self.hidden(self.context_norm(context)))
        a = torch.tanh(self.a_head(encoded)).view(batch, self.d_model, self.rank)
        b = torch.tanh(self.b_head(encoded)).view(batch, self.rank, self.d_ff)
        a = a / math.sqrt(self.rank)
        b = b / math.sqrt(self.d_ff)
        low_rank = torch.einsum("bnf,brf->bnr", hidden, b)
        return torch.einsum("bnr,bdr->bnd", low_rank, a)
