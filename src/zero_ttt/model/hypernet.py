"""Shared per-position low-rank hypernetwork branch."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from zero_ttt.model.components import initialize_linear
from zero_ttt.model.interfaces import BlockResidualOutput, BlockResidualPlugin
from zero_ttt.model.norm import StableRMSNorm


def scale_gradient(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    detached = tensor.detach()
    return detached + scale * (tensor - detached)


class SharedDynamicLowRank(BlockResidualPlugin):
    optimization_group = "hypernet"

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        rank: int,
        hidden_dim: int,
        n_layers: int,
        context_gradient_scale: float,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.rank = rank
        self.context_gradient_scale = context_gradient_scale
        self.context_norm = StableRMSNorm(d_model)
        self.context_projection = nn.Linear(d_model, hidden_dim)
        self.layer_embedding = nn.Embedding(n_layers, hidden_dim)
        self.a_head = nn.Linear(hidden_dim, d_model * rank)
        self.b_head = nn.Linear(hidden_dim, rank * d_ff)
        initialize_linear(self.context_projection)
        nn.init.normal_(self.layer_embedding.weight, mean=0.0, std=0.02)
        initialize_linear(self.a_head)
        nn.init.zeros_(self.b_head.weight)
        nn.init.zeros_(self.b_head.bias)

    def forward(
        self,
        hidden: torch.Tensor,
        static_output: torch.Tensor,
        context: torch.Tensor,
        layer_selector: torch.Tensor,
    ) -> BlockResidualOutput:
        """Return a dynamic residual and raw A/B saturation fractions."""

        batch = hidden.shape[0]
        context = scale_gradient(context, self.context_gradient_scale)
        encoded = self.context_projection(self.context_norm(context))
        layer = torch.matmul(layer_selector, self.layer_embedding.weight).to(dtype=encoded.dtype)
        encoded = F.silu(encoded + layer)
        raw_a = torch.tanh(self.a_head(encoded))
        raw_b = torch.tanh(self.b_head(encoded))
        a = raw_a.view(batch, self.d_model, self.rank)
        b = raw_b.view(batch, self.rank, self.d_ff)
        low_rank = torch.einsum("bnf,brf->bnr", hidden, b)
        dynamic = torch.einsum("bnr,bdr->bnd", low_rank, a) / math.sqrt(self.rank)
        with torch.no_grad():
            a_saturation = (raw_a.abs() >= 0.95).float().mean()
            b_saturation = (raw_b.abs() >= 0.95).float().mean()
            dynamic_rms = dynamic.float().square().mean().sqrt()
            static_rms = static_output.float().square().mean().sqrt()
        return BlockResidualOutput(
            dynamic,
            a_saturation,
            b_saturation,
            dynamic_rms,
            static_rms,
        )
