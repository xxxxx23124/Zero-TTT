"""Sparse DenseFormer depth-weighted averaging."""

from __future__ import annotations

import torch
from torch import nn

from zero_ttt.config import DepthMixingConfig
from zero_ttt.model.interfaces import DepthMixer


class IdentityDepthMixer(DepthMixer):
    """Explicit disabled implementation."""

    def should_retain(self, depth: int) -> bool:
        del depth
        return False

    def forward(
        self,
        depth: int,
        raw_states: dict[int, torch.Tensor],
        current: torch.Tensor,
    ) -> torch.Tensor:
        del depth, raw_states
        return current


class SparseDepthWeightedAverage(DepthMixer):
    """Mix raw block outputs at configured depths with identity initialization."""

    def __init__(self, n_layers: int, config: DepthMixingConfig) -> None:
        super().__init__()
        self.dilation = config.dilation
        self.period = config.period
        self.mixing_depths = tuple(range(config.period, n_layers + 1, config.period))
        self.source_depths = {
            depth: tuple(range(depth % config.dilation, depth + 1, config.dilation))
            for depth in self.mixing_depths
        }
        self.retained_depths = frozenset(
            source for sources in self.source_depths.values() for source in sources
        )
        self.weights = nn.ParameterDict()
        for depth, sources in self.source_depths.items():
            values = torch.zeros(len(sources), dtype=torch.float32)
            values[-1] = 1.0
            self.weights[str(depth)] = nn.Parameter(values)

    def should_retain(self, depth: int) -> bool:
        return depth in self.retained_depths

    def forward(
        self,
        depth: int,
        raw_states: dict[int, torch.Tensor],
        current: torch.Tensor,
    ) -> torch.Tensor:
        if depth not in self.source_depths:
            return current
        stacked = torch.stack(
            [raw_states[source] for source in self.source_depths[depth]],
            dim=0,
        )
        weights = self.weights[str(depth)]
        return torch.tensordot(weights, stacked, dims=1)
