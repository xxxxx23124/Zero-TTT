"""Narrow extension points used by the Transformer backbone."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import NamedTuple

import torch
from torch import nn


class BlockResidualOutput(NamedTuple):
    residual: torch.Tensor
    a_saturation: torch.Tensor
    b_saturation: torch.Tensor
    dynamic_rms: torch.Tensor
    static_rms: torch.Tensor


class BlockOutput(NamedTuple):
    hidden: torch.Tensor
    a_saturation: torch.Tensor
    b_saturation: torch.Tensor
    dynamic_rms: torch.Tensor
    static_rms: torch.Tensor


class BlockResidualPlugin(nn.Module, ABC):
    """Produce one additive board residual after a block FFN."""

    optimization_group: str | None = None

    @abstractmethod
    def forward(
        self,
        intermediate: torch.Tensor,
        static_output: torch.Tensor,
        context: torch.Tensor,
        layer_selector: torch.Tensor,
    ) -> BlockResidualOutput:
        raise NotImplementedError


class NoOpBlockResidualPlugin(BlockResidualPlugin):
    """Explicit disabled implementation with no parameters."""

    def forward(
        self,
        intermediate: torch.Tensor,
        static_output: torch.Tensor,
        context: torch.Tensor,
        layer_selector: torch.Tensor,
    ) -> BlockResidualOutput:
        del intermediate, context, layer_selector
        zero = static_output.new_zeros((), dtype=torch.float32)
        return BlockResidualOutput(torch.zeros_like(static_output), zero, zero, zero, zero)


class DepthMixer(nn.Module, ABC):
    """Retain and mix raw hidden states after Transformer blocks."""

    @abstractmethod
    def should_retain(self, depth: int) -> bool:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        depth: int,
        raw_states: dict[int, torch.Tensor],
        current: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError
