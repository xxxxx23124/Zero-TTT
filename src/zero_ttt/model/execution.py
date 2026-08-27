"""Execution policy for Transformer blocks."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from zero_ttt.config import ExecutionConfig
from zero_ttt.model.interfaces import BlockOutput, BlockResidualPlugin
from zero_ttt.model.rope import AxialRoPE2D


class BlockExecutor:
    def __init__(self, config: ExecutionConfig) -> None:
        self.config = config

    @staticmethod
    def _run(
        block: nn.Module,
        rope: AxialRoPE2D,
        plugin: BlockResidualPlugin,
        hidden: torch.Tensor,
    ) -> BlockOutput:
        return block(hidden, rope, plugin)

    def run(
        self,
        *,
        index: int,
        training: bool,
        block: nn.Module,
        rope: AxialRoPE2D,
        plugin: BlockResidualPlugin,
        hidden: torch.Tensor,
    ) -> BlockOutput:
        should_checkpoint = (
            training
            and self.config.activation_checkpoint
            and index % self.config.activation_checkpoint_stride == 0
        )
        if not should_checkpoint:
            return self._run(block, rope, plugin, hidden)
        return cast(
            BlockOutput,
            checkpoint(
                self._run,
                block,
                rope,
                plugin,
                hidden,
                use_reentrant=False,
                preserve_rng_state=True,
            ),
        )
