"""Transformer blocks and backbone orchestration."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from zero_ttt.config import ExecutionConfig, ModelConfig
from zero_ttt.model.components import initialize_linear
from zero_ttt.model.contracts import BackboneOutput, ModelDiagnostics
from zero_ttt.model.execution import BlockExecutor
from zero_ttt.model.interfaces import (
    BlockOutput,
    BlockResidualPlugin,
    DepthMixer,
)
from zero_ttt.model.norm import StableRMSNorm
from zero_ttt.model.rope import AxialRoPE2D
from zero_ttt.model.tokens import TokenLayout


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig, residual_std: float) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.output = nn.Linear(config.d_model, config.d_model, bias=False)
        self.q_norm = StableRMSNorm(self.head_dim)
        self.k_norm = StableRMSNorm(self.head_dim)
        initialize_linear(self.qkv)
        initialize_linear(self.output, std=residual_std)

    def forward(
        self,
        hidden: torch.Tensor,
        rope: AxialRoPE2D,
    ) -> torch.Tensor:
        batch, tokens, _ = hidden.shape
        qkv = self.qkv(hidden).view(
            batch,
            tokens,
            3,
            self.n_heads,
            self.head_dim,
        )
        query, key, value = qkv.unbind(dim=2)
        query = self.q_norm(query).transpose(1, 2)
        key = self.k_norm(key).transpose(1, 2)
        value = value.transpose(1, 2)
        query, key = rope(query, key)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(batch, tokens, self.d_model)
        return self.output(attended)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        layout: TokenLayout,
        *,
        layer_index: int,
        plugin_enabled: bool,
        residual_std: float,
    ) -> None:
        super().__init__()
        self.layout = layout
        self.plugin_enabled = plugin_enabled
        selector = torch.zeros(config.n_layers, dtype=torch.float32)
        selector[layer_index] = 1.0
        self.register_buffer("layer_selector", selector, persistent=False)
        self.attention_norm = StableRMSNorm(config.d_model)
        self.attention = MultiHeadSelfAttention(config, residual_std)
        self.ffn_norm = StableRMSNorm(config.d_model)
        self.ffn_up = nn.Linear(config.d_model, 2 * config.d_ff, bias=False)
        self.ffn_down = nn.Linear(config.d_ff, config.d_model, bias=False)
        initialize_linear(self.ffn_up)
        initialize_linear(self.ffn_down, std=residual_std)

    def forward(
        self,
        hidden: torch.Tensor,
        rope: AxialRoPE2D,
        plugin: BlockResidualPlugin,
    ) -> BlockOutput:
        hidden = hidden + self.attention(self.attention_norm(hidden), rope)
        normalized = self.ffn_norm(hidden)
        gate, values = self.ffn_up(normalized).chunk(2, dim=-1)
        intermediate = F.silu(gate) * values
        output = self.ffn_down(intermediate)
        zero = output.new_zeros((), dtype=torch.float32)
        diagnostics = (zero, zero, zero, zero)
        if self.plugin_enabled:
            plugin_output = plugin(
                self.layout.board(intermediate),
                self.layout.board(output),
                self.layout.summary(normalized),
                self.layer_selector,
            )
            board_output = self.layout.board(output) + plugin_output.residual
            output = torch.cat((board_output, self.layout.special(output)), dim=1)
            diagnostics = plugin_output[1:]
        return BlockOutput(hidden + output, *diagnostics)


class TransformerBackbone(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        execution: ExecutionConfig,
        layout: TokenLayout,
        block_plugin: BlockResidualPlugin,
        depth_mixer: DepthMixer,
    ) -> None:
        super().__init__()
        self.layout = layout
        self.block_plugin = block_plugin
        self.depth_mixer = depth_mixer
        self.executor = BlockExecutor(execution)
        self.rope = AxialRoPE2D(
            config.rope,
            config.d_model // config.n_heads,
            layout,
            board_size=int(math.isqrt(layout.board_tokens)),
        )
        plugin_start = config.n_layers - config.hypernet.num_layers
        residual_std = 0.02 / math.sqrt(2 * config.n_layers)
        self.blocks = nn.ModuleList(
            TransformerBlock(
                config,
                layout,
                layer_index=index,
                plugin_enabled=config.hypernet.enabled and index >= plugin_start,
                residual_std=residual_std,
            )
            for index in range(config.n_layers)
        )
        self.final_norm = StableRMSNorm(config.d_model)

    def forward(self, hidden: torch.Tensor) -> BackboneOutput:
        raw_states: dict[int, torch.Tensor] = {}
        if self.depth_mixer.should_retain(0):
            raw_states[0] = hidden
        diagnostic_totals = [hidden.new_zeros((), dtype=torch.float32) for _ in range(4)]
        diagnostic_count = 0
        for index, block in enumerate(self.blocks):
            result = self.executor.run(
                index=index,
                training=self.training,
                block=block,
                rope=self.rope,
                plugin=self.block_plugin,
                hidden=hidden,
            )
            raw_hidden = result.hidden
            depth = index + 1
            if self.depth_mixer.should_retain(depth):
                raw_states[depth] = raw_hidden
            hidden = self.depth_mixer(depth, raw_states, raw_hidden)
            if block.plugin_enabled:
                diagnostic_totals = [
                    total + value
                    for total, value in zip(
                        diagnostic_totals,
                        result[1:],
                        strict=True,
                    )
                ]
                diagnostic_count += 1
        hidden = self.final_norm(hidden)
        if diagnostic_count:
            diagnostic_totals = [value / diagnostic_count for value in diagnostic_totals]
        return BackboneOutput(
            hidden=hidden,
            diagnostics=ModelDiagnostics(*diagnostic_totals),
        )

    def compile_blocks(self, *, dynamic: bool, mode: str) -> None:
        """Compile blocks independently to bound AOTAutograd buffer lifetimes."""

        for block in self.blocks:
            block.compile(dynamic=dynamic, mode=mode)
