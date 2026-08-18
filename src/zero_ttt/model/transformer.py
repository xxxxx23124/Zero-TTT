"""Policy-value Transformer for 19x19 Go."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from zero_ttt.config import ModelConfig
from zero_ttt.game.rules import ACTION_SIZE, BOARD_AREA, BOARD_SIZE
from zero_ttt.model.hypernet import DynamicLowRank
from zero_ttt.model.norm import StableRMSNorm
from zero_ttt.model.rope import AxialRoPE2D


@dataclass(frozen=True, slots=True)
class ModelOutput:
    policy_logits: torch.Tensor
    value: torch.Tensor
    ownership: torch.Tensor
    score_margin: torch.Tensor


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig, rope: AxialRoPE2D) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.rope = rope
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.output = nn.Linear(config.d_model, config.d_model, bias=False)
        self.q_norm = StableRMSNorm(self.head_dim)
        self.k_norm = StableRMSNorm(self.head_dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
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
        query, key = self.rope(query, key)
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
        rope: AxialRoPE2D,
        use_hypernet: bool,
    ) -> None:
        super().__init__()
        self.attention_norm = StableRMSNorm(config.d_model)
        self.attention = MultiHeadSelfAttention(config, rope)
        self.ffn_norm = StableRMSNorm(config.d_model)
        self.ffn_up = nn.Linear(config.d_model, 2 * config.d_ff, bias=False)
        self.ffn_down = nn.Linear(config.d_ff, config.d_model, bias=False)
        hyper = config.hypernet
        self.hypernet = (
            DynamicLowRank(
                d_model=config.d_model,
                d_ff=config.d_ff,
                rank=hyper.rank,
                hidden_dim=hyper.hidden_dim,
                context_gradient_scale=hyper.context_gradient_scale,
            )
            if use_hypernet
            else None
        )

    def forward(self, hidden: torch.Tensor, hypernet_scale: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.attention(self.attention_norm(hidden))
        normalized = self.ffn_norm(hidden)
        gate, values = self.ffn_up(normalized).chunk(2, dim=-1)
        intermediate = F.silu(gate) * values
        output = self.ffn_down(intermediate)
        if self.hypernet is not None:
            dynamic = self.hypernet(intermediate[:, :BOARD_AREA], normalized[:, -1])
            board_output = output[:, :BOARD_AREA] + hypernet_scale * dynamic
            output = torch.cat((board_output, output[:, BOARD_AREA:]), dim=1)
        return hidden + output


class PolicyValueTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.point_projection = nn.Linear(config.input_planes, config.d_model)
        self.global_projection = nn.Linear(config.global_features, config.d_model, bias=False)
        self.cls_token = nn.Parameter(torch.empty(config.d_model))
        rope = AxialRoPE2D(config.rope, config.d_model // config.n_heads, BOARD_SIZE)
        hyper_start = config.n_layers - config.hypernet.num_layers
        self.blocks = nn.ModuleList(
            TransformerBlock(
                config,
                rope,
                use_hypernet=config.hypernet.enabled and index >= hyper_start,
            )
            for index in range(config.n_layers)
        )
        self.final_norm = StableRMSNorm(config.d_model)
        self.point_policy = nn.Linear(config.d_model, 1)
        self.pass_policy = nn.Linear(config.d_model, 1)
        head_hidden = max(config.d_model // 2, 32)
        self.value_head = nn.Sequential(
            nn.Linear(config.d_model, head_hidden),
            nn.SiLU(),
            nn.Linear(head_hidden, 1),
        )
        self.score_head = nn.Sequential(
            nn.Linear(config.d_model, head_hidden),
            nn.SiLU(),
            nn.Linear(head_hidden, 1),
        )
        self.ownership_head = nn.Linear(config.d_model, 1)
        self.register_buffer("hypernet_scale", torch.zeros((), dtype=torch.float32), persistent=True)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        residual_std = 0.02 / math.sqrt(2 * self.config.n_layers)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for block in self.blocks:
            nn.init.normal_(block.attention.output.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.ffn_down.weight, mean=0.0, std=residual_std)
            if block.hypernet is not None:
                nn.init.zeros_(block.hypernet.b_head.weight)
                nn.init.zeros_(block.hypernet.b_head.bias)

    def set_hypernet_scale(self, value: float) -> None:
        self.hypernet_scale.fill_(value)

    def hypernet_parameters(self) -> Iterator[nn.Parameter]:
        for block in self.blocks:
            if block.hypernet is not None:
                yield from block.hypernet.parameters()

    def base_parameters(self) -> Iterator[nn.Parameter]:
        hyper_ids = {id(parameter) for parameter in self.hypernet_parameters()}
        for parameter in self.parameters():
            if id(parameter) not in hyper_ids:
                yield parameter

    def _run_block(self, block: TransformerBlock, hidden: torch.Tensor) -> torch.Tensor:
        return block(hidden, self.hypernet_scale)

    def forward(
        self,
        board_features: torch.Tensor,
        global_features: torch.Tensor,
        legal_mask: torch.Tensor,
    ) -> ModelOutput:
        if board_features.ndim != 4 or board_features.shape[1:] != (
            self.config.input_planes,
            BOARD_SIZE,
            BOARD_SIZE,
        ):
            raise ValueError("board_features has the wrong shape")
        if global_features.shape != (board_features.shape[0], self.config.global_features):
            raise ValueError("global_features has the wrong shape")
        if legal_mask.shape != (board_features.shape[0], ACTION_SIZE):
            raise ValueError("legal_mask has the wrong shape")
        points = board_features.flatten(2).transpose(1, 2)
        points = self.point_projection(points)
        cls = self.cls_token[None, :] + self.global_projection(global_features)
        hidden = torch.cat((points, cls[:, None, :]), dim=1)
        for index, block in enumerate(self.blocks):
            should_checkpoint = (
                self.training
                and self.config.activation_checkpoint
                and index % self.config.checkpoint_every == 0
            )
            if should_checkpoint:
                hidden = checkpoint(
                    self._run_block,
                    block,
                    hidden,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                hidden = block(hidden, self.hypernet_scale)
        hidden = self.final_norm(hidden)
        board_hidden = hidden[:, :BOARD_AREA]
        cls_hidden = hidden[:, -1]
        point_logits = self.point_policy(board_hidden).squeeze(-1)
        pass_logit = self.pass_policy(cls_hidden)
        policy_logits = torch.cat((point_logits, pass_logit), dim=-1)
        policy_logits = policy_logits.masked_fill(~legal_mask.bool(), -torch.inf)
        value = torch.tanh(self.value_head(cls_hidden))
        ownership = torch.tanh(self.ownership_head(board_hidden).squeeze(-1))
        score_margin = 400.0 * torch.tanh(self.score_head(cls_hidden))
        return ModelOutput(
            policy_logits=policy_logits,
            value=value,
            ownership=ownership,
            score_margin=score_margin,
        )
