"""Input and output components for the Go policy-value model."""

from __future__ import annotations

import torch
from torch import nn
from zero_ttt.config import ModelConfig
from zero_ttt.model.contracts import ModelPredictions
from zero_ttt.model.tokens import TokenLayout


def initialize_linear(module: nn.Linear, *, std: float = 0.02) -> None:
    nn.init.normal_(module.weight, mean=0.0, std=std)
    bias = getattr(module, "bias", None)
    if bias is not None:
        nn.init.zeros_(bias)


class GoTokenEncoder(nn.Module):
    def __init__(self, config: ModelConfig, layout: TokenLayout) -> None:
        super().__init__()
        self.layout = layout
        self.point_projection = nn.Linear(config.input_planes, config.d_model)
        self.global_projection = nn.Linear(
            config.global_features,
            config.d_model,
            bias=False,
        )
        self.summary_token = nn.Parameter(torch.empty(config.d_model))
        initialize_linear(self.point_projection)
        initialize_linear(self.global_projection)
        nn.init.normal_(self.summary_token, mean=0.0, std=0.02)

    def forward(
        self,
        board_features: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        points = board_features.flatten(2).transpose(1, 2)
        if points.shape[1] != self.layout.board_tokens:
            raise ValueError("flattened board does not match the token layout")
        points = self.point_projection(points)
        summary = self.summary_token[None, :] + self.global_projection(global_features)
        hidden = torch.cat((points, summary[:, None, :]), dim=1)
        self.layout.validate(hidden)
        return hidden

    def no_weight_decay_parameters(self) -> tuple[nn.Parameter, ...]:
        return (self.summary_token,)


class PolicyValueHeads(nn.Module):
    def __init__(self, config: ModelConfig, layout: TokenLayout) -> None:
        super().__init__()
        self.layout = layout
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
        for module in self.modules():
            if isinstance(module, nn.Linear):
                initialize_linear(module)

    def forward(self, hidden: torch.Tensor) -> ModelPredictions:
        board_hidden = self.layout.board(hidden)
        summary_hidden = self.layout.summary(hidden)
        point_logits = self.point_policy(board_hidden).squeeze(-1)
        pass_logit = self.pass_policy(summary_hidden)
        return ModelPredictions(
            policy_logits=torch.cat((point_logits, pass_logit), dim=-1),
            value=torch.tanh(self.value_head(summary_hidden)),
            ownership=torch.tanh(self.ownership_head(board_hidden).squeeze(-1)),
            score_margin=400.0 * torch.tanh(self.score_head(summary_hidden)),
        )
