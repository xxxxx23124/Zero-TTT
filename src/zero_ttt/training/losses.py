"""Policy, value, ownership, and score losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from zero_ttt.config import TrainingConfig
from zero_ttt.model import ModelOutput


@dataclass(frozen=True, slots=True)
class TrainingTargets:
    policy: torch.Tensor
    value: torch.Tensor
    ownership: torch.Tensor
    score_margin: torch.Tensor
    value_mask: torch.Tensor
    ownership_mask: torch.Tensor
    score_mask: torch.Tensor

    def __post_init__(self) -> None:
        floating = (self.policy, self.value, self.ownership, self.score_margin)
        if any(tensor.dtype != torch.float32 for tensor in floating):
            raise TypeError("training targets must have torch.float32 dtype")
        masks = (self.value_mask, self.ownership_mask, self.score_mask)
        if any(mask.dtype != torch.bool for mask in masks):
            raise TypeError("training target masks must have torch.bool dtype")


@dataclass(frozen=True, slots=True)
class LossOutput:
    total: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor
    ownership: torch.Tensor
    score: torch.Tensor


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=values.dtype)
    denominator = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denominator


def compute_losses(
    output: ModelOutput,
    target: TrainingTargets,
    config: TrainingConfig,
) -> LossOutput:
    predictions = (output.policy_logits, output.value, output.ownership, output.score_margin)
    if any(tensor.dtype != torch.float32 for tensor in predictions):
        raise TypeError("model loss inputs must have torch.float32 dtype")
    log_policy = F.log_softmax(output.policy_logits, dim=-1)
    policy_target = target.policy
    policy_terms = torch.where(
        policy_target > 0,
        policy_target * log_policy,
        torch.zeros_like(log_policy),
    )
    policy = -policy_terms.sum(dim=-1).mean()
    value_per_sample = F.mse_loss(
        output.value.squeeze(-1),
        target.value,
        reduction="none",
    )
    value = _masked_mean(value_per_sample, target.value_mask)
    ownership_per_point = F.smooth_l1_loss(
        output.ownership,
        target.ownership,
        reduction="none",
    ).mean(dim=-1)
    ownership = _masked_mean(ownership_per_point, target.ownership_mask)
    score_prediction = output.score_margin.squeeze(-1) / 400.0
    score_target = (target.score_margin / 400.0).clamp(-1.0, 1.0)
    score_per_sample = F.smooth_l1_loss(
        score_prediction,
        score_target,
        reduction="none",
    )
    score = _masked_mean(score_per_sample, target.score_mask)
    total = (
        config.policy_loss_weight * policy
        + config.value_loss_weight * value
        + config.ownership_loss_weight * ownership
        + config.score_loss_weight * score
    )
    return LossOutput(total=total, policy=policy, value=value, ownership=ownership, score=score)
