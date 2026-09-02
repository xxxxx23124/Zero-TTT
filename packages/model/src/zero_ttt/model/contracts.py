"""Typed contracts shared by policy-value model implementations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class ModelDiagnostics:
    """Scalar observability values produced alongside model predictions."""

    hyper_a_saturation: torch.Tensor
    hyper_b_saturation: torch.Tensor
    hyper_dynamic_rms: torch.Tensor
    hyper_static_rms: torch.Tensor


@dataclass(frozen=True, slots=True)
class ModelPredictions:
    """Unmasked task predictions decoded from the final hidden state."""

    policy_logits: torch.Tensor
    value: torch.Tensor
    ownership: torch.Tensor
    score_margin: torch.Tensor


@dataclass(frozen=True, slots=True)
class ModelOutput:
    policy_logits: torch.Tensor
    value: torch.Tensor
    ownership: torch.Tensor
    score_margin: torch.Tensor
    diagnostics: ModelDiagnostics


@dataclass(frozen=True, slots=True)
class BackboneOutput:
    hidden: torch.Tensor
    diagnostics: ModelDiagnostics


@dataclass(frozen=True, slots=True)
class ModelParameterGroup:
    """A validated optimizer group split by weight-decay behavior."""

    name: str
    decay: tuple[nn.Parameter, ...]
    no_decay: tuple[nn.Parameter, ...]

    @property
    def parameters(self) -> tuple[nn.Parameter, ...]:
        return self.decay + self.no_decay
