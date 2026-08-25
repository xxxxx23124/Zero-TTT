"""Explicit gradient-group clipping and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from zero_ttt.model import ModelParameterGroup


class NonFiniteGradientError(FloatingPointError):
    def __init__(self, group_name: str) -> None:
        super().__init__(f"non-finite {group_name} gradient")
        self.group_name = group_name


@dataclass(frozen=True, slots=True)
class GradientNorms:
    base: float
    hypernet: float | None


def _clip_group(group: ModelParameterGroup, max_norm: float) -> float:
    parameters = tuple(
        parameter for parameter in group.parameters if parameter.grad is not None
    )
    if not parameters:
        return 0.0
    norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm)
    if not torch.isfinite(norm):
        raise NonFiniteGradientError(group.name)
    return float(norm)


def clip_model_gradients(
    groups: tuple[ModelParameterGroup, ...],
    *,
    base_max_norm: float,
    hypernet_max_norm: float,
) -> GradientNorms:
    """Clip every declared model parameter exactly once by component."""

    by_name = {group.name: group for group in groups}
    if len(by_name) != len(groups) or "base" not in by_name:
        raise ValueError("model parameter groups must contain one unique base group")
    unknown = set(by_name) - {"base", "hypernet"}
    if unknown:
        raise ValueError(f"no gradient policy for groups: {', '.join(sorted(unknown))}")
    base_norm = _clip_group(by_name["base"], base_max_norm)
    hypernet_norm = (
        _clip_group(by_name["hypernet"], hypernet_max_norm)
        if "hypernet" in by_name
        else None
    )
    return GradientNorms(base=base_norm, hypernet=hypernet_norm)
