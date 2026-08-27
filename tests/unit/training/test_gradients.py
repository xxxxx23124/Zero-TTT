from __future__ import annotations

import torch
from torch import nn

from zero_ttt.config import load_config
from zero_ttt.model import PolicyValueTransformer
from zero_ttt.training.gradients import clip_model_gradients, parameters_are_finite


def test_parameter_finiteness_check_detects_nan_and_infinity() -> None:
    parameter = nn.Parameter(torch.tensor([1.0, -2.0]))
    assert parameters_are_finite((parameter,))
    with torch.no_grad():
        parameter[0] = torch.nan
    assert not parameters_are_finite((parameter,))
    with torch.no_grad():
        parameter[0] = torch.inf
    assert not parameters_are_finite((parameter,))


def test_parameter_groups_are_complete_disjoint_and_clipped_once() -> None:
    config = load_config("configs/test.toml")
    model = PolicyValueTransformer(config.model, config.execution)
    groups = model.parameter_groups()
    assert {group.name for group in groups} == {"base", "hypernet"}
    grouped_ids = [id(parameter) for group in groups for parameter in group.parameters]
    trainable_ids = [id(parameter) for parameter in model.parameters() if parameter.requires_grad]
    assert len(grouped_ids) == len(set(grouped_ids))
    assert set(grouped_ids) == set(trainable_ids)
    base = next(group for group in groups if group.name == "base")
    assert any(parameter is model.encoder.summary_token for parameter in base.no_decay)

    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    norms = clip_model_gradients(
        groups,
        base_max_norm=0.5,
        hypernet_max_norm=0.25,
    )
    assert norms.base > 0.5
    assert norms.hypernet is not None and norms.hypernet > 0.25
    hypernet = next(group for group in groups if group.name == "hypernet")
    for group, limit in ((base, 0.5), (hypernet, 0.25)):
        post_clip = torch.linalg.vector_norm(
            torch.stack(
                [
                    torch.linalg.vector_norm(parameter.grad)
                    for parameter in group.parameters
                    if parameter.grad is not None
                ]
            )
        )
        assert post_clip <= limit + 1e-5
