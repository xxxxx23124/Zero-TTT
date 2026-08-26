from __future__ import annotations

from dataclasses import replace

import torch

from zero_ttt.config import load_config
from zero_ttt.model.contracts import ModelDiagnostics, ModelOutput
from zero_ttt.training.losses import TrainingTargets, compute_losses


def model_output() -> ModelOutput:
    zero = torch.zeros(())
    return ModelOutput(
        policy_logits=torch.tensor([[1.0, -1.0], [0.5, -0.5]]),
        value=torch.tensor([[0.25], [9.0]]),
        ownership=torch.tensor([[0.0, 0.5], [8.0, -8.0]]),
        score_margin=torch.tensor([[20.0], [300.0]]),
        diagnostics=ModelDiagnostics(zero, zero, zero, zero),
    )


def targets() -> TrainingTargets:
    return TrainingTargets(
        policy=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        value=torch.tensor([0.0, -9.0]),
        ownership=torch.tensor([[0.0, 1.0], [-8.0, 8.0]]),
        score_margin=torch.tensor([0.0, -300.0]),
        value_mask=torch.tensor([1.0, 0.0]),
        ownership_mask=torch.tensor([1.0, 0.0]),
        score_mask=torch.tensor([1.0, 0.0]),
    )


def test_masked_auxiliary_targets_do_not_change_losses() -> None:
    config = load_config("configs/test.toml")
    original = targets()
    changed = replace(
        original,
        value=torch.tensor([0.0, 1_000.0]),
        ownership=torch.tensor([[0.0, 1.0], [1_000.0, -1_000.0]]),
        score_margin=torch.tensor([0.0, 1_000.0]),
    )
    expected = compute_losses(model_output(), original, config.training)
    actual = compute_losses(model_output(), changed, config.training)

    assert torch.equal(actual.value, expected.value)
    assert torch.equal(actual.ownership, expected.ownership)
    assert torch.equal(actual.score, expected.score)
    assert torch.equal(actual.total, expected.total)


def test_all_zero_auxiliary_masks_produce_finite_zero_components() -> None:
    config = load_config("configs/test.toml")
    target = targets()
    masked = replace(
        target,
        value_mask=torch.zeros_like(target.value_mask),
        ownership_mask=torch.zeros_like(target.ownership_mask),
        score_mask=torch.zeros_like(target.score_mask),
    )
    losses = compute_losses(model_output(), masked, config.training)

    assert losses.value.item() == 0.0
    assert losses.ownership.item() == 0.0
    assert losses.score.item() == 0.0
    assert torch.isfinite(losses.total)
