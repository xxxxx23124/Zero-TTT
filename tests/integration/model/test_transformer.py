from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from zero_ttt.config import load_config
from zero_ttt.game.features import encode_position
from zero_ttt.game.rules import PASS_ACTION
from zero_ttt.game.state import GameState
from zero_ttt.model import PolicyValueTransformer
from zero_ttt.training.losses import TrainingTargets, compute_losses


def tensors_for_empty(batch: int = 2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    config = load_config("configs/test.toml")
    encoded = encode_position(GameState.new(config.game))
    board = torch.from_numpy(np.stack([encoded.board] * batch))
    global_features = torch.from_numpy(np.stack([encoded.global_features] * batch))
    legal = torch.from_numpy(np.stack([encoded.legal] * batch))
    return board, global_features, legal


def test_model_outputs_and_illegal_mask() -> None:
    config = load_config("configs/test.toml")
    model = PolicyValueTransformer(config.model, config.execution)
    board, global_features, legal = tensors_for_empty()
    legal[:, 0] = False
    output = model(board, global_features, legal)
    assert output.policy_logits.shape == (2, 362)
    assert output.value.shape == (2, 1)
    assert output.ownership.shape == (2, 361)
    assert output.score_margin.shape == (2, 1)
    assert output.diagnostics.hyper_a_saturation.ndim == 0
    assert output.diagnostics.hyper_b_saturation.ndim == 0
    assert torch.isneginf(output.policy_logits[:, 0]).all()
    assert torch.isfinite(output.policy_logits[:, PASS_ACTION]).all()


def test_losses_remain_finite_with_masked_logits() -> None:
    config = load_config("configs/test.toml")
    model = PolicyValueTransformer(config.model, config.execution)
    board, global_features, legal = tensors_for_empty()
    output = model(board, global_features, legal)
    policy = legal.float()
    policy /= policy.sum(dim=-1, keepdim=True)
    targets = TrainingTargets(
        policy=policy,
        value=torch.zeros(2),
        ownership=torch.zeros(2, 361),
        score_margin=torch.zeros(2),
        value_mask=torch.ones(2, dtype=torch.bool),
        ownership_mask=torch.ones(2, dtype=torch.bool),
        score_mask=torch.ones(2, dtype=torch.bool),
    )
    losses = compute_losses(output, targets, config.training)
    assert torch.isfinite(losses.total)
    losses.total.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_checkpointed_training_backward() -> None:
    config = load_config("configs/test.toml")
    execution = replace(
        config.execution,
        activation_checkpoint=True,
        activation_checkpoint_stride=1,
    )
    model = PolicyValueTransformer(config.model, execution).train()
    board, global_features, legal = tensors_for_empty(batch=1)
    output = model(board, global_features, legal)
    output.value.sum().backward()
    assert model.encoder.summary_token.grad is not None


def test_checkpoint_and_compile_match_eager_outputs_and_gradients() -> None:
    config = load_config("configs/test.toml")
    board, global_features, legal = tensors_for_empty(batch=1)
    eager = PolicyValueTransformer(config.model, config.execution).train()
    checkpointed = PolicyValueTransformer(
        config.model,
        replace(
            config.execution,
            activation_checkpoint=True,
            activation_checkpoint_stride=1,
        ),
    ).train()
    checkpointed.load_state_dict(eager.state_dict())

    eager_output = eager(board, global_features, legal)
    checkpoint_output = checkpointed(board, global_features, legal)
    for expected, actual in (
        (eager_output.policy_logits, checkpoint_output.policy_logits),
        (eager_output.value, checkpoint_output.value),
        (eager_output.ownership, checkpoint_output.ownership),
        (eager_output.score_margin, checkpoint_output.score_margin),
        (
            eager_output.diagnostics.hyper_a_saturation,
            checkpoint_output.diagnostics.hyper_a_saturation,
        ),
        (
            eager_output.diagnostics.hyper_b_saturation,
            checkpoint_output.diagnostics.hyper_b_saturation,
        ),
        (
            eager_output.diagnostics.hyper_dynamic_rms,
            checkpoint_output.diagnostics.hyper_dynamic_rms,
        ),
        (
            eager_output.diagnostics.hyper_static_rms,
            checkpoint_output.diagnostics.hyper_static_rms,
        ),
    ):
        assert torch.allclose(expected, actual, atol=1e-6, rtol=1e-5)
    eager_loss = (
        eager_output.policy_logits.sum()
        + eager_output.value.sum()
        + eager_output.ownership.sum()
        + eager_output.score_margin.sum()
    )
    checkpoint_loss = (
        checkpoint_output.policy_logits.sum()
        + checkpoint_output.value.sum()
        + checkpoint_output.ownership.sum()
        + checkpoint_output.score_margin.sum()
    )
    eager_loss.backward()
    checkpoint_loss.backward()
    eager_gradients = {name: parameter.grad for name, parameter in eager.named_parameters()}
    checkpoint_gradients = {
        name: parameter.grad for name, parameter in checkpointed.named_parameters()
    }
    assert eager_gradients.keys() == checkpoint_gradients.keys()
    for name in eager_gradients:
        expected = eager_gradients[name]
        actual = checkpoint_gradients[name]
        assert (expected is None) == (actual is None), name
        if expected is not None and actual is not None:
            assert torch.allclose(expected, actual, atol=1e-5, rtol=1e-4), name

    compiled_source = PolicyValueTransformer(config.model, config.execution).eval()
    compiled_source.load_state_dict(eager.state_dict())
    compiled = torch.compile(compiled_source, backend="eager", dynamic=False)
    with torch.no_grad():
        expected = compiled_source(board, global_features, legal).value
        actual = compiled(board, global_features, legal).value
    assert torch.allclose(expected, actual, atol=1e-6, rtol=1e-5)


def test_shared_hypernet_and_dwa_start_as_exact_baseline_and_receive_gradient() -> None:
    config = load_config("configs/test.toml")
    baseline_config = replace(
        config.model,
        hypernet=replace(config.model.hypernet, enabled=False),
        depth_mixing=replace(config.model.depth_mixing, enabled=False),
    )
    baseline = PolicyValueTransformer(baseline_config, config.execution).train()
    model = PolicyValueTransformer(config.model, config.execution).train()
    model.load_state_dict(baseline.state_dict(), strict=False)
    board, global_features, legal = tensors_for_empty(batch=1)
    expected = baseline(board, global_features, legal)
    actual = model(board, global_features, legal)
    assert torch.equal(expected.policy_logits, actual.policy_logits)
    assert torch.equal(expected.value, actual.value)
    assert sum(block.plugin_enabled for block in model.backbone.blocks) == 1
    assert [index for index, block in enumerate(model.backbone.blocks) if block.plugin_enabled] == [
        1
    ]
    assert actual.diagnostics.hyper_dynamic_rms.item() == 0.0
    (actual.policy_logits.sum() + actual.ownership.sum()).backward()
    assert model.block_plugin.b_head.weight.grad is not None
    assert torch.isfinite(model.block_plugin.b_head.weight.grad).all()
    assert model.block_plugin.b_head.weight.grad.norm() > 0
