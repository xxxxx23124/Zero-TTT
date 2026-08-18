from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from zero_ttt.config import load_config
from zero_ttt.game.features import encode_position
from zero_ttt.game.rules import BOARD_AREA, PASS_ACTION
from zero_ttt.game.state import GameState
from zero_ttt.model.rope import AxialRoPE2D
from zero_ttt.model.transformer import PolicyValueTransformer
from zero_ttt.training.losses import TrainingTargets, compute_losses


def tensors_for_empty(batch: int = 2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    config = load_config("configs/test.toml")
    encoded = encode_position(GameState.new(config.game))
    board = torch.from_numpy(np.stack([encoded.board] * batch))
    global_features = torch.from_numpy(np.stack([encoded.global_features] * batch))
    legal = torch.from_numpy(np.stack([encoded.legal] * batch))
    return board, global_features, legal


def test_centered_rope_leaves_center_and_cls_unrotated() -> None:
    config = load_config("configs/rtx4090l.toml")
    head_dim = config.model.d_model // config.model.n_heads
    rope = AxialRoPE2D(config.model.rope, head_dim)
    values = torch.randn(1, config.model.n_heads, BOARD_AREA + 1, head_dim)
    rotated = rope.apply(values)
    center = 9 * 19 + 9
    assert torch.equal(rotated[..., center, :], values[..., center, :])
    assert torch.equal(rotated[..., -1, :], values[..., -1, :])
    slowest_phase = 18.0 * float(rope.inv_freq[-1])
    assert 0.2 < slowest_phase < 0.3


def test_model_outputs_and_illegal_mask() -> None:
    config = load_config("configs/test.toml")
    model = PolicyValueTransformer(config.model)
    board, global_features, legal = tensors_for_empty()
    legal[:, 0] = False
    output = model(board, global_features, legal)
    assert output.policy_logits.shape == (2, 362)
    assert output.value.shape == (2, 1)
    assert output.ownership.shape == (2, 361)
    assert output.score_margin.shape == (2, 1)
    assert torch.isneginf(output.policy_logits[:, 0]).all()
    assert torch.isfinite(output.policy_logits[:, PASS_ACTION]).all()


def test_losses_remain_finite_with_masked_logits() -> None:
    config = load_config("configs/test.toml")
    model = PolicyValueTransformer(config.model)
    board, global_features, legal = tensors_for_empty()
    output = model(board, global_features, legal)
    policy = legal.float()
    policy /= policy.sum(dim=-1, keepdim=True)
    targets = TrainingTargets(
        policy=policy,
        value=torch.zeros(2),
        ownership=torch.zeros(2, 361),
        score_margin=torch.zeros(2),
        ownership_mask=torch.ones(2),
        score_mask=torch.ones(2),
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
    model_config = replace(config.model, activation_checkpoint=True, checkpoint_every=1)
    model = PolicyValueTransformer(model_config).train()
    board, global_features, legal = tensors_for_empty(batch=1)
    output = model(board, global_features, legal)
    output.value.sum().backward()
    assert model.cls_token.grad is not None


def test_checkpoint_and_compile_match_eager_outputs_and_gradients() -> None:
    config = load_config("configs/test.toml")
    board, global_features, legal = tensors_for_empty(batch=1)
    eager = PolicyValueTransformer(config.model).train()
    checkpointed = PolicyValueTransformer(
        replace(config.model, activation_checkpoint=True, checkpoint_every=1)
    ).train()
    checkpointed.load_state_dict(eager.state_dict())

    eager_value = eager(board, global_features, legal).value
    eager_value.sum().backward()
    checkpoint_value = checkpointed(board, global_features, legal).value
    checkpoint_value.sum().backward()
    assert torch.allclose(eager_value, checkpoint_value, atol=1e-6, rtol=1e-5)
    assert torch.allclose(
        eager.cls_token.grad,
        checkpointed.cls_token.grad,
        atol=1e-6,
        rtol=1e-5,
    )

    compiled_source = PolicyValueTransformer(config.model).eval()
    compiled_source.load_state_dict(eager.state_dict())
    compiled = torch.compile(compiled_source, backend="eager", dynamic=False)
    with torch.no_grad():
        expected = compiled_source(board, global_features, legal).value
        actual = compiled(board, global_features, legal).value
    assert torch.allclose(expected, actual, atol=1e-6, rtol=1e-5)


def test_hypernet_starts_as_exact_zero_residual_and_receives_gradient() -> None:
    config = load_config("configs/test.toml")
    hyper = replace(config.model.hypernet, enabled=True)
    model = PolicyValueTransformer(replace(config.model, hypernet=hyper)).train()
    board, global_features, legal = tensors_for_empty(batch=1)
    model.set_hypernet_scale(0.0)
    without_branch = model(board, global_features, legal).value.detach()
    model.set_hypernet_scale(1.0)
    with_branch = model(board, global_features, legal).value
    assert torch.equal(without_branch, with_branch.detach())
    with_branch.sum().backward()
    b_head = next(block.hypernet.b_head for block in model.blocks if block.hypernet is not None)
    assert b_head.weight.grad is not None
    assert torch.isfinite(b_head.weight.grad).all()
