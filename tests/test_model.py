from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from zero_ttt.config import load_config
from zero_ttt.game.features import encode_position
from zero_ttt.game.rules import BOARD_AREA, PASS_ACTION
from zero_ttt.game.state import GameState
from zero_ttt.model.rope import AxialRoPE2D
from zero_ttt.model.depth_mixing import SparseDepthWeightedAverage
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
    assert output.hyper_a_saturation.ndim == 0
    assert output.hyper_b_saturation.ndim == 0
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


def test_shared_hypernet_and_dwa_start_as_exact_baseline_and_receive_gradient() -> None:
    config = load_config("configs/test.toml")
    baseline_config = replace(
        config.model,
        hypernet=replace(config.model.hypernet, enabled=False),
        depth_mixing=replace(config.model.depth_mixing, enabled=False),
    )
    baseline = PolicyValueTransformer(baseline_config).train()
    model = PolicyValueTransformer(config.model).train()
    model.load_state_dict(baseline.state_dict(), strict=False)
    board, global_features, legal = tensors_for_empty(batch=1)
    expected = baseline(board, global_features, legal)
    actual = model(board, global_features, legal)
    assert torch.equal(expected.policy_logits, actual.policy_logits)
    assert torch.equal(expected.value, actual.value)
    assert model.hypernet is not None
    assert sum(block.use_hypernet for block in model.blocks) == config.model.hypernet.num_layers
    assert [index for index, block in enumerate(model.blocks) if block.use_hypernet] == [1]
    assert actual.hyper_dynamic_rms.item() == 0.0
    (actual.policy_logits.sum() + actual.ownership.sum()).backward()
    assert model.hypernet.b_head.weight.grad is not None
    assert torch.isfinite(model.hypernet.b_head.weight.grad).all()
    assert model.hypernet.b_head.weight.grad.norm() > 0


def test_sparse_depth_weighted_average_sources_and_identity() -> None:
    config = load_config("configs/test.toml")
    depth_config = replace(config.model.depth_mixing, dilation=4, period=4)
    mixing = SparseDepthWeightedAverage(8, depth_config)
    assert mixing.source_depths == {4: (0, 4), 8: (0, 4, 8)}
    states = {
        0: torch.full((1, 2, 3), 1.0),
        4: torch.full((1, 2, 3), 4.0, requires_grad=True),
        8: torch.full((1, 2, 3), 8.0, requires_grad=True),
    }
    output = mixing(8, states, states[8])
    assert torch.equal(output, states[8])
    output.sum().backward()
    assert mixing.weights["8"].grad is not None
    assert torch.count_nonzero(mixing.weights["8"].grad) == 3
