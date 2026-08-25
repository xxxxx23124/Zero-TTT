from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from zero_ttt.config import load_config
from zero_ttt.game.features import encode_position
from zero_ttt.game.rules import BOARD_AREA, PASS_ACTION
from zero_ttt.game.state import GameState
from zero_ttt.model import PolicyValueTransformer, TokenLayout
from zero_ttt.model.base import BasePolicyValueModel
from zero_ttt.model.contracts import BackboneOutput, ModelDiagnostics, ModelPredictions
from zero_ttt.model.execution import BlockExecutor
from zero_ttt.model.interfaces import BlockOutput, NoOpBlockResidualPlugin
from zero_ttt.model.rope import AxialRoPE2D
from zero_ttt.model.depth_mixing import SparseDepthWeightedAverage
from zero_ttt.training.losses import TrainingTargets, compute_losses


def tensors_for_empty(batch: int = 2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    config = load_config("configs/test.toml")
    encoded = encode_position(GameState.new(config.game))
    board = torch.from_numpy(np.stack([encoded.board] * batch))
    global_features = torch.from_numpy(np.stack([encoded.global_features] * batch))
    legal = torch.from_numpy(np.stack([encoded.legal] * batch))
    return board, global_features, legal


def test_base_model_owns_macro_forward_order_and_masking() -> None:
    calls: list[str] = []

    class RecordingModel(BasePolicyValueModel):
        def __init__(self) -> None:
            super().__init__(
                input_planes=1,
                global_features=1,
                board_size=1,
                action_size=2,
            )
            self.weight = nn.Parameter(torch.ones(()))

        def _encode_tokens(self, board_features, global_features):
            calls.append("encode")
            return (board_features.flatten(1) + global_features)[:, :, None]

        def _forward_backbone(self, hidden):
            calls.append("backbone")
            zero = hidden.new_zeros(())
            return BackboneOutput(hidden, ModelDiagnostics(zero, zero, zero, zero))

        def _decode_outputs(self, hidden):
            calls.append("decode")
            value = hidden[:, 0] * self.weight
            return ModelPredictions(
                policy_logits=torch.cat((value, value), dim=-1),
                value=value,
                ownership=value,
                score_margin=value,
            )

        def _parameter_group_members(self):
            return {"base": (self.weight,)}

        def configure_execution(self, config):
            del config

        def compile_training_components(self, *, dynamic, mode):
            del dynamic, mode

    model = RecordingModel()
    output = model(
        torch.ones(1, 1, 1, 1),
        torch.ones(1, 1),
        torch.tensor([[False, True]]),
    )
    assert calls == ["encode", "backbone", "decode"]
    assert torch.isneginf(output.policy_logits[0, 0])
    assert torch.isfinite(output.policy_logits[0, 1])


def test_centered_rope_leaves_center_and_cls_unrotated() -> None:
    config = load_config("configs/rtx4090l.toml")
    head_dim = config.model.d_model // config.model.n_heads
    layout = TokenLayout(board_tokens=BOARD_AREA, special_tokens=1)
    rope = AxialRoPE2D(config.model.rope, head_dim, layout)
    values = torch.randn(1, config.model.n_heads, BOARD_AREA + 1, head_dim)
    rotated = rope.apply(values)
    center = 9 * 19 + 9
    assert torch.equal(rotated[..., center, :], values[..., center, :])
    assert torch.equal(rotated[..., -1, :], values[..., -1, :])
    slowest_phase = 18.0 * float(rope.inv_freq[-1])
    assert 0.2 < slowest_phase < 0.3


def test_token_layout_and_rope_support_multiple_special_tokens() -> None:
    config = load_config("configs/test.toml")
    layout = TokenLayout(board_tokens=BOARD_AREA, special_tokens=3, summary_offset=1)
    head_dim = config.model.d_model // config.model.n_heads
    rope = AxialRoPE2D(config.model.rope, head_dim, layout)
    values = torch.randn(1, config.model.n_heads, layout.total_tokens, head_dim)
    rotated = rope.apply(values)
    assert layout.board_slice == slice(0, BOARD_AREA)
    assert layout.special_slice == slice(BOARD_AREA, BOARD_AREA + 3)
    assert layout.summary_index == BOARD_AREA + 1
    assert torch.equal(layout.special(rotated), layout.special(values))
    assert torch.equal(layout.summary(rotated), layout.summary(values))


def test_model_rejects_invalid_input_contracts() -> None:
    config = load_config("configs/test.toml")
    model = PolicyValueTransformer(config.model, config.execution)
    board, global_features, legal = tensors_for_empty(batch=1)
    with pytest.raises(TypeError, match="boolean"):
        model(board, global_features, legal.float())
    with pytest.raises(TypeError, match="same dtype"):
        model(board.double(), global_features, legal)
    with pytest.raises(ValueError, match="cannot be empty"):
        model(board[:0], global_features[:0], legal[:0])


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


def test_checkpoint_executor_preserves_rng_during_recomputation() -> None:
    config = load_config("configs/test.toml")
    layout = TokenLayout(board_tokens=BOARD_AREA, special_tokens=1)
    rope = AxialRoPE2D(
        config.model.rope,
        config.model.d_model // config.model.n_heads,
        layout,
    )
    plugin = NoOpBlockResidualPlugin()

    class StochasticBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.0))

        def forward(self, hidden, rope, plugin):
            del rope, plugin
            output = F.dropout(hidden, p=0.5, training=True) * self.scale
            zero = output.new_zeros(())
            return BlockOutput(output, zero, zero, zero, zero)

    eager_block = StochasticBlock()
    checkpoint_block = StochasticBlock()
    checkpoint_block.load_state_dict(eager_block.state_dict())
    eager_executor = BlockExecutor(
        replace(config.execution, activation_checkpoint=False)
    )
    checkpoint_executor = BlockExecutor(
        replace(
            config.execution,
            activation_checkpoint=True,
            activation_checkpoint_stride=1,
        )
    )
    eager_hidden = torch.randn(2, 3, requires_grad=True)
    checkpoint_hidden = eager_hidden.detach().clone().requires_grad_(True)
    torch.manual_seed(123)
    eager_output = eager_executor.run(
        index=0,
        training=True,
        block=eager_block,
        rope=rope,
        plugin=plugin,
        hidden=eager_hidden,
    )
    torch.manual_seed(123)
    checkpoint_output = checkpoint_executor.run(
        index=0,
        training=True,
        block=checkpoint_block,
        rope=rope,
        plugin=plugin,
        hidden=checkpoint_hidden,
    )
    eager_output.hidden.sum().backward()
    checkpoint_output.hidden.sum().backward()
    assert torch.equal(eager_output.hidden, checkpoint_output.hidden)
    assert torch.equal(eager_hidden.grad, checkpoint_hidden.grad)
    assert torch.equal(eager_block.scale.grad, checkpoint_block.scale.grad)


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
    assert sum(
        block.plugin_enabled for block in model.backbone.blocks
    ) == config.model.hypernet.num_layers
    assert [
        index
        for index, block in enumerate(model.backbone.blocks)
        if block.plugin_enabled
    ] == [1]
    assert actual.diagnostics.hyper_dynamic_rms.item() == 0.0
    (actual.policy_logits.sum() + actual.ownership.sum()).backward()
    assert model.block_plugin.b_head.weight.grad is not None
    assert torch.isfinite(model.block_plugin.b_head.weight.grad).all()
    assert model.block_plugin.b_head.weight.grad.norm() > 0


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
