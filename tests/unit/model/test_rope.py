from __future__ import annotations

import pytest
import torch
from torch import nn

from zero_ttt.config import RoPEConfig, load_config
from zero_ttt.game.rules import BOARD_AREA
from zero_ttt.model.rope import AxialRoPE2D
from zero_ttt.model.tokens import TokenLayout


def rope_config(
    *,
    centered: bool = True,
    learnable: bool = False,
    rotary_dim: int = 8,
) -> RoPEConfig:
    return RoPEConfig(
        base=100.0,
        scale=1.0,
        rotary_dim=rotary_dim,
        centered=centered,
        learnable=learnable,
    )


def token_index(row: int, col: int, board_size: int) -> int:
    return row * board_size + col


def test_coordinate_origin_and_all_special_tokens_are_unrotated() -> None:
    torch.manual_seed(1)
    layout = TokenLayout(board_tokens=9, special_tokens=3, summary_offset=1)
    values = torch.randn(2, 2, layout.total_tokens, 12)

    centered = AxialRoPE2D(rope_config(centered=True), 12, layout, board_size=3)
    centered_values = centered.apply(values)
    center = token_index(1, 1, 3)
    assert torch.equal(centered_values[..., center, :], values[..., center, :])
    assert torch.equal(layout.special(centered_values), layout.special(values))

    uncentered = AxialRoPE2D(rope_config(centered=False), 12, layout, board_size=3)
    uncentered_values = uncentered.apply(values)
    top_left = token_index(0, 0, 3)
    assert torch.equal(uncentered_values[..., top_left, :], values[..., top_left, :])
    assert not torch.equal(uncentered_values[..., center, :], values[..., center, :])


def test_row_and_column_rotations_use_independent_subspaces() -> None:
    layout = TokenLayout(board_tokens=9, special_tokens=1)
    rope = AxialRoPE2D(rope_config(), 8, layout, board_size=3)
    vector = torch.arange(1, 9, dtype=torch.float32)
    values = vector.repeat(layout.total_tokens).reshape(1, 1, layout.total_tokens, 8)
    rotated = rope.apply(values)

    top_left = rotated[..., token_index(0, 0, 3), :]
    top_right = rotated[..., token_index(0, 2, 3), :]
    bottom_left = rotated[..., token_index(2, 0, 3), :]
    assert torch.equal(top_left[..., :4], top_right[..., :4])
    assert not torch.equal(top_left[..., 4:8], top_right[..., 4:8])
    assert torch.equal(top_left[..., 4:8], bottom_left[..., 4:8])
    assert not torch.equal(top_left[..., :4], bottom_left[..., :4])


def test_rotary_dot_products_depend_only_on_relative_position() -> None:
    board_size = 5
    layout = TokenLayout(board_tokens=board_size**2, special_tokens=1)
    rope = AxialRoPE2D(rope_config(), 8, layout, board_size=board_size)
    query_vector = torch.tensor([0.5, -1.0, 2.0, 0.25, -0.5, 0.75, 1.5, -2.0])
    key_vector = torch.tensor([-1.0, 0.5, 0.75, 2.0, 1.0, -0.25, 0.5, 1.25])
    query = query_vector.repeat(layout.total_tokens).reshape(1, 1, layout.total_tokens, 8)
    key = key_vector.repeat(layout.total_tokens).reshape(1, 1, layout.total_tokens, 8)
    rotated_query, rotated_key = rope(query, key)

    first = torch.dot(
        rotated_query[0, 0, token_index(0, 1, board_size)],
        rotated_key[0, 0, token_index(2, 3, board_size)],
    )
    shifted = torch.dot(
        rotated_query[0, 0, token_index(1, 2, board_size)],
        rotated_key[0, 0, token_index(3, 4, board_size)],
    )
    assert torch.allclose(first, shifted, atol=1e-6, rtol=1e-6)


def test_rotation_preserves_norm_dtype_and_unrotated_tail() -> None:
    layout = TokenLayout(board_tokens=9, special_tokens=2)
    rope = AxialRoPE2D(rope_config(), 12, layout, board_size=3)
    values = torch.linspace(-2.0, 3.0, layout.total_tokens * 12).reshape(
        1, 1, layout.total_tokens, 12
    )
    rotated = rope.apply(values)

    assert rotated.dtype == values.dtype
    assert torch.allclose(
        torch.linalg.vector_norm(rotated, dim=-1),
        torch.linalg.vector_norm(values, dim=-1),
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.equal(rotated[..., :9, 8:], values[..., :9, 8:])


def test_learnable_frequencies_receive_gradient_and_fixed_frequencies_are_buffers() -> None:
    layout = TokenLayout(board_tokens=9, special_tokens=1)
    learnable = AxialRoPE2D(
        rope_config(learnable=True),
        8,
        layout,
        board_size=3,
    )
    assert isinstance(learnable.inv_freq, nn.Parameter)
    values = torch.linspace(-1.0, 2.0, layout.total_tokens * 8).reshape(
        1, 1, layout.total_tokens, 8
    )
    weights = torch.linspace(0.1, 1.0, values.numel()).reshape_as(values)
    (learnable.apply(values) * weights).sum().backward()
    assert learnable.inv_freq.grad is not None
    assert torch.isfinite(learnable.inv_freq.grad).all()
    assert torch.count_nonzero(learnable.inv_freq.grad) > 0

    fixed = AxialRoPE2D(rope_config(), 8, layout, board_size=3)
    assert "inv_freq" not in dict(fixed.named_parameters())
    assert "inv_freq" in dict(fixed.named_buffers())
    assert "inv_freq" in fixed.state_dict()


def test_rope_rejects_invalid_configuration_layout_and_token_count() -> None:
    layout = TokenLayout(board_tokens=9, special_tokens=1)
    with pytest.raises(ValueError, match="divisible by four"):
        AxialRoPE2D(rope_config(rotary_dim=6), 8, layout, board_size=3)
    with pytest.raises(ValueError, match="fit the head"):
        AxialRoPE2D(rope_config(rotary_dim=12), 8, layout, board_size=3)
    with pytest.raises(ValueError, match="does not match"):
        AxialRoPE2D(rope_config(), 8, layout, board_size=4)

    rope = AxialRoPE2D(rope_config(), 8, layout, board_size=3)
    with pytest.raises(ValueError, match="expected 10 tokens"):
        rope.apply(torch.zeros(1, 1, 9, 8))


def test_production_rope_frequency_span_is_stable() -> None:
    config = load_config("configs/rtx4090l.toml")
    head_dim = config.model.d_model // config.model.n_heads
    layout = TokenLayout(board_tokens=BOARD_AREA, special_tokens=1)
    rope = AxialRoPE2D(config.model.rope, head_dim, layout)
    slowest_phase = 18.0 * float(rope.inv_freq[-1])
    assert 0.2 < slowest_phase < 0.3
