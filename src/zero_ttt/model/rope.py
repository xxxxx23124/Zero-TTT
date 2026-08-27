"""Centered axial 2D rotary position embeddings."""

from __future__ import annotations

import torch
from torch import nn

from zero_ttt.config import RoPEConfig
from zero_ttt.model.tokens import TokenLayout


class AxialRoPE2D(nn.Module):
    """Apply independent row and column rotations, leaving CLS unrotated."""

    inv_freq: torch.Tensor
    row_positions: torch.Tensor
    col_positions: torch.Tensor

    def __init__(
        self,
        config: RoPEConfig,
        head_dim: int,
        layout: TokenLayout,
        board_size: int = 19,
    ) -> None:
        super().__init__()
        if config.rotary_dim > head_dim or config.rotary_dim % 4:
            raise ValueError("rotary_dim must fit the head and be divisible by four")
        if layout.board_tokens != board_size * board_size:
            raise ValueError("token layout does not match the RoPE board size")
        self.layout = layout
        self.rotary_dim = config.rotary_dim
        self.axis_dim = config.rotary_dim // 2
        self.scale = config.scale
        pairs_per_axis = self.axis_dim // 2
        inv_freq = config.base ** (
            -torch.arange(pairs_per_axis, dtype=torch.float32) / pairs_per_axis
        )
        if config.learnable:
            self.inv_freq = nn.Parameter(inv_freq)
        else:
            self.register_buffer("inv_freq", inv_freq, persistent=True)
        if config.centered:
            coordinates = torch.arange(board_size, dtype=torch.float32) - (board_size - 1) / 2
        else:
            coordinates = torch.arange(board_size, dtype=torch.float32)
        rows = coordinates[:, None].expand(board_size, board_size).reshape(-1)
        cols = coordinates[None, :].expand(board_size, board_size).reshape(-1)
        self.register_buffer("row_positions", rows, persistent=True)
        self.register_buffer("col_positions", cols, persistent=True)

    @staticmethod
    def _rotate_axis(values: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        even = values[..., 0::2]
        odd = values[..., 1::2]
        rotated_even = even * cos - odd * sin
        rotated_odd = even * sin + odd * cos
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)

    def _trig(
        self, positions: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        angles = torch.outer(positions * self.scale, self.inv_freq)
        return angles.cos().to(dtype=dtype)[None, None], angles.sin().to(dtype=dtype)[None, None]

    def apply(self, tensor: torch.Tensor) -> torch.Tensor:
        """Rotate board positions and leave every special token unchanged."""

        self.layout.validate(tensor)
        board = self.layout.board(tensor)
        special = self.layout.special(tensor)
        row_cos, row_sin = self._trig(self.row_positions, tensor.dtype)
        col_cos, col_sin = self._trig(self.col_positions, tensor.dtype)
        row = self._rotate_axis(board[..., : self.axis_dim], row_cos, row_sin)
        col = self._rotate_axis(
            board[..., self.axis_dim : self.rotary_dim],
            col_cos,
            col_sin,
        )
        if self.rotary_dim == board.shape[-1]:
            rotated_board = torch.cat((row, col), dim=-1)
        else:
            rotated_board = torch.cat((row, col, board[..., self.rotary_dim :]), dim=-1)
        return torch.cat((rotated_board, special), dim=-2)

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.apply(query), self.apply(key)
