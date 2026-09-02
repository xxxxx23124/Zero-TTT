"""Token sequence layout and accessors."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class TokenLayout:
    """Describe board tokens followed by one or more special tokens."""

    board_tokens: int
    special_tokens: int
    summary_offset: int = 0

    def __post_init__(self) -> None:
        if self.board_tokens <= 0:
            raise ValueError("board_tokens must be positive")
        if self.special_tokens <= 0:
            raise ValueError("special_tokens must be positive")
        if not 0 <= self.summary_offset < self.special_tokens:
            raise ValueError("summary_offset must select a special token")

    @property
    def board_slice(self) -> slice:
        return slice(0, self.board_tokens)

    @property
    def special_slice(self) -> slice:
        return slice(self.board_tokens, self.total_tokens)

    @property
    def summary_index(self) -> int:
        return self.board_tokens + self.summary_offset

    @property
    def total_tokens(self) -> int:
        return self.board_tokens + self.special_tokens

    def validate(self, tensor: torch.Tensor) -> None:
        if tensor.ndim < 2 or tensor.shape[-2] != self.total_tokens:
            raise ValueError(
                f"expected {self.total_tokens} tokens, got shape {tuple(tensor.shape)}"
            )

    def board(self, tensor: torch.Tensor) -> torch.Tensor:
        self.validate(tensor)
        return tensor[..., self.board_slice, :]

    def special(self, tensor: torch.Tensor) -> torch.Tensor:
        self.validate(tensor)
        return tensor[..., self.special_slice, :]

    def summary(self, tensor: torch.Tensor) -> torch.Tensor:
        self.validate(tensor)
        return tensor[..., self.summary_index, :]
