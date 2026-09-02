"""Batch inference interface shared by pure-policy and future search clients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from zero_ttt.game.features import GLOBAL_FEATURES, POINT_FEATURES
from zero_ttt.game.rules import ACTION_SIZE, BOARD_AREA, BOARD_SIZE


@dataclass(frozen=True, slots=True)
class InferenceBatch:
    board: torch.Tensor
    global_features: torch.Tensor
    legal: torch.Tensor

    def __post_init__(self) -> None:
        batch = self.board.shape[0] if self.board.ndim == 4 else -1
        if batch <= 0:
            raise ValueError("an inference batch cannot be empty")
        if self.board.shape != (batch, POINT_FEATURES, BOARD_SIZE, BOARD_SIZE):
            raise ValueError("board has the wrong shape")
        if self.global_features.shape != (batch, GLOBAL_FEATURES):
            raise ValueError("global_features has the wrong shape")
        if self.board.dtype != torch.float32 or self.global_features.dtype != torch.float32:
            raise TypeError("inference inputs must have torch.float32 dtype")
        if self.legal.shape != (batch, ACTION_SIZE) or self.legal.dtype != torch.bool:
            raise ValueError("legal has the wrong shape or dtype")


@dataclass(frozen=True, slots=True)
class InferenceOutput:
    policy_logits: torch.Tensor
    value: torch.Tensor
    ownership: torch.Tensor | None = None
    score_margin: torch.Tensor | None = None

    def __post_init__(self) -> None:
        batch = self.policy_logits.shape[0] if self.policy_logits.ndim == 2 else -1
        if batch <= 0:
            raise ValueError("an inference output cannot be empty")
        if self.policy_logits.shape != (batch, ACTION_SIZE):
            raise ValueError("policy_logits has the wrong shape")
        if self.value.shape not in {(batch,), (batch, 1)}:
            raise ValueError("value has the wrong shape")
        if self.ownership is not None and self.ownership.shape != (batch, BOARD_AREA):
            raise ValueError("ownership has the wrong shape")
        if self.score_margin is not None and self.score_margin.shape not in {
            (batch,),
            (batch, 1),
        }:
            raise ValueError("score_margin has the wrong shape")
        floating = (self.policy_logits, self.value, self.ownership, self.score_margin)
        if any(tensor is not None and tensor.dtype != torch.float32 for tensor in floating):
            raise TypeError("inference outputs must have torch.float32 dtype")


@runtime_checkable
class PositionEvaluator(Protocol):
    """Immutable-version evaluator usable by policy sampling or future MCTS."""

    @property
    def model_version(self) -> str: ...

    def evaluate(self, batch: InferenceBatch) -> InferenceOutput: ...
