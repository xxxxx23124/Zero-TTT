"""Stable search and evaluator interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from zero_ttt.config import SearchConfig
from zero_ttt.game.state import GameState


@dataclass(frozen=True, slots=True)
class Evaluation:
    policy: np.ndarray
    value: float
    ownership: np.ndarray | None = None
    score_margin: float | None = None

    def __post_init__(self) -> None:
        if self.policy.shape != (362,):
            raise ValueError("evaluation policy must have 362 entries")


class BatchEvaluator(Protocol):
    def evaluate_batch(
        self,
        states: Sequence[GameState],
        model_version: int,
    ) -> list[Evaluation]: ...


class LeafEvaluator(Protocol):
    def evaluate(self, state: GameState, model_version: int) -> Evaluation: ...


@dataclass(frozen=True, slots=True)
class SearchResult:
    action: int
    visit_counts: np.ndarray
    root_value: float
    simulations: int
    stop_reason: str
    normalized_entropy: float
    top_gap: float


class SearchBackend(Protocol):
    def search(
        self,
        state: GameState,
        evaluator: LeafEvaluator,
        config: SearchConfig,
        rng: np.random.Generator,
        model_version: int,
        selfplay: bool = True,
    ) -> SearchResult: ...
