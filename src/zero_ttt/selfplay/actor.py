"""Generate complete self-play games from one immutable slow-model version."""

from __future__ import annotations

import numpy as np

from zero_ttt.config import ExperimentConfig
from zero_ttt.game.rules import BOARD_AREA
from zero_ttt.game.state import GameState
from zero_ttt.replay.records import GameRecord
from zero_ttt.search.protocol import LeafEvaluator
from zero_ttt.search.tree import PythonMCTS


class SelfPlayActor:
    def __init__(self, config: ExperimentConfig, evaluator: LeafEvaluator) -> None:
        self.config = config
        self.evaluator = evaluator

    def play_game(self, model_version: int, rng: np.random.Generator) -> GameRecord:
        state = GameState.new(self.config.game)
        search = PythonMCTS()
        moves: list[int] = []
        visits: list[np.ndarray] = []
        budgets: list[int] = []
        root_values: list[float] = []
        while not state.is_terminal():
            result = search.search(
                state=state,
                evaluator=self.evaluator,
                config=self.config.search,
                rng=rng,
                model_version=model_version,
                selfplay=True,
            )
            moves.append(result.action)
            visits.append(result.visit_counts.astype(np.uint16))
            budgets.append(result.simulations)
            root_values.append(result.root_value)
            state = state.play(result.action)
        final = state.score()
        ownership = np.frombuffer(final.score.ownership, dtype=np.int8).copy()
        length = len(moves)
        return GameRecord(
            source_kind="selfplay/search_visits",
            model_version=model_version,
            config_sha256=self.config.sha256,
            komi_half_points=self.config.game.komi_half_points,
            max_moves=self.config.game.max_moves,
            history_length=self.config.game.history_length,
            moves=np.asarray(moves, dtype=np.uint16),
            visit_counts=np.stack(visits).reshape(length, BOARD_AREA + 1),
            search_budgets=np.asarray(budgets, dtype=np.uint16),
            root_values=np.asarray(root_values, dtype=np.float32),
            final_margin_half_points=final.score.margin_half_points,
            final_ownership=ownership,
            ownership_mask=np.ones(length, dtype=np.bool_),
            score_mask=np.ones(length, dtype=np.bool_),
            termination=final.termination,
        )
