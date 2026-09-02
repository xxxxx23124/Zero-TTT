from __future__ import annotations

import dataclasses

import numpy as np
import torch
import zero_ttt_selfplay_worker.search.open_spiel as open_spiel_adapter
from zero_ttt.config import load_config
from zero_ttt.game.rules import ACTION_SIZE, PASS_ACTION
from zero_ttt_selfplay_worker.inference import (
    BatchedInferenceBroker,
    InferenceBatch,
    InferenceOutput,
)
from zero_ttt_selfplay_worker.search import OpenSpielEvaluator, OpenSpielGoGame, search_position


class UniformEvaluator:
    model_version = "uniform-v1"

    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def evaluate(self, batch: InferenceBatch) -> InferenceOutput:
        size = batch.board.shape[0]
        return InferenceOutput(
            policy_logits=torch.zeros(size, ACTION_SIZE),
            value=torch.full((size,), self.value),
            ownership=torch.zeros(size, 361),
            score_margin=torch.zeros(size),
        )


def test_open_spiel_state_forwards_local_rules_and_returns() -> None:
    config = load_config("configs/test.toml")
    game = OpenSpielGoGame(dataclasses.replace(config.game, max_moves=4))
    state = game.new_initial_state()
    clone = state.clone()
    assert state.legal_actions() == [
        action for action, legal in enumerate(state.local_state.legal_actions()) if legal
    ]
    assert PASS_ACTION in state.legal_actions()
    clone.apply_action(PASS_ACTION)
    assert state.local_state.move_number == 0
    clone.apply_action(PASS_ACTION)
    assert clone.is_terminal()
    assert clone.returns() == [-1.0, 1.0]

    limited = OpenSpielGoGame(dataclasses.replace(config.game, max_moves=2))
    limit_state = limited.new_initial_state()
    limit_state.apply_action(0)
    limit_state.apply_action(1)
    assert limit_state.is_terminal()
    assert sum(limit_state.returns()) == 0.0


def test_evaluator_converts_current_player_value_to_black_white() -> None:
    config = load_config("configs/test.toml")
    game = OpenSpielGoGame(config.game)
    state = game.new_initial_state()
    with BatchedInferenceBroker(
        UniformEvaluator(0.25), batch_size=16, batch_wait_ms=0, cache_size=4
    ) as broker:
        evaluator = OpenSpielEvaluator(broker)
        np.testing.assert_allclose(evaluator.evaluate(state), [0.25, -0.25])
        state.apply_action(0)
        np.testing.assert_allclose(evaluator.evaluate(state), [-0.25, 0.25])


def test_open_spiel_puct_exposes_visit_policy_and_value() -> None:
    config = load_config("configs/test.toml")
    search = dataclasses.replace(
        config.search,
        max_simulations=64,
        dirichlet_epsilon=0.25,
        temperature=0.0,
        temperature_drop_ply=0,
    )
    game = OpenSpielGoGame(dataclasses.replace(config.game, max_moves=2))
    state = game.new_initial_state()
    with BatchedInferenceBroker(
        UniformEvaluator(), batch_size=16, batch_wait_ms=0, cache_size=64
    ) as broker:
        evaluator = OpenSpielEvaluator(broker)
        result = search_position(
            game,
            state,
            evaluator,
            search,
            search_seed=3,
            selection_seed=4,
        )
        repeated = search_position(
            game,
            state,
            evaluator,
            search,
            search_seed=3,
            selection_seed=4,
        )
    assert result.simulations == 64
    assert np.isclose(sum(result.policy_values), 1.0)
    assert result.action in state.legal_actions()
    assert result.temperature == 0.0
    assert repeated == result


def test_zero_dirichlet_epsilon_disables_root_noise(monkeypatch) -> None:
    config = load_config("configs/test.toml")
    search = dataclasses.replace(
        config.search,
        max_simulations=2,
        dirichlet_epsilon=0.0,
        temperature=0.0,
        temperature_drop_ply=0,
    )
    game = OpenSpielGoGame(dataclasses.replace(config.game, max_moves=2))
    state = game.new_initial_state()
    captured: list[tuple[float, float] | None] = []
    original = open_spiel_adapter.mcts.MCTSBot

    def recording_bot(*args, **kwargs):
        captured.append(kwargs["dirichlet_noise"])
        return original(*args, **kwargs)

    monkeypatch.setattr(open_spiel_adapter.mcts, "MCTSBot", recording_bot)
    with BatchedInferenceBroker(
        UniformEvaluator(), batch_size=1, batch_wait_ms=0, cache_size=4
    ) as broker:
        result = search_position(
            game,
            state,
            OpenSpielEvaluator(broker),
            search,
            search_seed=3,
            selection_seed=4,
        )
    assert captured == [None]
    assert np.isclose(sum(result.policy_values), 1.0)
