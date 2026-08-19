from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest

from zero_ttt.config import load_config
from zero_ttt.game.rules import ACTION_SIZE, PASS_ACTION
from zero_ttt.game.state import GameState
from zero_ttt.model.transformer import PolicyValueTransformer
from zero_ttt.search.cache import EvaluationCache
from zero_ttt.search.inference import InferenceServer, TorchBatchEvaluator
from zero_ttt.search.protocol import Evaluation
from zero_ttt.search.tree import PythonMCTS, SearchEdge, SearchNode, terminal_value


class UniformEvaluator:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value
        self.calls = 0
        self.lock = threading.Lock()

    def evaluate(self, state: GameState, model_version: int) -> Evaluation:
        del model_version
        with self.lock:
            self.calls += 1
        legal = np.asarray(state.legal_actions(), dtype=np.bool_)
        policy = legal.astype(np.float32)
        policy /= policy.sum()
        return Evaluation(policy=policy, value=self.value)


class CountingBatchEvaluator:
    def __init__(self) -> None:
        self.calls = 0
        self.items = 0
        self.lock = threading.Lock()

    def evaluate_batch(self, states, model_version: int) -> list[Evaluation]:
        del model_version
        with self.lock:
            self.calls += 1
            self.items += len(states)
        results = []
        for state in states:
            policy = np.asarray(state.legal_actions(), dtype=np.float32)
            policy /= policy.sum()
            results.append(Evaluation(policy=policy, value=0.0))
        return results


def _state(max_moves: int = 12) -> GameState:
    game = replace(load_config("configs/test.toml").game, max_moves=max_moves)
    return GameState.new(game)


def test_terminal_value_uses_side_to_play_perspective() -> None:
    terminal = _state().play(PASS_ACTION).play(PASS_ACTION)
    assert terminal.is_terminal()
    # Empty board with positive komi is a white win; after two passes black is to play.
    assert terminal_value(terminal) == -1.0


def test_puct_fpu_penalizes_unvisited_edges() -> None:
    config = load_config("configs/test.toml").search
    node = SearchNode(_state(), expanded=True, network_value=0.5)
    node.edges = {
        0: SearchEdge(prior=0.9, visits=1, value_sum=0.0),
        1: SearchEdge(prior=0.1),
    }
    action, edge = PythonMCTS._select_edge(node, replace(config, c_puct=0.0))
    assert action == 0
    assert edge.virtual_loss == config.virtual_loss


def test_search_dynamic_budget_and_tree_reuse() -> None:
    config = replace(
        load_config("configs/test.toml").search,
        num_threads=1,
        budget_1=4,
        budget_2=6,
        budget_3=8,
        budget_4=10,
        entropy_threshold=1.0,
        gap_threshold_1=0.0,
    )
    evaluator = UniformEvaluator()
    search = PythonMCTS()
    state = _state()
    result = search.search(state, evaluator, config, np.random.default_rng(3), 7, False)
    assert result.simulations == 4
    assert result.stop_reason == "budget_1"
    assert state.legal_actions()[result.action]

    child_state = state.play(result.action)
    child = search._root.edges[result.action].child
    assert child is not None
    search.search(child_state, evaluator, config, np.random.default_rng(4), 7, False)
    assert search._root is child

    # Search statistics cannot cross a published model version.
    search.search(child_state, evaluator, config, np.random.default_rng(5), 8, False)
    assert search._root is not child


def test_dynamic_budget_threshold_boundaries() -> None:
    config = load_config("configs/test.toml").search
    assert PythonMCTS._should_stop(
        0, [3], config.entropy_threshold, config.gap_threshold_1, config
    )
    assert not PythonMCTS._should_stop(
        0, [3], config.entropy_threshold + 1e-6, config.gap_threshold_1, config
    )
    assert PythonMCTS._should_stop(1, [3, 3], 1.0, config.gap_threshold_2, config)
    assert not PythonMCTS._should_stop(1, [3, 4], 0.0, 1.0, config)
    assert PythonMCTS._should_stop(2, [1, 4, 4], 1.0, config.gap_threshold_3, config)
    assert not PythonMCTS._should_stop(2, [4, 4, 5], 0.0, 1.0, config)
    assert PythonMCTS._should_stop(3, [1, 2, 3, 4], 1.0, 0.0, config)


def test_root_noise_is_a_probability_mixture() -> None:
    config = load_config("configs/test.toml").search
    root = SearchNode(_state(), expanded=True)
    root.edges = {action: SearchEdge(0.25) for action in range(4)}
    PythonMCTS._add_root_noise(root, config, np.random.default_rng(3))
    priors = np.asarray([edge.prior for edge in root.edges.values()])
    assert np.isclose(priors.sum(), 1.0)
    assert np.all(priors >= (1.0 - config.dirichlet_weight) * 0.25)
    assert not np.allclose(priors, 0.25)


def test_terminal_leaf_value_is_flipped_into_parent_edge() -> None:
    config = replace(
        load_config("configs/test.toml").search,
        num_threads=1,
        budget_1=1,
        budget_2=1,
        budget_3=1,
        budget_4=1,
    )
    search = PythonMCTS()
    result = search.search(
        _state(max_moves=1),
        UniformEvaluator(),
        config,
        np.random.default_rng(1),
        model_version=1,
        selfplay=False,
    )
    assert result.root_value == 1.0


def test_virtual_loss_is_recovered_when_evaluation_fails() -> None:
    config = replace(
        load_config("configs/test.toml").search,
        num_threads=1,
        budget_1=1,
        budget_2=1,
        budget_3=1,
        budget_4=1,
    )

    class FailsAfterRoot(UniformEvaluator):
        def evaluate(self, state: GameState, model_version: int) -> Evaluation:
            if self.calls:
                raise FloatingPointError("synthetic failure")
            return super().evaluate(state, model_version)

    search = PythonMCTS()
    with pytest.raises(FloatingPointError):
        search.search(_state(), FailsAfterRoot(), config, np.random.default_rng(0), 1, False)
    assert search._root is not None
    assert all(edge.virtual_loss == 0 for edge in search._root.edges.values())


def test_inference_queue_deduplicates_and_cache_is_versioned() -> None:
    config = replace(load_config("configs/test.toml").search, batch_delay_ms=10.0)
    backend = CountingBatchEvaluator()
    state = _state()
    with InferenceServer(backend, config, EvaluationCache(8)) as server:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(server.evaluate, state, 1) for _ in range(4)]
            results = [future.result(timeout=2) for future in futures]
        assert all(result.policy.shape == (ACTION_SIZE,) for result in results)
        assert backend.items == 1
        server.evaluate(state, 1)
        assert backend.calls == 1
        server.evaluate(state, 2)
        assert backend.calls == 2


def test_torch_evaluator_rejects_a_mismatched_publication_version() -> None:
    config = load_config("configs/test.toml")
    backend = TorchBatchEvaluator(
        PolicyValueTransformer(config.model),
        config.runtime,
        config.search.max_batch_size,
        model_version=7,
    )
    with pytest.raises(ValueError, match="requested model_version=8"):
        backend.evaluate_batch([_state()], model_version=8)


def test_cache_identity_includes_pass_and_feature_history() -> None:
    cache = EvaluationCache(8)
    state = _state()
    evaluation = UniformEvaluator().evaluate(state, 1)
    cache.put(state, 1, evaluation)
    same_board_and_player = state.play(PASS_ACTION).play(PASS_ACTION)
    assert same_board_and_player.board == state.board
    assert same_board_and_player.to_play == state.to_play
    assert cache.get(same_board_and_player, 1) is None


def test_threaded_search_completes_without_virtual_loss_leaks() -> None:
    config = replace(
        load_config("configs/test.toml").search,
        num_threads=16,
        budget_1=16,
        budget_2=16,
        budget_3=16,
        budget_4=16,
    )
    search = PythonMCTS()
    result = search.search(
        _state(), UniformEvaluator(), config, np.random.default_rng(10), 1, False
    )
    assert result.simulations == 16
    assert search._root is not None
    assert all(edge.virtual_loss == 0 for edge in search._root.edges.values())
