from __future__ import annotations

from dataclasses import replace

import numpy as np

from zero_ttt.config import load_config
from zero_ttt.game.rules import ACTION_SIZE
from zero_ttt.search.protocol import Evaluation
from zero_ttt.selfplay.actor import SelfPlayActor
from zero_ttt.selfplay.loop import CoreLoop


class UniformEvaluator:
    def evaluate(self, state, model_version: int) -> Evaluation:
        del model_version
        policy = np.asarray(state.legal_actions(), dtype=np.float32)
        policy /= policy.sum()
        return Evaluation(policy=policy, value=0.0)


def test_selfplay_actor_produces_replayable_complete_record() -> None:
    config = load_config("configs/test.toml")
    config = replace(config, game=replace(config.game, max_moves=4))
    record = SelfPlayActor(config, UniformEvaluator()).play_game(
        model_version=9,
        rng=np.random.default_rng(2),
    )
    assert record.length <= 4
    assert record.visit_counts.shape == (record.length, ACTION_SIZE)
    assert record.model_version == 9
    assert record.termination in {"two_passes", "move_limit"}
    assert np.all(record.search_budgets >= 1)


def test_tiny_closed_loop_and_resume(tmp_path) -> None:
    config = load_config("configs/test.toml")
    config = replace(
        config,
        game=replace(config.game, max_moves=2),
        training=replace(config.training, publish_interval=1),
        selfplay=replace(
            config.selfplay,
            games_per_cycle=1,
            minimum_replay_positions=1,
            train_samples_per_new_position=2.0,
        ),
        runtime=replace(config.runtime, run_dir=str(tmp_path / "run")),
    )
    with CoreLoop(config) as loop:
        result = loop.run_cycle()
        assert result.games == 1
        assert result.new_positions == 2
        assert result.optimizer_steps == 1
        assert result.final_optimizer_step == 1
        assert (config.run_dir / "published" / "current.pt").exists()
        assert (config.run_dir / config.replay.database_name).exists()
    with CoreLoop(config) as restored:
        assert restored.trainer.state.optimizer_step == 1
        assert restored.replay.position_count == 2
