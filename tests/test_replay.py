from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from zero_ttt.config import load_config
from zero_ttt.game.rules import ACTION_SIZE, BOARD_AREA
from zero_ttt.replay.records import (
    GameRecord,
    StoredPosition,
    deserialize_game,
    serialize_game,
)
from zero_ttt.replay.sampler import ReplaySampler
from zero_ttt.replay.sqlite_store import ReplayCorruptionError, ReplayStore


def make_record(length: int = 2, first_action: int = 0) -> GameRecord:
    config = load_config("configs/test.toml")
    moves = np.arange(first_action, first_action + length, dtype=np.uint16)
    visits = np.zeros((length, ACTION_SIZE), dtype=np.uint16)
    for index, action in enumerate(moves):
        visits[index, int(action)] = index + 1
    return GameRecord(
        source_kind="selfplay/search_visits",
        model_version=3,
        config_sha256=config.sha256,
        komi_half_points=config.game.komi_half_points,
        max_moves=config.game.max_moves,
        history_length=config.game.history_length,
        moves=moves,
        visit_counts=visits,
        search_budgets=np.full(length, 4, dtype=np.uint16),
        root_values=np.linspace(-0.2, 0.2, length, dtype=np.float32),
        final_margin_half_points=5,
        final_ownership=np.ones(BOARD_AREA, dtype=np.int8),
        ownership_mask=np.ones(length, dtype=np.bool_),
        score_mask=np.ones(length, dtype=np.bool_),
        termination="move_limit",
    )


def test_game_serialization_round_trip() -> None:
    record = make_record()
    restored = deserialize_game(serialize_game(record))
    assert restored.source_kind == record.source_kind
    assert restored.model_version == record.model_version
    assert np.array_equal(restored.moves, record.moves)
    assert np.array_equal(restored.visit_counts, record.visit_counts)


def test_sqlite_whole_game_fifo_and_restart(tmp_path) -> None:
    path = tmp_path / "replay.sqlite3"
    with ReplayStore(path, capacity_positions=5, decoded_cache_games=2) as store:
        first = store.add_game(make_record(3, 0))
        second = store.add_game(make_record(3, 20))
        assert store.position_count == 3
        assert store.game_count == 1
        assert second > first
        store.verify()
    with ReplayStore(path, capacity_positions=5, decoded_cache_games=2) as reopened:
        assert reopened.position_count == 3
        assert reopened.game_count == 1


def test_replay_checksum_failure_is_detected(tmp_path) -> None:
    path = tmp_path / "replay.sqlite3"
    with ReplayStore(path, capacity_positions=10, decoded_cache_games=1) as store:
        game_id = store.add_game(make_record())
    connection = sqlite3.connect(path)
    connection.execute("UPDATE games SET payload = ? WHERE id = ?", (b"broken", game_id))
    connection.commit()
    connection.close()
    with ReplayStore(path, capacity_positions=10, decoded_cache_games=1) as store:
        with pytest.raises(ReplayCorruptionError):
            store.verify()


def test_uniform_offsets_map_to_game_and_move(tmp_path) -> None:
    class FixedRng:
        def integers(self, low, high=None, size=None):
            del low, high
            return np.arange(size, dtype=np.int64)

    with ReplayStore(tmp_path / "replay.sqlite3", 10, 2) as store:
        first = store.add_game(make_record(2, 0))
        second = store.add_game(make_record(3, 20))
        positions = store.sample_positions(5, FixedRng())
    assert [(item.game_id, item.move_index) for item in positions] == [
        (first, 0),
        (first, 1),
        (second, 0),
        (second, 1),
        (second, 2),
    ]


def test_sampler_builds_current_player_targets() -> None:
    record = make_record(2, 0)

    class FakeStore:
        def sample_positions(self, count, rng):
            del count, rng
            return [StoredPosition(1, 0, record), StoredPosition(1, 1, record)]

    class IdentityRng:
        def integers(self, low, high=None, size=None):
            del low, high, size
            return 0

    sampler = ReplaySampler(FakeStore(), decoded_cache_games=1)
    batch = sampler.sample_batch(2, IdentityRng())
    assert batch.board.shape == (2, 25, 19, 19)
    assert batch.policy.shape == (2, ACTION_SIZE)
    assert batch.value.tolist() == [1.0, -1.0]
    assert batch.score_margin.tolist() == [2.5, -2.5]
    assert np.all(batch.ownership[0] == 1.0)
    assert np.all(batch.ownership[1] == -1.0)
