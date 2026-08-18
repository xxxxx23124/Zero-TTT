"""Replay reconstruction, label creation, and D4 augmentation."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from zero_ttt.config import GameConfig
from zero_ttt.game.features import encode_position
from zero_ttt.game.rules import BOARD_AREA, Color
from zero_ttt.game.state import GameState
from zero_ttt.game.symmetry import SYMMETRY_COUNT, augment_sample
from zero_ttt.replay.records import GameRecord
from zero_ttt.replay.sqlite_store import ReplayStore


@dataclass(frozen=True, slots=True)
class SampledBatch:
    board: np.ndarray
    global_features: np.ndarray
    legal: np.ndarray
    policy: np.ndarray
    value: np.ndarray
    ownership: np.ndarray
    score_margin: np.ndarray
    ownership_mask: np.ndarray
    score_mask: np.ndarray


class ReplaySampler:
    def __init__(self, store: ReplayStore, decoded_cache_games: int) -> None:
        self.store = store
        self.decoded_cache_games = decoded_cache_games
        self._state_cache: OrderedDict[int, tuple[GameState, ...]] = OrderedDict()

    def _states(self, game_id: int, record: GameRecord) -> tuple[GameState, ...]:
        cached = self._state_cache.get(game_id)
        if cached is not None:
            self._state_cache.move_to_end(game_id)
            return cached
        game_config = GameConfig(
            board_size=19,
            komi_half_points=record.komi_half_points,
            max_moves=record.max_moves,
            history_length=record.history_length,
        )
        state = GameState.new(game_config)
        states: list[GameState] = []
        for action in record.moves:
            states.append(state)
            state = state.play(int(action))
        result = tuple(states)
        self._state_cache[game_id] = result
        self._state_cache.move_to_end(game_id)
        while len(self._state_cache) > self.decoded_cache_games:
            self._state_cache.popitem(last=False)
        return result

    def sample_batch(self, count: int, rng: np.random.Generator) -> SampledBatch:
        positions = self.store.sample_positions(count, rng)
        boards: list[np.ndarray] = []
        globals_: list[np.ndarray] = []
        legals: list[np.ndarray] = []
        policies: list[np.ndarray] = []
        values: list[float] = []
        ownerships: list[np.ndarray] = []
        scores: list[float] = []
        ownership_masks: list[bool] = []
        score_masks: list[bool] = []
        for position in positions:
            record = position.game
            index = position.move_index
            state = self._states(position.game_id, record)[index]
            visits = record.visit_counts[index].astype(np.float32)
            policy = visits / visits.sum()
            perspective = 1.0 if state.to_play is Color.BLACK else -1.0
            ownership = record.final_ownership.astype(np.float32) * perspective
            features = encode_position(state)
            symmetry = int(rng.integers(0, SYMMETRY_COUNT))
            augmented = augment_sample(features, policy, ownership, symmetry)
            boards.append(augmented.features.board)
            globals_.append(augmented.features.global_features)
            legals.append(augmented.features.legal)
            policies.append(augmented.policy)
            margin_points = record.final_margin_half_points / 2.0
            values.append(float(np.sign(margin_points) * perspective))
            ownerships.append(augmented.ownership)
            scores.append(margin_points * perspective)
            ownership_masks.append(bool(record.ownership_mask[index]))
            score_masks.append(bool(record.score_mask[index]))
        return SampledBatch(
            board=np.stack(boards).astype(np.float32, copy=False),
            global_features=np.stack(globals_).astype(np.float32, copy=False),
            legal=np.stack(legals).astype(np.bool_, copy=False),
            policy=np.stack(policies).astype(np.float32, copy=False),
            value=np.asarray(values, dtype=np.float32),
            ownership=np.stack(ownerships).reshape(count, BOARD_AREA).astype(np.float32, copy=False),
            score_margin=np.asarray(scores, dtype=np.float32),
            ownership_mask=np.asarray(ownership_masks, dtype=np.bool_),
            score_mask=np.asarray(score_masks, dtype=np.bool_),
        )
