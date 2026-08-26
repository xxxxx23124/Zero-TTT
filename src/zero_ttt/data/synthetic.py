"""In-memory data source for Docker smoke tests, not model training."""

from __future__ import annotations

import numpy as np

from zero_ttt.data.contracts import TrainBatch
from zero_ttt.game.features import GLOBAL_FEATURES, POINT_FEATURES
from zero_ttt.game.rules import ACTION_SIZE, BOARD_AREA, BOARD_SIZE


class SyntheticBatchSource:
    """Produce valid, deterministic-shape batches without persistent data."""

    def next_batch(
        self,
        batch_size: int,
        rng: np.random.Generator,
    ) -> TrainBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        policy = np.zeros((batch_size, ACTION_SIZE), dtype=np.float32)
        actions = rng.integers(0, ACTION_SIZE, size=batch_size)
        policy[np.arange(batch_size), actions] = 1.0
        return TrainBatch(
            board=np.zeros(
                (batch_size, POINT_FEATURES, BOARD_SIZE, BOARD_SIZE),
                dtype=np.float32,
            ),
            global_features=np.zeros((batch_size, GLOBAL_FEATURES), dtype=np.float32),
            legal=np.ones((batch_size, ACTION_SIZE), dtype=np.bool_),
            policy=policy,
            value=np.zeros(batch_size, dtype=np.float32),
            ownership=np.zeros((batch_size, BOARD_AREA), dtype=np.float32),
            score_margin=np.zeros(batch_size, dtype=np.float32),
            value_mask=np.ones(batch_size, dtype=np.bool_),
            ownership_mask=np.ones(batch_size, dtype=np.bool_),
            score_mask=np.ones(batch_size, dtype=np.bool_),
        )
