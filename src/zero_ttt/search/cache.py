"""Thread-safe, exact-identity neural evaluation cache."""

from __future__ import annotations

import threading
from collections import OrderedDict

from zero_ttt.game.state import GameState
from zero_ttt.search.protocol import Evaluation


CacheKey = tuple[int, tuple[object, ...]]


class EvaluationCache:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("cache capacity must be positive")
        self.capacity = capacity
        self._values: OrderedDict[CacheKey, Evaluation] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def key(state: GameState, model_version: int) -> CacheKey:
        return (model_version, state.identity())

    def get(self, state: GameState, model_version: int) -> Evaluation | None:
        key = self.key(state, model_version)
        with self._lock:
            value = self._values.get(key)
            if value is not None:
                self._values.move_to_end(key)
            return value

    def put(self, state: GameState, model_version: int, value: Evaluation) -> None:
        key = self.key(state, model_version)
        with self._lock:
            self._values[key] = value
            self._values.move_to_end(key)
            while len(self._values) > self.capacity:
                self._values.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._values)
