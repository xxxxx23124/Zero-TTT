"""Transactional whole-game SQLite replay storage."""

from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

from zero_ttt.replay.records import (
    GameRecord,
    StoredPosition,
    deserialize_game,
    serialize_game,
)


class ReplayCorruptionError(RuntimeError):
    pass


class ReplayStore:
    def __init__(self, path: str | Path, capacity_positions: int, decoded_cache_games: int) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.capacity_positions = capacity_positions
        self.decoded_cache_games = decoded_cache_games
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.path, timeout=30.0, check_same_thread=False)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS games (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_ns INTEGER NOT NULL,
                source_kind TEXT NOT NULL,
                model_version INTEGER NOT NULL,
                position_count INTEGER NOT NULL CHECK(position_count > 0),
                payload BLOB NOT NULL,
                sha256 TEXT NOT NULL
            )
            """
        )
        self._connection.commit()
        self._index: list[tuple[int, int]] = []
        self._decoded: OrderedDict[int, GameRecord] = OrderedDict()
        self._reload_index()

    def _reload_index(self) -> None:
        rows = self._connection.execute(
            "SELECT id, position_count FROM games ORDER BY id"
        ).fetchall()
        self._index = [(int(game_id), int(length)) for game_id, length in rows]

    @property
    def position_count(self) -> int:
        with self._lock:
            return sum(length for _, length in self._index)

    @property
    def game_count(self) -> int:
        with self._lock:
            return len(self._index)

    def add_game(self, record: GameRecord) -> int:
        if record.length > self.capacity_positions:
            raise ValueError("a single game cannot exceed the replay capacity")
        payload = serialize_game(record)
        digest = hashlib.sha256(payload).hexdigest()
        with self._lock:
            current = sum(length for _, length in self._index)
            delete_count = 0
            while current + record.length > self.capacity_positions:
                current -= self._index[delete_count][1]
                delete_count += 1
            deleted = self._index[:delete_count]
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                for game_id, _ in deleted:
                    self._connection.execute("DELETE FROM games WHERE id = ?", (game_id,))
                cursor = self._connection.execute(
                    """
                    INSERT INTO games(
                        created_ns, source_kind, model_version, position_count, payload, sha256
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        time.time_ns(),
                        record.source_kind,
                        record.model_version,
                        record.length,
                        payload,
                        digest,
                    ),
                )
                game_id = int(cursor.lastrowid)
                self._connection.commit()
            except BaseException:
                self._connection.rollback()
                raise
            for deleted_id, _ in deleted:
                self._decoded.pop(deleted_id, None)
            self._index = self._index[delete_count:] + [(game_id, record.length)]
            self._cache_record(game_id, record)
            return game_id

    def _cache_record(self, game_id: int, record: GameRecord) -> None:
        self._decoded[game_id] = record
        self._decoded.move_to_end(game_id)
        while len(self._decoded) > self.decoded_cache_games:
            self._decoded.popitem(last=False)

    def _load_game(self, game_id: int) -> GameRecord:
        cached = self._decoded.get(game_id)
        if cached is not None:
            self._decoded.move_to_end(game_id)
            return cached
        row = self._connection.execute(
            "SELECT payload, sha256 FROM games WHERE id = ?", (game_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"replay game {game_id} does not exist")
        payload = bytes(row[0])
        if hashlib.sha256(payload).hexdigest() != row[1]:
            raise ReplayCorruptionError(f"replay checksum mismatch for game {game_id}")
        try:
            record = deserialize_game(payload)
        except ValueError as error:
            raise ReplayCorruptionError(f"invalid replay game {game_id}") from error
        self._cache_record(game_id, record)
        return record

    def sample_positions(
        self,
        count: int,
        rng: np.random.Generator,
    ) -> list[StoredPosition]:
        if count <= 0:
            raise ValueError("sample count must be positive")
        with self._lock:
            total = sum(length for _, length in self._index)
            if total == 0:
                raise RuntimeError("cannot sample an empty replay")
            cumulative = np.cumsum([length for _, length in self._index], dtype=np.int64)
            offsets = np.asarray(rng.integers(0, total, size=count), dtype=np.int64)
            selected = np.searchsorted(cumulative, offsets, side="right")
            result: list[StoredPosition] = []
            for offset, index in zip(offsets, selected, strict=True):
                game_id, _ = self._index[int(index)]
                previous = 0 if index == 0 else int(cumulative[int(index) - 1])
                result.append(
                    StoredPosition(
                        game_id=game_id,
                        move_index=int(offset) - previous,
                        game=self._load_game(game_id),
                    )
                )
            return result

    def verify(self) -> None:
        with self._lock:
            for game_id, _ in self._index:
                self._decoded.pop(game_id, None)
                self._load_game(game_id)
            result = self._connection.execute("PRAGMA integrity_check").fetchone()
            if result != ("ok",):
                raise ReplayCorruptionError(f"SQLite integrity check failed: {result!r}")

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "ReplayStore":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
