"""Stable records at the self-play, replay, and future data-source boundary."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from zero_ttt.game.rules import ACTION_SIZE, BOARD_AREA


REPLAY_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class GameRecord:
    source_kind: str
    model_version: int
    config_sha256: str
    komi_half_points: int
    max_moves: int
    history_length: int
    moves: np.ndarray
    visit_counts: np.ndarray
    search_budgets: np.ndarray
    root_values: np.ndarray
    final_margin_half_points: int
    final_ownership: np.ndarray
    ownership_mask: np.ndarray
    score_mask: np.ndarray
    termination: str

    def __post_init__(self) -> None:
        length = len(self.moves)
        expected = {
            "moves": ((length,), np.uint16),
            "visit_counts": ((length, ACTION_SIZE), np.uint16),
            "search_budgets": ((length,), np.uint16),
            "root_values": ((length,), np.float32),
            "final_ownership": ((BOARD_AREA,), np.int8),
            "ownership_mask": ((length,), np.bool_),
            "score_mask": ((length,), np.bool_),
        }
        for name, (shape, dtype) in expected.items():
            value = getattr(self, name)
            if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != dtype:
                raise ValueError(f"{name} must have shape {shape} and dtype {dtype}")
        if length == 0:
            raise ValueError("a game record must contain at least one position")
        if np.any(self.moves >= ACTION_SIZE):
            raise ValueError("game record contains an invalid move")
        if np.any(self.visit_counts.sum(axis=1, dtype=np.uint64) == 0):
            raise ValueError("each replay position needs at least one root visit")
        if np.any(self.search_budgets == 0):
            raise ValueError("search budgets must be positive")
        if not np.isfinite(self.root_values).all():
            raise ValueError("root values must be finite")
        if not np.isin(self.final_ownership, (-1, 0, 1)).all():
            raise ValueError("ownership labels must be -1, 0, or 1")
        if self.source_kind not in {"selfplay/search_visits", "external/played_move"}:
            raise ValueError("unsupported source_kind")
        if len(self.config_sha256) != 64:
            raise ValueError("config_sha256 must be a hexadecimal SHA-256")

    @property
    def length(self) -> int:
        return len(self.moves)


class GameSource(Protocol):
    def next_game(self) -> GameRecord: ...


@dataclass(frozen=True, slots=True)
class StoredPosition:
    game_id: int
    move_index: int
    game: GameRecord


def serialize_game(record: GameRecord) -> bytes:
    metadata = json.dumps(
        {
            "schema_version": REPLAY_SCHEMA_VERSION,
            "source_kind": record.source_kind,
            "model_version": record.model_version,
            "config_sha256": record.config_sha256,
            "komi_half_points": record.komi_half_points,
            "max_moves": record.max_moves,
            "history_length": record.history_length,
            "final_margin_half_points": record.final_margin_half_points,
            "termination": record.termination,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    output = io.BytesIO()
    np.savez_compressed(
        output,
        metadata=np.asarray(metadata),
        moves=record.moves,
        visit_counts=record.visit_counts,
        search_budgets=record.search_budgets,
        root_values=record.root_values,
        final_ownership=record.final_ownership,
        ownership_mask=record.ownership_mask,
        score_mask=record.score_mask,
    )
    return output.getvalue()


def deserialize_game(payload: bytes) -> GameRecord:
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            schema = metadata.pop("schema_version")
            if schema != REPLAY_SCHEMA_VERSION:
                raise ValueError(f"unsupported replay schema {schema}")
            return GameRecord(
                **metadata,
                moves=np.asarray(archive["moves"], dtype=np.uint16),
                visit_counts=np.asarray(archive["visit_counts"], dtype=np.uint16),
                search_budgets=np.asarray(archive["search_budgets"], dtype=np.uint16),
                root_values=np.asarray(archive["root_values"], dtype=np.float32),
                final_ownership=np.asarray(archive["final_ownership"], dtype=np.int8),
                ownership_mask=np.asarray(archive["ownership_mask"], dtype=np.bool_),
                score_mask=np.asarray(archive["score_mask"], dtype=np.bool_),
            )
    except (KeyError, OSError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("invalid replay payload") from error
