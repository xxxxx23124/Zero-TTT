"""Stable value types returned by the catalog facade."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TrajectoryLocator:
    game_id: str
    content_sha256: str
    shard_sha256: str
    relative_path: str
    game_index: int
    trainable_start_ply: int
    trainable_positions: int


@dataclass(frozen=True, slots=True)
class AnnotationLocator:
    game_id: str
    content_sha256: str
    ply: int
    teacher_fingerprint: str
    shard_sha256: str
    relative_path: str
    record_index: int


@dataclass(frozen=True, slots=True)
class SelfPlayStatistics:
    sealed_tasks: int
    collecting_tasks: int
    failed_tasks: int
    games: int
    positions: int


@dataclass(frozen=True, slots=True)
class ImportStatistics:
    verified_assets: int
    partial_assets: int
    imported_assets: int
    games: int
    positions: int
    shards: int


@dataclass(frozen=True, slots=True)
class SnapshotStatistics:
    snapshot_id: str
    source_kind: str | None
    task_id: str | None
    games: int
    positions: int


@dataclass(frozen=True, slots=True)
class SnapshotSummary:
    snapshot_id: str
    seed: int
    split: str
    validation_fraction: float
    source_kind: str | None
    task_id: str | None
    created_ns: int
    games: int
    positions: int
