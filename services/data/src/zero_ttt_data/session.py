"""SQLite connection, schema bootstrap, and transaction boundary."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from zero_ttt.versioning import CATALOG_SCHEMA

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS catalog_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS assets (
    asset_sha256 TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    status TEXT NOT NULL,
    source_kind TEXT NOT NULL DEFAULT 'external',
    task_id TEXT,
    UNIQUE(dataset_id, relative_path)
);
CREATE TABLE IF NOT EXISTS shards (
    shard_sha256 TEXT PRIMARY KEY,
    kind TEXT NOT NULL CHECK(kind IN ('trajectory','annotation')),
    relative_path TEXT NOT NULL UNIQUE,
    size_bytes INTEGER NOT NULL,
    record_count INTEGER NOT NULL,
    position_count INTEGER NOT NULL,
    deleted INTEGER NOT NULL DEFAULT 0,
    created_ns INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS trajectories (
    game_id TEXT PRIMARY KEY,
    content_sha256 TEXT NOT NULL,
    asset_sha256 TEXT NOT NULL REFERENCES assets(asset_sha256),
    shard_sha256 TEXT NOT NULL REFERENCES shards(shard_sha256),
    game_index INTEGER NOT NULL,
    trainable_start_ply INTEGER NOT NULL,
    trainable_positions INTEGER NOT NULL,
    deleted INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS annotations (
    game_id TEXT NOT NULL REFERENCES trajectories(game_id),
    content_sha256 TEXT NOT NULL,
    ply INTEGER NOT NULL,
    teacher_fingerprint TEXT NOT NULL,
    shard_sha256 TEXT NOT NULL REFERENCES shards(shard_sha256),
    record_index INTEGER NOT NULL,
    deleted INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY(game_id, ply, teacher_fingerprint)
);
CREATE TABLE IF NOT EXISTS rejections (
    asset_sha256 TEXT NOT NULL REFERENCES assets(asset_sha256),
    game_id TEXT NOT NULL,
    member_path TEXT,
    ordinal INTEGER,
    reason_code TEXT NOT NULL,
    message TEXT,
    PRIMARY KEY(asset_sha256, game_id)
);
CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id TEXT PRIMARY KEY,
    seed INTEGER NOT NULL,
    split TEXT NOT NULL,
    validation_fraction REAL NOT NULL,
    source_kind TEXT,
    task_id TEXT,
    created_ns INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS selfplay_tasks (
    task_id TEXT PRIMARY KEY,
    asset_sha256 TEXT NOT NULL REFERENCES assets(asset_sha256),
    publication_sha256 TEXT NOT NULL,
    evaluator_id TEXT NOT NULL,
    search_config_sha256 TEXT NOT NULL,
    requested_games INTEGER NOT NULL CHECK(requested_games > 0),
    status TEXT NOT NULL,
    created_ns INTEGER NOT NULL,
    completed_ns INTEGER
);
CREATE TABLE IF NOT EXISTS snapshot_trajectories (
    snapshot_id TEXT NOT NULL REFERENCES snapshots(snapshot_id),
    ordinal INTEGER NOT NULL,
    game_id TEXT NOT NULL REFERENCES trajectories(game_id),
    PRIMARY KEY(snapshot_id, game_id),
    UNIQUE(snapshot_id, ordinal)
);
CREATE TABLE IF NOT EXISTS snapshot_annotations (
    snapshot_id TEXT NOT NULL REFERENCES snapshots(snapshot_id),
    game_id TEXT NOT NULL,
    ply INTEGER NOT NULL,
    teacher_fingerprint TEXT NOT NULL,
    PRIMARY KEY(snapshot_id, game_id, ply, teacher_fingerprint),
    FOREIGN KEY(game_id, ply, teacher_fingerprint)
        REFERENCES annotations(game_id, ply, teacher_fingerprint)
);
CREATE TABLE IF NOT EXISTS shard_pins (
    owner TEXT NOT NULL,
    shard_sha256 TEXT NOT NULL REFERENCES shards(shard_sha256),
    PRIMARY KEY(owner, shard_sha256)
);
CREATE TABLE IF NOT EXISTS publications (
    run_id TEXT NOT NULL,
    optimizer_step INTEGER NOT NULL,
    samples_seen INTEGER NOT NULL,
    relative_path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    PRIMARY KEY(run_id, optimizer_step)
);
CREATE INDEX IF NOT EXISTS trajectories_shard_idx
    ON trajectories(shard_sha256, game_index);
CREATE INDEX IF NOT EXISTS annotations_shard_idx
    ON annotations(shard_sha256, record_index);
CREATE INDEX IF NOT EXISTS snapshot_ordinal_idx
    ON snapshot_trajectories(snapshot_id, ordinal);
"""


class CatalogSession:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=30.0)
        self.connection.row_factory = sqlite3.Row
        try:
            self._initialize()
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA foreign_keys=ON")
            self.connection.execute("PRAGMA synchronous=FULL")
            self.connection.execute("PRAGMA busy_timeout=30000")
        except BaseException:
            self.connection.close()
            raise

    def _initialize(self) -> None:
        existing = self.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' LIMIT 1"
        ).fetchone()
        if existing is not None:
            self._require_current_schema()
            return
        with self.connection:
            self.connection.executescript(_SCHEMA_SQL)
            self.connection.execute(
                "INSERT INTO catalog_meta(key,value) VALUES('schema_version',?)",
                (str(CATALOG_SCHEMA.current),),
            )

    def _require_current_schema(self) -> None:
        meta_table = self.connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='catalog_meta'"
        ).fetchone()
        if meta_table is None:
            CATALOG_SCHEMA.require(None)
        current = self.connection.execute(
            "SELECT value FROM catalog_meta WHERE key='schema_version'"
        ).fetchone()
        raw_version = None if current is None else current["value"]
        if raw_version is None:
            CATALOG_SCHEMA.require(None)
            return
        try:
            actual_version: object = int(raw_version)
        except (TypeError, ValueError):
            actual_version = raw_version
        CATALOG_SCHEMA.require(actual_version)

    def close(self) -> None:
        self.connection.close()
