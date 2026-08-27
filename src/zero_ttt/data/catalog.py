"""SQLite control plane for immutable data shards and snapshots."""

from __future__ import annotations

import hashlib
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from zero_ttt.data.manifest import ManifestAsset, SourceManifest
from zero_ttt.data.records import AnnotationRecord, ImportEvent, TrajectoryRecord
from zero_ttt.data.shards import ShardInfo, ShardStore


CATALOG_SCHEMA_VERSION = 3


def _hash_identity_field(digest: "hashlib._Hash", value: str) -> None:
    encoded = value.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


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


class Catalog:
    def __init__(self, path: str | Path, store: ShardStore) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.store = store
        self.connection = sqlite3.connect(self.path, timeout=30.0)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA busy_timeout=30000")
        try:
            self._migrate()
        except BaseException:
            self.connection.close()
            raise

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "Catalog":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _migrate(self) -> None:
        with self.connection:
            self.connection.executescript(
                """
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
            )
            current = self.connection.execute(
                "SELECT value FROM catalog_meta WHERE key='schema_version'"
            ).fetchone()
            if current is None:
                self.connection.execute(
                    "INSERT INTO catalog_meta(key,value) VALUES('schema_version',?)",
                    (str(CATALOG_SCHEMA_VERSION),),
                )
            elif int(current["value"]) == 2:
                asset_columns = {
                    row["name"] for row in self.connection.execute("PRAGMA table_info(assets)")
                }
                if "source_kind" not in asset_columns:
                    self.connection.execute(
                        "ALTER TABLE assets ADD COLUMN source_kind TEXT NOT NULL DEFAULT 'external'"
                    )
                if "task_id" not in asset_columns:
                    self.connection.execute("ALTER TABLE assets ADD COLUMN task_id TEXT")
                snapshot_columns = {
                    row["name"]
                    for row in self.connection.execute("PRAGMA table_info(snapshots)")
                }
                if "source_kind" not in snapshot_columns:
                    self.connection.execute("ALTER TABLE snapshots ADD COLUMN source_kind TEXT")
                if "task_id" not in snapshot_columns:
                    self.connection.execute("ALTER TABLE snapshots ADD COLUMN task_id TEXT")
                self.connection.execute(
                    "UPDATE catalog_meta SET value=? WHERE key='schema_version'",
                    (str(CATALOG_SCHEMA_VERSION),),
                )
            elif int(current["value"]) != CATALOG_SCHEMA_VERSION:
                raise ValueError(
                    f"unsupported catalog schema v{current['value']}; "
                    "rebuild the catalog and snapshots for v2 or later"
                )

    def register_asset(self, manifest: SourceManifest, asset: ManifestAsset) -> None:
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO assets(asset_sha256,dataset_id,relative_path,size_bytes,status)
                VALUES(?,?,?,?, 'verified')
                ON CONFLICT(asset_sha256) DO UPDATE SET status='verified'
                """,
                (asset.sha256, manifest.dataset_id, asset.relative_path, asset.size_bytes),
            )

    def register_selfplay_task(
        self,
        *,
        task_id: str,
        manifest_relative_path: str,
        manifest_sha256: str,
        manifest_size_bytes: int,
        publication_sha256: str,
        evaluator_id: str,
        search_config_sha256: str,
        requested_games: int,
    ) -> None:
        if not task_id or requested_games <= 0:
            raise ValueError("invalid self-play task identity")
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO assets(
                    asset_sha256,dataset_id,relative_path,size_bytes,status,source_kind,task_id
                ) VALUES(?,?,?,?, 'verified','selfplay',?)
                ON CONFLICT(asset_sha256) DO UPDATE SET status='verified'
                """,
                (
                    manifest_sha256,
                    f"selfplay/{task_id}",
                    manifest_relative_path,
                    manifest_size_bytes,
                    task_id,
                ),
            )
            existing = self.connection.execute(
                "SELECT * FROM selfplay_tasks WHERE task_id=?", (task_id,)
            ).fetchone()
            identity = (
                manifest_sha256,
                publication_sha256,
                evaluator_id,
                search_config_sha256,
                requested_games,
            )
            if existing is not None:
                stored = (
                    existing["asset_sha256"],
                    existing["publication_sha256"],
                    existing["evaluator_id"],
                    existing["search_config_sha256"],
                    existing["requested_games"],
                )
                if stored != identity:
                    raise ValueError("conflicting self-play task identity")
                return
            self.connection.execute(
                """
                INSERT INTO selfplay_tasks(
                    task_id,asset_sha256,publication_sha256,evaluator_id,
                    search_config_sha256,requested_games,status,created_ns
                ) VALUES(?,?,?,?,?,?,'collecting',?)
                """,
                (task_id, *identity, time.time_ns()),
            )

    def set_selfplay_task_status(self, task_id: str, status: str) -> None:
        if status not in {"collecting", "sealed", "failed"}:
            raise ValueError("invalid self-play task status")
        completed_ns = time.time_ns() if status == "sealed" else None
        with self.connection:
            changed = self.connection.execute(
                "UPDATE selfplay_tasks SET status=?,completed_ns=? WHERE task_id=?",
                (status, completed_ns, task_id),
            ).rowcount
        if not changed:
            raise KeyError(task_id)

    def set_asset_status(self, asset_sha256: str, status: str) -> None:
        if status not in {"verified", "partial", "imported"}:
            raise ValueError("invalid asset status")
        with self.connection:
            changed = self.connection.execute(
                "UPDATE assets SET status=? WHERE asset_sha256=?",
                (status, asset_sha256),
            ).rowcount
        if not changed:
            raise KeyError(asset_sha256)

    def asset_status(self, asset_sha256: str) -> str:
        row = self.connection.execute(
            "SELECT status FROM assets WHERE asset_sha256=?",
            (asset_sha256,),
        ).fetchone()
        if row is None:
            raise KeyError(asset_sha256)
        return str(row["status"])

    def has_trajectory(self, game_id: str) -> bool:
        return self.connection.execute(
            "SELECT 1 FROM trajectories WHERE game_id=?", (game_id,)
        ).fetchone() is not None

    def commit_trajectory_shard(
        self,
        info: ShardInfo,
        records: list[TrajectoryRecord],
        rejections: list[ImportEvent] | None = None,
    ) -> None:
        if (
            info.kind != "trajectory"
            or len(records) != info.record_count
            or sum(record.trainable_position_count for record in records)
            != info.position_count
        ):
            raise ValueError("trajectory shard metadata does not match its records")
        stored_records = self.store.read_verified_trajectories(
            info.relative_path,
            info.sha256,
        )
        if stored_records != tuple(records):
            raise ValueError("trajectory shard contents do not match catalog records")
        with self.connection:
            self._insert_shard(info)
            for game_index, record in enumerate(records):
                self.connection.execute(
                    """
                    INSERT INTO trajectories(
                        game_id,content_sha256,asset_sha256,shard_sha256,
                        game_index,trainable_start_ply,trainable_positions
                    ) VALUES(?,?,?,?,?,?,?)
                    """,
                    (
                        record.game_id,
                        record.content_sha256,
                        record.asset_sha256,
                        info.sha256,
                        game_index,
                        record.trainable_start_ply,
                        record.trainable_position_count,
                    ),
                )
            self._insert_rejections(rejections or [])

    def commit_annotation_shard(
        self,
        info: ShardInfo,
        records: list[AnnotationRecord],
    ) -> None:
        if (
            info.kind != "annotation"
            or len(records) != info.record_count
            or info.position_count != len(records)
        ):
            raise ValueError("annotation shard metadata does not match its records")
        stored_records = self.store.read_verified_annotations(
            info.relative_path,
            info.sha256,
        )
        if stored_records != tuple(records):
            raise ValueError("annotation shard contents do not match catalog records")
        with self.connection:
            self._insert_shard(info)
            for record_index, record in enumerate(records):
                self.connection.execute(
                    """
                    INSERT INTO annotations(
                        game_id,content_sha256,ply,teacher_fingerprint,
                        shard_sha256,record_index
                    ) VALUES(?,?,?,?,?,?)
                    """,
                    (
                        record.game_id,
                        record.content_sha256,
                        record.ply,
                        record.teacher_fingerprint,
                        info.sha256,
                        record_index,
                    ),
                )

    def record_rejections(self, events: list[ImportEvent]) -> None:
        with self.connection:
            self._insert_rejections(events)

    def _insert_shard(self, info: ShardInfo) -> None:
        self.connection.execute(
            """
            INSERT INTO shards(
                shard_sha256,kind,relative_path,size_bytes,record_count,
                position_count,created_ns
            ) VALUES(?,?,?,?,?,?,?)
            """,
            (
                info.sha256,
                info.kind,
                info.relative_path,
                info.size_bytes,
                info.record_count,
                info.position_count,
                time.time_ns(),
            ),
        )

    def _insert_rejections(self, events: list[ImportEvent]) -> None:
        for event in events:
            if event.kind != "reject":
                raise ValueError("only rejection events can be recorded here")
            if event.asset_sha256 is None or self.connection.execute(
                "SELECT 1 FROM assets WHERE asset_sha256=?",
                (event.asset_sha256,),
            ).fetchone() is None:
                raise ValueError("rejection asset is not registered")
            self.connection.execute(
                """
                INSERT OR REPLACE INTO rejections(
                    asset_sha256,game_id,member_path,ordinal,reason_code,message
                ) VALUES(?,?,?,?,?,?)
                """,
                (
                    event.asset_sha256,
                    event.game_id,
                    event.member_path,
                    event.ordinal,
                    event.reason_code,
                    event.message,
                ),
            )

    def create_snapshot(
        self,
        seed: int,
        split: str = "train",
        validation_fraction: float = 0.1,
        source_kind: str | None = None,
        task_id: str | None = None,
    ) -> str:
        if split not in {"train", "validation"}:
            raise ValueError("split must be train or validation")
        if not 0.0 <= validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1)")
        if source_kind not in {None, "external", "selfplay"}:
            raise ValueError("source_kind must be external or selfplay")
        if task_id is not None and source_kind != "selfplay":
            raise ValueError("task_id filtering requires source_kind=selfplay")
        threshold = int(validation_fraction * (1 << 64))
        self.connection.execute(
            """
            CREATE TEMP TABLE IF NOT EXISTS snapshot_build_trajectories (
                ordinal INTEGER PRIMARY KEY,
                game_id TEXT NOT NULL UNIQUE,
                content_sha256 TEXT NOT NULL
            )
            """
        )
        self.connection.execute(
            """
            CREATE TEMP TABLE IF NOT EXISTS snapshot_build_annotations (
                game_id TEXT NOT NULL,
                content_sha256 TEXT NOT NULL,
                ply INTEGER NOT NULL,
                teacher_fingerprint TEXT NOT NULL,
                PRIMARY KEY(game_id,ply,teacher_fingerprint)
            )
            """
        )
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            self.connection.execute("DELETE FROM snapshot_build_trajectories")
            self.connection.execute("DELETE FROM snapshot_build_annotations")
            ordinal = 0
            for row in self.connection.execute(
                """
                SELECT game_id,content_sha256
                FROM trajectories WHERE deleted=0
                  AND (? IS NULL OR asset_sha256 IN (
                    SELECT asset_sha256 FROM assets WHERE source_kind=?
                      AND (? IS NULL OR task_id=?)
                  ))
                ORDER BY game_id
                """,
                (source_kind, source_kind, task_id, task_id),
            ):
                game_id = row["game_id"]
                split_digest = hashlib.sha256(f"{seed}:{game_id}".encode("ascii")).digest()
                validation = int.from_bytes(split_digest[:8], "big") < threshold
                if (split == "validation") != validation:
                    continue
                self.connection.execute(
                    """
                    INSERT INTO snapshot_build_trajectories(
                        ordinal,game_id,content_sha256
                    ) VALUES(?,?,?)
                    """,
                    (ordinal, game_id, row["content_sha256"]),
                )
                ordinal += 1
            self.connection.execute(
                """
                INSERT INTO snapshot_build_annotations(
                    game_id,content_sha256,ply,teacher_fingerprint
                )
                SELECT a.game_id,a.content_sha256,a.ply,a.teacher_fingerprint
                FROM annotations a
                JOIN snapshot_build_trajectories bt ON bt.game_id=a.game_id
                WHERE a.deleted=0
                ORDER BY a.game_id,a.ply,a.teacher_fingerprint
                """
            )

            identity = hashlib.sha256()
            for field in (
                "zero-ttt-snapshot-v2",
                str(seed),
                split,
                float(validation_fraction).hex(),
            ):
                _hash_identity_field(identity, field)
            if source_kind is not None:
                _hash_identity_field(identity, "source-filter")
                _hash_identity_field(identity, source_kind)
                _hash_identity_field(identity, task_id or "")
            for row in self.connection.execute(
                """
                SELECT game_id,content_sha256
                FROM snapshot_build_trajectories ORDER BY ordinal
                """
            ):
                _hash_identity_field(identity, "trajectory")
                _hash_identity_field(identity, row["game_id"])
                _hash_identity_field(identity, row["content_sha256"])
            for row in self.connection.execute(
                """
                SELECT game_id,content_sha256,ply,teacher_fingerprint
                FROM snapshot_build_annotations
                ORDER BY game_id,ply,teacher_fingerprint
                """
            ):
                _hash_identity_field(identity, "annotation")
                _hash_identity_field(identity, row["game_id"])
                _hash_identity_field(identity, str(int(row["ply"])))
                _hash_identity_field(identity, row["teacher_fingerprint"])
                _hash_identity_field(identity, row["content_sha256"])
            snapshot_id = identity.hexdigest()
            created = self.connection.execute(
                """
                INSERT OR IGNORE INTO snapshots(
                    snapshot_id,seed,split,validation_fraction,source_kind,task_id,created_ns
                ) VALUES(?,?,?,?,?,?,?)
                """,
                (
                    snapshot_id,
                    seed,
                    split,
                    validation_fraction,
                    source_kind,
                    task_id,
                    time.time_ns(),
                ),
            ).rowcount
            if created:
                self.connection.execute(
                    """
                    INSERT INTO snapshot_trajectories(snapshot_id,ordinal,game_id)
                    SELECT ?,ordinal,game_id FROM snapshot_build_trajectories
                    ORDER BY ordinal
                    """,
                    (snapshot_id,),
                )
                self.connection.execute(
                    """
                    INSERT INTO snapshot_annotations(
                        snapshot_id,game_id,ply,teacher_fingerprint
                    )
                    SELECT ?,game_id,ply,teacher_fingerprint
                    FROM snapshot_build_annotations
                    ORDER BY game_id,ply,teacher_fingerprint
                    """,
                    (snapshot_id,),
                )
            self.connection.commit()
            return snapshot_id
        except BaseException:
            self.connection.rollback()
            raise

    def snapshot_trajectories(self, snapshot_id: str) -> tuple[TrajectoryLocator, ...]:
        rows = self.connection.execute(
            """
            SELECT t.game_id,t.content_sha256,t.shard_sha256,s.relative_path,
                   t.game_index,t.trainable_start_ply,t.trainable_positions
            FROM snapshot_trajectories st
            JOIN trajectories t ON t.game_id=st.game_id
            JOIN shards s ON s.shard_sha256=t.shard_sha256
            WHERE st.snapshot_id=?
            ORDER BY st.ordinal
            """,
            (snapshot_id,),
        ).fetchall()
        if not rows and self.connection.execute(
            "SELECT 1 FROM snapshots WHERE snapshot_id=?", (snapshot_id,)
        ).fetchone() is None:
            raise KeyError(f"unknown snapshot {snapshot_id}")
        return tuple(TrajectoryLocator(**dict(row)) for row in rows)

    def annotation_locator(
        self,
        game_id: str,
        ply: int,
        teacher_fingerprint: str,
        snapshot_id: str | None = None,
    ) -> AnnotationLocator | None:
        snapshot_join = (
            "JOIN snapshot_annotations sa ON sa.game_id=a.game_id AND sa.ply=a.ply "
            "AND sa.teacher_fingerprint=a.teacher_fingerprint "
            if snapshot_id is not None
            else ""
        )
        snapshot_filter = "AND sa.snapshot_id=?" if snapshot_id is not None else ""
        visibility_filter = "" if snapshot_id is not None else "AND a.deleted=0 AND s.deleted=0"
        parameters: tuple[object, ...] = (game_id, ply, teacher_fingerprint)
        if snapshot_id is not None:
            parameters += (snapshot_id,)
        row = self.connection.execute(
            f"""
            SELECT a.game_id,a.content_sha256,a.ply,a.teacher_fingerprint,a.shard_sha256,
                   s.relative_path,a.record_index
            FROM annotations a JOIN shards s ON s.shard_sha256=a.shard_sha256
            {snapshot_join}
            WHERE a.game_id=? AND a.ply=? AND a.teacher_fingerprint=?
              {visibility_filter}
              {snapshot_filter}
            """,
            parameters,
        ).fetchone()
        return None if row is None else AnnotationLocator(**dict(row))

    def snapshot_annotations(
        self,
        snapshot_id: str,
        teacher_fingerprint: str,
    ) -> tuple[AnnotationLocator, ...]:
        rows = self.connection.execute(
            """
            SELECT a.game_id,a.content_sha256,a.ply,a.teacher_fingerprint,a.shard_sha256,
                   s.relative_path,a.record_index
            FROM snapshot_annotations sa
            JOIN annotations a
              ON a.game_id=sa.game_id AND a.ply=sa.ply
             AND a.teacher_fingerprint=sa.teacher_fingerprint
            JOIN shards s ON s.shard_sha256=a.shard_sha256
            WHERE sa.snapshot_id=? AND sa.teacher_fingerprint=?
            ORDER BY a.game_id,a.ply
            """,
            (snapshot_id, teacher_fingerprint),
        ).fetchall()
        return tuple(AnnotationLocator(**dict(row)) for row in rows)

    def verify(self) -> None:
        rows = self.connection.execute(
            "SELECT shard_sha256,relative_path FROM shards WHERE deleted=0 ORDER BY shard_sha256"
        ).fetchall()
        for row in rows:
            self.store.verify(row["relative_path"], row["shard_sha256"])

    def recover(self) -> tuple[str, ...]:
        for temporary in self.store.root.rglob(".shard-*.tmp"):
            temporary.unlink()
        registered = {
            row["relative_path"]
            for row in self.connection.execute("SELECT relative_path FROM shards")
        }
        orphans = []
        for path in self.store.root.glob("*/*.npz"):
            relative = path.relative_to(self.store.root).as_posix()
            if relative not in registered:
                orphans.append(relative)
        self.verify()
        return tuple(sorted(orphans))

    def mark_trajectory_deleted(self, game_id: str) -> None:
        with self.connection:
            changed = self.connection.execute(
                "UPDATE trajectories SET deleted=1 WHERE game_id=?", (game_id,)
            ).rowcount
            if changed:
                self.connection.execute(
                    "UPDATE annotations SET deleted=1 WHERE game_id=?",
                    (game_id,),
                )
        if not changed:
            raise KeyError(game_id)

    def mark_annotation_deleted(
        self,
        game_id: str,
        ply: int,
        teacher_fingerprint: str,
    ) -> None:
        with self.connection:
            changed = self.connection.execute(
                """
                UPDATE annotations SET deleted=1
                WHERE game_id=? AND ply=? AND teacher_fingerprint=?
                """,
                (game_id, ply, teacher_fingerprint),
            ).rowcount
        if not changed:
            raise KeyError((game_id, ply, teacher_fingerprint))

    def pin_shard(self, owner: str, shard_sha256: str) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO shard_pins(owner,shard_sha256) VALUES(?,?)",
                (owner, shard_sha256),
            )

    def unpin_shard(self, owner: str, shard_sha256: str) -> None:
        with self.connection:
            self.connection.execute(
                "DELETE FROM shard_pins WHERE owner=? AND shard_sha256=?",
                (owner, shard_sha256),
            )

    def garbage_collect(self) -> tuple[str, ...]:
        rows = self.connection.execute(
            """
            SELECT s.shard_sha256,s.relative_path,s.kind
            FROM shards s
            WHERE s.deleted=0
              AND NOT EXISTS(SELECT 1 FROM shard_pins p WHERE p.shard_sha256=s.shard_sha256)
              AND (
                (s.kind='trajectory'
                 AND NOT EXISTS(SELECT 1 FROM trajectories t WHERE t.shard_sha256=s.shard_sha256 AND t.deleted=0)
                 AND NOT EXISTS(
                    SELECT 1 FROM snapshot_trajectories st
                    JOIN trajectories t ON t.game_id=st.game_id
                    WHERE t.shard_sha256=s.shard_sha256
                 ))
                OR
                (s.kind='annotation'
                 AND NOT EXISTS(SELECT 1 FROM annotations a WHERE a.shard_sha256=s.shard_sha256 AND a.deleted=0)
                 AND NOT EXISTS(
                    SELECT 1 FROM snapshot_annotations sa
                    JOIN annotations a
                      ON a.game_id=sa.game_id AND a.ply=sa.ply
                     AND a.teacher_fingerprint=sa.teacher_fingerprint
                    WHERE a.shard_sha256=s.shard_sha256
                 ))
              )
            """
        ).fetchall()
        removed = []
        for row in rows:
            path = self.store.resolve(row["relative_path"])
            if path.exists():
                path.unlink()
            with self.connection:
                self.connection.execute(
                    "UPDATE shards SET deleted=1 WHERE shard_sha256=?",
                    (row["shard_sha256"],),
                )
            removed.append(row["relative_path"])
        return tuple(removed)

    def record_publication(
        self,
        run_id: str,
        optimizer_step: int,
        samples_seen: int,
        relative_path: str,
        sha256: str,
    ) -> None:
        if Path(relative_path).is_absolute() or ".." in Path(relative_path).parts:
            raise ValueError("publication path must be relative")
        with self.connection:
            existing = self.connection.execute(
                "SELECT samples_seen,relative_path,sha256 FROM publications "
                "WHERE run_id=? AND optimizer_step=?",
                (run_id, optimizer_step),
            ).fetchone()
            if existing is not None:
                if dict(existing) != {
                    "samples_seen": samples_seen,
                    "relative_path": relative_path,
                    "sha256": sha256,
                }:
                    raise ValueError("conflicting publication catalog entry")
                return
            self.connection.execute(
                """
                INSERT INTO publications(run_id,optimizer_step,samples_seen,relative_path,sha256)
                VALUES(?,?,?,?,?)
                """,
                (run_id, optimizer_step, samples_seen, relative_path, sha256),
            )
