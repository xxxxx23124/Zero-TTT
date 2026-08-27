"""Deterministic immutable snapshot construction and lookup."""

from __future__ import annotations

import hashlib
import sqlite3
import time
from dataclasses import dataclass

from zero_ttt.data.catalog_types import (
    AnnotationLocator,
    SnapshotStatistics,
    TrajectoryLocator,
)


def _hash_identity_field(digest: hashlib._Hash, value: str) -> None:
    encoded = value.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


@dataclass(frozen=True, slots=True)
class SnapshotSpec:
    seed: int
    split: str
    validation_fraction: float
    source_kind: str | None
    task_id: str | None

    def validate(self) -> None:
        if self.split not in {"train", "validation"}:
            raise ValueError("split must be train or validation")
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1)")
        if self.source_kind not in {None, "external", "selfplay"}:
            raise ValueError("source_kind must be external or selfplay")
        if self.task_id is not None and self.source_kind != "selfplay":
            raise ValueError("task_id filtering requires source_kind=selfplay")


class SnapshotService:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def create(self, spec: SnapshotSpec) -> str:
        spec.validate()
        self._create_build_tables()
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            self._clear_build_tables()
            self._select_trajectories(spec)
            self._select_annotations()
            snapshot_id = self._compute_identity(spec)
            self._persist(snapshot_id, spec)
            self.connection.commit()
            return snapshot_id
        except BaseException:
            self.connection.rollback()
            raise

    def _create_build_tables(self) -> None:
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

    def _clear_build_tables(self) -> None:
        self.connection.execute("DELETE FROM snapshot_build_trajectories")
        self.connection.execute("DELETE FROM snapshot_build_annotations")

    def _select_trajectories(self, spec: SnapshotSpec) -> None:
        threshold = int(spec.validation_fraction * (1 << 64))
        rows = self.connection.execute(
            """
            SELECT game_id,content_sha256
            FROM trajectories WHERE deleted=0
              AND (? IS NULL OR asset_sha256 IN (
                SELECT a.asset_sha256
                FROM assets a
                LEFT JOIN selfplay_tasks st ON st.task_id=a.task_id
                WHERE a.source_kind=?
                  AND (? IS NULL OR a.task_id=?)
                  AND (? != 'selfplay' OR st.status='sealed')
              ))
            ORDER BY game_id
            """,
            (
                spec.source_kind,
                spec.source_kind,
                spec.task_id,
                spec.task_id,
                spec.source_kind,
            ),
        )
        ordinal = 0
        for row in rows:
            digest = hashlib.sha256(f"{spec.seed}:{row['game_id']}".encode("ascii")).digest()
            validation = int.from_bytes(digest[:8], "big") < threshold
            if (spec.split == "validation") != validation:
                continue
            self.connection.execute(
                "INSERT INTO snapshot_build_trajectories"
                "(ordinal,game_id,content_sha256) VALUES(?,?,?)",
                (ordinal, row["game_id"], row["content_sha256"]),
            )
            ordinal += 1

    def _select_annotations(self) -> None:
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

    def _compute_identity(self, spec: SnapshotSpec) -> str:
        identity = hashlib.sha256()
        for field in (
            "zero-ttt-snapshot-v2",
            str(spec.seed),
            spec.split,
            float(spec.validation_fraction).hex(),
        ):
            _hash_identity_field(identity, field)
        if spec.source_kind is not None:
            _hash_identity_field(identity, "source-filter")
            _hash_identity_field(identity, spec.source_kind)
            _hash_identity_field(identity, spec.task_id or "")
        self._hash_build_rows(identity)
        return identity.hexdigest()

    def _hash_build_rows(self, identity: hashlib._Hash) -> None:
        trajectory_rows = self.connection.execute(
            "SELECT game_id,content_sha256 FROM snapshot_build_trajectories ORDER BY ordinal"
        )
        for row in trajectory_rows:
            for value in ("trajectory", row["game_id"], row["content_sha256"]):
                _hash_identity_field(identity, value)
        annotation_rows = self.connection.execute(
            "SELECT game_id,content_sha256,ply,teacher_fingerprint "
            "FROM snapshot_build_annotations ORDER BY game_id,ply,teacher_fingerprint"
        )
        for row in annotation_rows:
            for value in (
                "annotation",
                row["game_id"],
                str(int(row["ply"])),
                row["teacher_fingerprint"],
                row["content_sha256"],
            ):
                _hash_identity_field(identity, value)

    def _persist(self, snapshot_id: str, spec: SnapshotSpec) -> None:
        created = self.connection.execute(
            """
            INSERT OR IGNORE INTO snapshots(
                snapshot_id,seed,split,validation_fraction,source_kind,task_id,created_ns
            ) VALUES(?,?,?,?,?,?,?)
            """,
            (
                snapshot_id,
                spec.seed,
                spec.split,
                spec.validation_fraction,
                spec.source_kind,
                spec.task_id,
                time.time_ns(),
            ),
        ).rowcount
        if not created:
            return
        self.connection.execute(
            "INSERT INTO snapshot_trajectories(snapshot_id,ordinal,game_id) "
            "SELECT ?,ordinal,game_id FROM snapshot_build_trajectories ORDER BY ordinal",
            (snapshot_id,),
        )
        self.connection.execute(
            "INSERT INTO snapshot_annotations(snapshot_id,game_id,ply,teacher_fingerprint) "
            "SELECT ?,game_id,ply,teacher_fingerprint "
            "FROM snapshot_build_annotations ORDER BY game_id,ply,teacher_fingerprint",
            (snapshot_id,),
        )

    def trajectories(self, snapshot_id: str) -> tuple[TrajectoryLocator, ...]:
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
        if not rows and not self._exists(snapshot_id):
            raise KeyError(f"unknown snapshot {snapshot_id}")
        return tuple(TrajectoryLocator(**dict(row)) for row in rows)

    def statistics(self, snapshot_id: str) -> SnapshotStatistics:
        row = self.connection.execute(
            """
            SELECT s.snapshot_id,s.source_kind,s.task_id,
                   COUNT(st.game_id) AS games,
                   COALESCE(SUM(t.trainable_positions), 0) AS positions
            FROM snapshots s
            LEFT JOIN snapshot_trajectories st ON st.snapshot_id=s.snapshot_id
            LEFT JOIN trajectories t ON t.game_id=st.game_id
            WHERE s.snapshot_id=?
            GROUP BY s.snapshot_id,s.source_kind,s.task_id
            """,
            (snapshot_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown snapshot {snapshot_id}")
        return SnapshotStatistics(
            row["snapshot_id"],
            row["source_kind"],
            row["task_id"],
            int(row["games"]),
            int(row["positions"]),
        )

    def selfplay_outside(self, snapshot_id: str | None) -> tuple[int, int]:
        if snapshot_id is not None:
            self.statistics(snapshot_id)
        row = self.connection.execute(
            """
            SELECT COUNT(*) AS games,
                   COALESCE(SUM(t.trainable_positions), 0) AS positions
            FROM trajectories t
            JOIN assets a ON a.asset_sha256=t.asset_sha256
            JOIN selfplay_tasks task ON task.task_id=a.task_id
            LEFT JOIN snapshot_trajectories st
              ON st.game_id=t.game_id AND st.snapshot_id=?
            WHERE t.deleted=0 AND a.source_kind='selfplay'
              AND task.status='sealed' AND st.game_id IS NULL
            """,
            (snapshot_id,),
        ).fetchone()
        return int(row["games"]), int(row["positions"])

    def annotation(
        self,
        game_id: str,
        ply: int,
        teacher_fingerprint: str,
        snapshot_id: str | None,
    ) -> AnnotationLocator | None:
        if snapshot_id is None:
            row = self.connection.execute(
                """
                SELECT a.game_id,a.content_sha256,a.ply,a.teacher_fingerprint,
                       a.shard_sha256,s.relative_path,a.record_index
                FROM annotations a JOIN shards s ON s.shard_sha256=a.shard_sha256
                WHERE a.game_id=? AND a.ply=? AND a.teacher_fingerprint=?
                  AND a.deleted=0 AND s.deleted=0
                """,
                (game_id, ply, teacher_fingerprint),
            ).fetchone()
        else:
            row = self.connection.execute(
                """
                SELECT a.game_id,a.content_sha256,a.ply,a.teacher_fingerprint,
                       a.shard_sha256,s.relative_path,a.record_index
                FROM annotations a
                JOIN shards s ON s.shard_sha256=a.shard_sha256
                JOIN snapshot_annotations sa
                  ON sa.game_id=a.game_id AND sa.ply=a.ply
                 AND sa.teacher_fingerprint=a.teacher_fingerprint
                WHERE a.game_id=? AND a.ply=? AND a.teacher_fingerprint=?
                  AND sa.snapshot_id=?
                """,
                (game_id, ply, teacher_fingerprint, snapshot_id),
            ).fetchone()
        return None if row is None else AnnotationLocator(**dict(row))

    def annotations(
        self,
        snapshot_id: str,
        teacher_fingerprint: str,
    ) -> tuple[AnnotationLocator, ...]:
        rows = self.connection.execute(
            """
            SELECT a.game_id,a.content_sha256,a.ply,a.teacher_fingerprint,
                   a.shard_sha256,s.relative_path,a.record_index
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

    def _exists(self, snapshot_id: str) -> bool:
        row = self.connection.execute(
            "SELECT 1 FROM snapshots WHERE snapshot_id=?",
            (snapshot_id,),
        ).fetchone()
        return row is not None
