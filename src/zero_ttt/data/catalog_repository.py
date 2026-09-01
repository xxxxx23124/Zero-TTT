"""Narrow SQL repository for mutable catalog metadata."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from zero_ttt.data.catalog_types import ImportStatistics, SelfPlayStatistics
from zero_ttt.data.manifest import ManifestAsset, SourceManifest
from zero_ttt.data.records import AnnotationRecord, ImportEvent, TrajectoryRecord
from zero_ttt.data.shards import ShardInfo, ShardStore


class CatalogRepository:
    """Own SQL writes that are not snapshot or shard-lifecycle operations."""

    def __init__(self, connection: sqlite3.Connection, store: ShardStore) -> None:
        self.connection = connection
        self.store = store

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
            self._register_selfplay_asset(
                task_id, manifest_relative_path, manifest_sha256, manifest_size_bytes
            )
            identity = (
                manifest_sha256,
                publication_sha256,
                evaluator_id,
                search_config_sha256,
                requested_games,
            )
            existing = self.connection.execute(
                "SELECT * FROM selfplay_tasks WHERE task_id=?", (task_id,)
            ).fetchone()
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

    def _register_selfplay_asset(
        self,
        task_id: str,
        relative_path: str,
        sha256: str,
        size_bytes: int,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO assets(
                asset_sha256,dataset_id,relative_path,size_bytes,status,source_kind,task_id
            ) VALUES(?,?,?,?, 'verified','selfplay',?)
            ON CONFLICT(asset_sha256) DO UPDATE SET status='verified'
            """,
            (sha256, f"selfplay/{task_id}", relative_path, size_bytes, task_id),
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
                "UPDATE assets SET status=? WHERE asset_sha256=?", (status, asset_sha256)
            ).rowcount
        if not changed:
            raise KeyError(asset_sha256)

    def asset_status(self, asset_sha256: str) -> str:
        row = self.connection.execute(
            "SELECT status FROM assets WHERE asset_sha256=?", (asset_sha256,)
        ).fetchone()
        if row is None:
            raise KeyError(asset_sha256)
        return str(row["status"])

    def has_trajectory(self, game_id: str) -> bool:
        row = self.connection.execute(
            "SELECT 1 FROM trajectories WHERE game_id=?", (game_id,)
        ).fetchone()
        return row is not None

    def commit_trajectory_shard(
        self,
        info: ShardInfo,
        records: list[TrajectoryRecord],
        rejections: list[ImportEvent] | None = None,
    ) -> None:
        if (
            info.kind != "trajectory"
            or len(records) != info.record_count
            or sum(record.trainable_position_count for record in records) != info.position_count
        ):
            raise ValueError("trajectory shard metadata does not match its records")
        stored_records = self.store.read_verified_trajectories(info.relative_path, info.sha256)
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

    def commit_annotation_shard(self, info: ShardInfo, records: list[AnnotationRecord]) -> None:
        if (
            info.kind != "annotation"
            or len(records) != info.record_count
            or info.position_count != len(records)
        ):
            raise ValueError("annotation shard metadata does not match its records")
        stored_records = self.store.read_verified_annotations(info.relative_path, info.sha256)
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
            if event.asset_sha256 is None or not self._asset_exists(event.asset_sha256):
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

    def _asset_exists(self, asset_sha256: str) -> bool:
        row = self.connection.execute(
            "SELECT 1 FROM assets WHERE asset_sha256=?", (asset_sha256,)
        ).fetchone()
        return row is not None

    def selfplay_statistics(self) -> SelfPlayStatistics:
        task_counts = {
            row["status"]: int(row["count"])
            for row in self.connection.execute(
                "SELECT status,COUNT(*) AS count FROM selfplay_tasks GROUP BY status"
            )
        }
        trajectory = self.connection.execute(
            """
            SELECT COUNT(*) AS games,
                   COALESCE(SUM(t.trainable_positions), 0) AS positions
            FROM trajectories t
            JOIN assets a ON a.asset_sha256=t.asset_sha256
            JOIN selfplay_tasks st ON st.task_id=a.task_id
            WHERE t.deleted=0 AND a.source_kind='selfplay' AND st.status='sealed'
            """
        ).fetchone()
        return SelfPlayStatistics(
            sealed_tasks=task_counts.get("sealed", 0),
            collecting_tasks=task_counts.get("collecting", 0),
            failed_tasks=task_counts.get("failed", 0),
            games=int(trajectory["games"]),
            positions=int(trajectory["positions"]),
        )

    def import_statistics(self, dataset_id: str) -> ImportStatistics:
        status_counts = {
            str(row["status"]): int(row["count"])
            for row in self.connection.execute(
                "SELECT status,COUNT(*) AS count FROM assets "
                "WHERE dataset_id=? AND source_kind='external' GROUP BY status",
                (dataset_id,),
            )
        }
        row = self.connection.execute(
            """
            SELECT COUNT(DISTINCT t.game_id) AS games,
                   COALESCE(SUM(t.trainable_positions), 0) AS positions,
                   COUNT(DISTINCT t.shard_sha256) AS shards
            FROM trajectories t
            JOIN assets a ON a.asset_sha256=t.asset_sha256
            WHERE a.dataset_id=? AND a.source_kind='external' AND t.deleted=0
            """,
            (dataset_id,),
        ).fetchone()
        return ImportStatistics(
            verified_assets=status_counts.get("verified", 0),
            partial_assets=status_counts.get("partial", 0),
            imported_assets=status_counts.get("imported", 0),
            games=int(row["games"]),
            positions=int(row["positions"]),
            shards=int(row["shards"]),
        )

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
            expected = {
                "samples_seen": samples_seen,
                "relative_path": relative_path,
                "sha256": sha256,
            }
            if existing is not None:
                if dict(existing) != expected:
                    raise ValueError("conflicting publication catalog entry")
                return
            self.connection.execute(
                """
                INSERT INTO publications(run_id,optimizer_step,samples_seen,relative_path,sha256)
                VALUES(?,?,?,?,?)
                """,
                (run_id, optimizer_step, samples_seen, relative_path, sha256),
            )
