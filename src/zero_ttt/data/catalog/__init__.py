"""Compatibility facade for the decomposed SQLite data control plane."""

from __future__ import annotations

from pathlib import Path

from zero_ttt.data.catalog_maintenance import (
    ShardLifecycle,
)
from zero_ttt.data.catalog_repository import CatalogRepository
from zero_ttt.data.catalog_session import CatalogSession
from zero_ttt.data.catalog_snapshots import SnapshotService, SnapshotSpec
from zero_ttt.data.catalog_types import (
    AnnotationLocator,
    ImportStatistics,
    SelfPlayStatistics,
    SnapshotStatistics,
    SnapshotSummary,
    TrajectoryLocator,
)
from zero_ttt.data.manifest import ManifestAsset, SourceManifest
from zero_ttt.data.records import AnnotationRecord, ImportEvent, TrajectoryRecord
from zero_ttt.data.shards import ShardInfo, ShardStore

__all__ = [
    "AnnotationLocator",
    "Catalog",
    "ImportStatistics",
    "SelfPlayStatistics",
    "SnapshotStatistics",
    "SnapshotSummary",
    "TrajectoryLocator",
]


class Catalog:
    """Stable public API backed by focused internal services."""

    def __init__(self, path: str | Path, store: ShardStore) -> None:
        self.session = CatalogSession(path)
        self.path = self.session.path
        self.store = store
        self.connection = self.session.connection
        self.repository = CatalogRepository(self.connection, store)
        self.snapshots = SnapshotService(self.connection)
        self.shards = ShardLifecycle(self.connection, store)

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> Catalog:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def register_asset(self, manifest: SourceManifest, asset: ManifestAsset) -> None:
        self.repository.register_asset(manifest, asset)

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
        self.repository.register_selfplay_task(
            task_id=task_id,
            manifest_relative_path=manifest_relative_path,
            manifest_sha256=manifest_sha256,
            manifest_size_bytes=manifest_size_bytes,
            publication_sha256=publication_sha256,
            evaluator_id=evaluator_id,
            search_config_sha256=search_config_sha256,
            requested_games=requested_games,
        )

    def set_selfplay_task_status(self, task_id: str, status: str) -> None:
        self.repository.set_selfplay_task_status(task_id, status)

    def set_asset_status(self, asset_sha256: str, status: str) -> None:
        self.repository.set_asset_status(asset_sha256, status)

    def asset_status(self, asset_sha256: str) -> str:
        return self.repository.asset_status(asset_sha256)

    def has_trajectory(self, game_id: str) -> bool:
        return self.repository.has_trajectory(game_id)

    def commit_trajectory_shard(
        self,
        info: ShardInfo,
        records: list[TrajectoryRecord],
        rejections: list[ImportEvent] | None = None,
    ) -> None:
        self.repository.commit_trajectory_shard(info, records, rejections)

    def commit_annotation_shard(self, info: ShardInfo, records: list[AnnotationRecord]) -> None:
        self.repository.commit_annotation_shard(info, records)

    def record_rejections(self, events: list[ImportEvent]) -> None:
        self.repository.record_rejections(events)

    def create_snapshot(
        self,
        seed: int,
        split: str = "train",
        validation_fraction: float = 0.1,
        source_kind: str | None = None,
        task_id: str | None = None,
    ) -> str:
        return self.snapshots.create(
            SnapshotSpec(seed, split, validation_fraction, source_kind, task_id)
        )

    def snapshot_trajectories(self, snapshot_id: str) -> tuple[TrajectoryLocator, ...]:
        return self.snapshots.trajectories(snapshot_id)

    def selfplay_statistics(self) -> SelfPlayStatistics:
        return self.repository.selfplay_statistics()

    def import_statistics(self, dataset_id: str) -> ImportStatistics:
        return self.repository.import_statistics(dataset_id)

    def snapshot_statistics(self, snapshot_id: str) -> SnapshotStatistics:
        return self.snapshots.statistics(snapshot_id)

    def list_snapshots(self) -> tuple[SnapshotSummary, ...]:
        return self.snapshots.list()

    def selfplay_outside_snapshot(self, snapshot_id: str | None) -> tuple[int, int]:
        return self.snapshots.selfplay_outside(snapshot_id)

    def annotation_locator(
        self,
        game_id: str,
        ply: int,
        teacher_fingerprint: str,
        snapshot_id: str | None = None,
    ) -> AnnotationLocator | None:
        return self.snapshots.annotation(game_id, ply, teacher_fingerprint, snapshot_id)

    def snapshot_annotations(
        self, snapshot_id: str, teacher_fingerprint: str
    ) -> tuple[AnnotationLocator, ...]:
        return self.snapshots.annotations(snapshot_id, teacher_fingerprint)

    def verify(self) -> None:
        self.shards.verify()

    def recover(self) -> tuple[str, ...]:
        return self.shards.recover()

    def mark_trajectory_deleted(self, game_id: str) -> None:
        self.shards.mark_trajectory_deleted(game_id)

    def mark_annotation_deleted(self, game_id: str, ply: int, teacher_fingerprint: str) -> None:
        self.shards.mark_annotation_deleted(game_id, ply, teacher_fingerprint)

    def pin_shard(self, owner: str, shard_sha256: str) -> None:
        self.shards.pin(owner, shard_sha256)

    def unpin_shard(self, owner: str, shard_sha256: str) -> None:
        self.shards.unpin(owner, shard_sha256)

    def garbage_collect(self) -> tuple[str, ...]:
        return self.shards.garbage_collect()

    def record_publication(
        self,
        run_id: str,
        optimizer_step: int,
        samples_seen: int,
        relative_path: str,
        sha256: str,
    ) -> None:
        self.repository.record_publication(
            run_id, optimizer_step, samples_seen, relative_path, sha256
        )
