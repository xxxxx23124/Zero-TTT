"""External import use case using the Data Service's private unit of work."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from zero_ttt_dataset.records import ImportEvent, TrajectoryRecord
from zero_ttt_dataset.shards import ShardInfo
from zero_ttt_dataset.writer import DEFAULT_TARGET_SHARD_BYTES, estimate_trajectory_bytes

from zero_ttt_data.importers import DEFAULT_IMPORTERS, ImporterRegistry
from zero_ttt_data.manifest import SourceManifest
from zero_ttt_data.unit_of_work import DataUnitOfWork


@dataclass(frozen=True, slots=True)
class ImportSummary:
    accepted: int
    duplicate: int
    rejected: int
    shards: int
    trainable_positions: int


class _ShardSink:
    def __init__(self, unit: DataUnitOfWork, target_bytes: int) -> None:
        self.unit = unit
        self.target_bytes = target_bytes
        self.records: list[TrajectoryRecord] = []
        self.rejections: list[ImportEvent] = []
        self.estimated_bytes = 0
        self.shards = 0
        self.positions = 0

    def reject(self, event: ImportEvent) -> None:
        self.rejections.append(event)

    def append(self, record: TrajectoryRecord) -> None:
        estimate = estimate_trajectory_bytes(record)
        if self.records and self.estimated_bytes + estimate > self.target_bytes:
            self.flush()
        self.records.append(record)
        self.estimated_bytes += estimate

    def flush(self) -> ShardInfo | None:
        if not self.records:
            if self.rejections:
                self.unit.repository.record_rejections(self.rejections)
                self.rejections = []
            return None
        info = self.unit.store.write_trajectories(self.records)
        self.unit.repository.commit_trajectory_shard(info, self.records, self.rejections)
        self.records = []
        self.rejections = []
        self.estimated_bytes = 0
        self.shards += 1
        self.positions += info.position_count
        return info


def import_source(  # noqa: C901 - streaming state transitions are intentionally linear
    *,
    manifest_path: str | Path,
    source_root: str | Path,
    database_path: str | Path,
    shard_root: str | Path,
    max_accepted: int | None,
    target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
    importer_registry: ImporterRegistry = DEFAULT_IMPORTERS,
    progress: Callable[[str, int, int, str], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> ImportSummary:
    if max_accepted is not None and max_accepted <= 0:
        raise ValueError("max_accepted must be positive")
    manifest = SourceManifest.load(manifest_path)
    manifest.verify(source_root, progress=progress, stop_requested=stop_requested)
    importer = importer_registry.resolve(manifest.source_type)
    accepted = duplicate = rejected = 0
    with DataUnitOfWork(database_path, shard_root) as unit:
        unit.lifecycle.recover()
        sink = _ShardSink(unit, target_shard_bytes)
        for asset_index, asset in enumerate(manifest.assets, start=1):
            if max_accepted is not None and accepted >= max_accepted:
                break
            if stop_requested is not None and stop_requested():
                break
            if progress:
                progress("importing", asset_index - 1, len(manifest.assets), asset.relative_path)
            unit.repository.register_asset(manifest, asset)
            complete = True
            for event in importer.import_asset(manifest, asset, source_root):
                if stop_requested is not None and stop_requested():
                    complete = False
                    break
                if event.kind == "reject":
                    sink.reject(event)
                    rejected += 1
                    continue
                record = event.record
                if not isinstance(record, TrajectoryRecord):
                    raise TypeError("importer emitted a non-trajectory record")
                if unit.repository.has_trajectory(record.game_id):
                    duplicate += 1
                    continue
                if max_accepted is not None and accepted >= max_accepted:
                    complete = False
                    break
                sink.append(record)
                accepted += 1
            sink.flush()
            unit.repository.set_asset_status(asset.sha256, "imported" if complete else "partial")
            if progress:
                progress("importing", asset_index, len(manifest.assets), asset.relative_path)
            if not complete:
                break
        return ImportSummary(accepted, duplicate, rejected, sink.shards, sink.positions)
