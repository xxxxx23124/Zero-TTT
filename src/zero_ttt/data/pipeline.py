"""Orchestration layer joining pure importers to storage transactions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from zero_ttt.data.catalog import Catalog
from zero_ttt.data.importers import KataGoSgfImporter
from zero_ttt.data.manifest import SourceManifest
from zero_ttt.data.records import ImportEvent, TrajectoryRecord
from zero_ttt.data.shards import ShardStore


DEFAULT_TARGET_SHARD_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ImportSummary:
    accepted: int
    duplicate: int
    rejected: int
    shards: int
    trainable_positions: int


def _estimated_record_bytes(record: TrajectoryRecord) -> int:
    return 2048 + 8 * len(record.moves) + 12 * record.trainable_position_count


def import_manifest(
    manifest_path: str | Path,
    source_root: str | Path,
    store_root: str | Path,
    catalog_path: str | Path,
    max_accepted: int | None = None,
    target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
) -> ImportSummary:
    if max_accepted is not None and max_accepted <= 0:
        raise ValueError("max_accepted must be positive")
    if target_shard_bytes <= 0:
        raise ValueError("target_shard_bytes must be positive")
    manifest = SourceManifest.load(manifest_path)
    manifest.verify(source_root)
    importer = KataGoSgfImporter()
    store = ShardStore(store_root)
    accepted = duplicate = rejected = shards = positions = 0
    pending: list[TrajectoryRecord] = []
    pending_bytes = 0
    pending_rejections: list[ImportEvent] = []

    with Catalog(catalog_path, store) as catalog:
        catalog.recover()

        def flush() -> None:
            nonlocal pending, pending_bytes, pending_rejections, shards, positions
            if not pending:
                if pending_rejections:
                    catalog.record_rejections(pending_rejections)
                    pending_rejections = []
                return
            info = store.write_trajectories(pending)
            catalog.commit_trajectory_shard(info, pending, pending_rejections)
            shards += 1
            positions += info.position_count
            pending = []
            pending_bytes = 0
            pending_rejections = []

        for asset in manifest.assets:
            if max_accepted is not None and accepted >= max_accepted:
                break
            catalog.register_asset(manifest, asset)
            asset_complete = True
            for event in importer.import_asset(manifest, asset, source_root):
                if event.kind == "reject":
                    pending_rejections.append(event)
                    rejected += 1
                    continue
                record = event.record
                if not isinstance(record, TrajectoryRecord):
                    raise TypeError("KataGo importer emitted a non-trajectory record")
                if catalog.has_trajectory(record.game_id):
                    duplicate += 1
                    continue
                if max_accepted is not None and accepted >= max_accepted:
                    asset_complete = False
                    break
                estimate = _estimated_record_bytes(record)
                if pending and pending_bytes + estimate > target_shard_bytes:
                    flush()
                pending.append(record)
                pending_bytes += estimate
                accepted += 1
            flush()
            catalog.set_asset_status(asset.sha256, "imported" if asset_complete else "partial")
            if not asset_complete:
                break
    return ImportSummary(
        accepted=accepted,
        duplicate=duplicate,
        rejected=rejected,
        shards=shards,
        trainable_positions=positions,
    )
