"""Orchestration layer joining pure importers to storage transactions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from zero_ttt.data.catalog import Catalog
from zero_ttt.data.importers import DEFAULT_IMPORTERS, ImporterRegistry
from zero_ttt.data.ingestion import DEFAULT_TARGET_SHARD_BYTES, TrajectoryShardSink
from zero_ttt.data.manifest import SourceManifest
from zero_ttt.data.records import TrajectoryRecord
from zero_ttt.data.shards import ShardStore


@dataclass(frozen=True, slots=True)
class ImportSummary:
    accepted: int
    duplicate: int
    rejected: int
    shards: int
    trainable_positions: int


@dataclass(slots=True)
class _ImportCounts:
    accepted: int = 0
    duplicate: int = 0
    rejected: int = 0


def _limit_reached(counts: _ImportCounts, maximum: int | None) -> bool:
    return maximum is not None and counts.accepted >= maximum


def _import_asset(
    *,
    manifest: SourceManifest,
    asset,
    source_root: str | Path,
    importer,
    catalog: Catalog,
    sink: TrajectoryShardSink,
    counts: _ImportCounts,
    max_accepted: int | None,
    stop_requested: Callable[[], bool] | None,
) -> bool:
    catalog.register_asset(manifest, asset)
    complete = True
    for event in importer.import_asset(manifest, asset, source_root):
        if stop_requested is not None and stop_requested():
            complete = False
            break
        if event.kind == "reject":
            sink.add_rejection(event)
            counts.rejected += 1
            continue
        record = event.record
        if not isinstance(record, TrajectoryRecord):
            raise TypeError("KataGo importer emitted a non-trajectory record")
        if catalog.has_trajectory(record.game_id):
            counts.duplicate += 1
            continue
        if _limit_reached(counts, max_accepted):
            complete = False
            break
        sink.append(record)
        counts.accepted += 1
    sink.flush()
    catalog.set_asset_status(asset.sha256, "imported" if complete else "partial")
    return complete


def import_manifest(
    manifest_path: str | Path,
    source_root: str | Path,
    store_root: str | Path,
    catalog_path: str | Path,
    max_accepted: int | None = None,
    target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
    importer_registry: ImporterRegistry = DEFAULT_IMPORTERS,
    *,
    progress: Callable[[str, int, int, str], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> ImportSummary:
    if max_accepted is not None and max_accepted <= 0:
        raise ValueError("max_accepted must be positive")
    if target_shard_bytes <= 0:
        raise ValueError("target_shard_bytes must be positive")
    manifest = SourceManifest.load(manifest_path)
    manifest.verify(source_root)
    importer = importer_registry.resolve(manifest.source_type)
    store = ShardStore(store_root)
    counts = _ImportCounts()

    with Catalog(catalog_path, store) as catalog:
        catalog.recover()
        sink = TrajectoryShardSink(store, catalog, target_shard_bytes)

        for asset_index, asset in enumerate(manifest.assets, start=1):
            if _limit_reached(counts, max_accepted):
                break
            if stop_requested is not None and stop_requested():
                break
            if progress is not None:
                progress("importing", asset_index - 1, len(manifest.assets), asset.relative_path)
            asset_complete = _import_asset(
                manifest=manifest,
                asset=asset,
                source_root=source_root,
                importer=importer,
                catalog=catalog,
                sink=sink,
                counts=counts,
                max_accepted=max_accepted,
                stop_requested=stop_requested,
            )
            if progress is not None:
                progress("importing", asset_index, len(manifest.assets), asset.relative_path)
            if not asset_complete:
                break
    return ImportSummary(
        accepted=counts.accepted,
        duplicate=counts.duplicate,
        rejected=counts.rejected,
        shards=sink.shard_count,
        trainable_positions=sink.position_count,
    )
