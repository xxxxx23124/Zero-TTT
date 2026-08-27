"""Orchestration layer joining pure importers to storage transactions."""

from __future__ import annotations

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


def import_manifest(
    manifest_path: str | Path,
    source_root: str | Path,
    store_root: str | Path,
    catalog_path: str | Path,
    max_accepted: int | None = None,
    target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
    importer_registry: ImporterRegistry = DEFAULT_IMPORTERS,
) -> ImportSummary:
    if max_accepted is not None and max_accepted <= 0:
        raise ValueError("max_accepted must be positive")
    if target_shard_bytes <= 0:
        raise ValueError("target_shard_bytes must be positive")
    manifest = SourceManifest.load(manifest_path)
    manifest.verify(source_root)
    importer = importer_registry.resolve(manifest.source_type)
    store = ShardStore(store_root)
    accepted = duplicate = rejected = 0

    with Catalog(catalog_path, store) as catalog:
        catalog.recover()
        sink = TrajectoryShardSink(store, catalog, target_shard_bytes)

        for asset in manifest.assets:
            if max_accepted is not None and accepted >= max_accepted:
                break
            catalog.register_asset(manifest, asset)
            asset_complete = True
            for event in importer.import_asset(manifest, asset, source_root):
                if event.kind == "reject":
                    sink.add_rejection(event)
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
                sink.append(record)
                accepted += 1
            sink.flush()
            catalog.set_asset_status(asset.sha256, "imported" if asset_complete else "partial")
            if not asset_complete:
                break
    return ImportSummary(
        accepted=accepted,
        duplicate=duplicate,
        rejected=rejected,
        shards=sink.shard_count,
        trainable_positions=sink.position_count,
    )
