"""Admission of sealed self-play bundles into the Data Service's owned catalog."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from zero_ttt._io import fsync_directory
from zero_ttt_contracts import ArtifactRef
from zero_ttt_contracts.hashing import sha256_file
from zero_ttt_dataset import LocalArtifactStore
from zero_ttt_dataset.selfplay import SelfPlayBundle
from zero_ttt_dataset.shards import ShardInfo

from zero_ttt_data.unit_of_work import DataUnitOfWork


def admit_bundle(
    reference: ArtifactRef,
    *,
    artifact_store: LocalArtifactStore,
    database_path: str | Path,
    shard_root: str | Path,
    task_id: str,
) -> dict[str, object]:
    bundle_path = artifact_store.verify(reference)
    bundle = SelfPlayBundle.model_validate_json(bundle_path.read_text(encoding="utf-8"))
    if bundle.task_id != task_id:
        raise ValueError("self-play bundle task does not match workflow identity")
    source_manifest = artifact_store.resolve(bundle.source_manifest_uri)
    if (
        not source_manifest.is_file()
        or source_manifest.stat().st_size != bundle.source_manifest_size_bytes
        or sha256_file(source_manifest) != bundle.source_manifest_sha256
    ):
        raise ValueError("self-play source manifest integrity check failed")
    with DataUnitOfWork(database_path, shard_root) as unit:
        unit.repository.register_selfplay_task(
            task_id=task_id,
            manifest_relative_path=bundle.source_manifest_uri,
            manifest_sha256=bundle.source_manifest_sha256,
            manifest_size_bytes=bundle.source_manifest_size_bytes,
            publication_sha256=bundle.publication_sha256,
            evaluator_id=bundle.evaluator_id,
            search_config_sha256=bundle.search_config_sha256,
            requested_games=bundle.requested_games,
        )
        admitted = 0
        positions = 0
        for shard in bundle.shards:
            source = artifact_store.resolve(shard.uri)
            if source.stat().st_size != shard.size_bytes or sha256_file(source) != shard.sha256:
                raise ValueError("self-play shard integrity check failed")
            records = unit.store.read_trajectories(source)
            if any(record.asset_sha256 != bundle.source_manifest_sha256 for record in records):
                raise ValueError("self-play record does not reference the sealed source manifest")
            destination = unit.store.trajectory_dir / f"{shard.sha256}.npz"
            if destination.exists():
                if sha256_file(destination) != shard.sha256:
                    raise ValueError("existing admitted shard conflicts with bundle content")
            else:
                descriptor, temporary_name = tempfile.mkstemp(
                    prefix=".admit-", suffix=".tmp", dir=unit.store.trajectory_dir
                )
                os.close(descriptor)
                temporary = Path(temporary_name)
                try:
                    shutil.copyfile(source, temporary)
                    with temporary.open("r+b") as handle:
                        os.fsync(handle.fileno())
                    os.replace(temporary, destination)
                    fsync_directory(destination.parent)
                finally:
                    if temporary.exists():
                        temporary.unlink()
            info = ShardInfo(
                kind="trajectory",
                sha256=shard.sha256,
                relative_path=destination.relative_to(unit.store.root).as_posix(),
                size_bytes=destination.stat().st_size,
                record_count=len(records),
                position_count=sum(record.trainable_position_count for record in records),
            )
            unit.repository.commit_trajectory_shard(info, list(records))
            admitted += len(records)
            positions += info.position_count
        unit.repository.set_selfplay_task_status(task_id, "sealed")
    return {"task_id": task_id, "games": admitted, "positions": positions}
