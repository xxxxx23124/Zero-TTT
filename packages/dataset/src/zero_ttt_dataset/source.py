"""Batch sources built solely from immutable snapshot manifests and shards."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from zero_ttt_contracts.hashing import canonical_json_bytes, sha256_bytes

from zero_ttt_dataset.contracts import TrainBatch
from zero_ttt_dataset.manifest import SnapshotManifest
from zero_ttt_dataset.materialization import TrajectoryBatchMaterializer
from zero_ttt_dataset.sampling import AnnotationMode, SnapshotPositionIndex
from zero_ttt_dataset.shards import ShardStore


class PortableSnapshotBatchSource:
    def __init__(
        self,
        manifest: SnapshotManifest,
        shard_root: str | Path,
        *,
        shard_cache_size: int = 4,
        annotation_mode: AnnotationMode = "none",
        teacher_fingerprint: str | None = None,
    ) -> None:
        if annotation_mode != "none" and not teacher_fingerprint:
            raise ValueError("annotation modes require a teacher fingerprint")
        self.manifest = manifest
        trajectories = tuple(item.locator() for item in manifest.trajectories)
        annotations = tuple(
            item.locator()
            for item in manifest.annotations
            if teacher_fingerprint is None or item.teacher_fingerprint == teacher_fingerprint
        )
        self.position_index = SnapshotPositionIndex(trajectories, annotations, annotation_mode)
        self.position_count = self.position_index.position_count
        self.store = ShardStore(shard_root, read_only=True)
        self.materializer = TrajectoryBatchMaterializer(self.store, shard_cache_size)
        identity = {
            "snapshot_sha256": manifest.content_sha256,
            "annotation_mode": annotation_mode,
            "teacher_fingerprint": teacher_fingerprint,
            "d4": True,
            "microbatch_shards": 1,
            "sampling": "weighted-shard-local-position-with-replacement-v2",
        }
        self.sampling_config_sha256 = sha256_bytes(canonical_json_bytes(identity))

    def close(self) -> None:
        return None

    def __enter__(self) -> PortableSnapshotBatchSource:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def next_batch(self, batch_size: int, rng: np.random.Generator) -> TrainBatch:
        references = self.position_index.draw_batch(batch_size, rng)
        return self.materializer.materialize(references, rng)


class PortableMixtureBatchSource:
    def __init__(
        self,
        components: tuple[tuple[PortableSnapshotBatchSource, float], ...],
    ) -> None:
        if not components or any(weight <= 0 for _, weight in components):
            raise ValueError("mixture components and weights must be positive")
        self.components = components
        weights = np.asarray([weight for _, weight in components], dtype=np.float64)
        self.weights = weights / weights.sum()
        self.component_snapshot_ids = tuple(source.manifest.snapshot_id for source, _ in components)
        identity = {
            "components": [
                {"snapshot": source.manifest.content_sha256, "weight": float(weight)}
                for source, weight in components
            ]
        }
        self.sampling_config_sha256 = sha256_bytes(canonical_json_bytes(identity))

    def close(self) -> None:
        for source, _ in self.components:
            source.close()

    def __enter__(self) -> PortableMixtureBatchSource:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def next_batch(self, batch_size: int, rng: np.random.Generator) -> TrainBatch:
        index = int(rng.choice(len(self.components), p=self.weights))
        return self.components[index][0].next_batch(batch_size, rng)
