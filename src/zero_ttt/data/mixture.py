"""Immutable weighted mixtures of snapshot-backed batch sources."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from zero_ttt._io import atomic_write_json, canonical_json_bytes, sha256_bytes
from zero_ttt.data.catalog_source import CatalogBatchSource, TrajectoryBatchMaterializer
from zero_ttt.data.contracts import TrainBatch
from zero_ttt.data.shards import ShardStore
from zero_ttt.versioning import TRAINING_MIXTURE_SCHEMA


@dataclass(frozen=True, slots=True)
class MixtureComponent:
    snapshot_id: str
    weight: float

    def __post_init__(self) -> None:
        if re.fullmatch(r"[0-9a-f]{64}", self.snapshot_id) is None:
            raise ValueError("mixture snapshot_id must be a lowercase SHA-256 hex string")
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("mixture component weight must be finite and positive")


@dataclass(frozen=True, slots=True)
class TrainingMixtureManifest:
    schema_version: int
    components: tuple[MixtureComponent, ...]
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        TRAINING_MIXTURE_SCHEMA.require(self.schema_version)
        if not self.components:
            raise ValueError("training mixture cannot be empty")
        identities = [component.snapshot_id for component in self.components]
        if len(identities) != len(set(identities)):
            raise ValueError("mixture snapshot components must be unique")
        object.__setattr__(self, "content_sha256", self.compute_sha256())

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "components": [
                {"snapshot_id": component.snapshot_id, "weight": component.weight}
                for component in self.components
            ],
        }

    def compute_sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self._payload()))

    def save(self, path: str | Path) -> None:
        payload = {
            **self._payload(),
            "content_sha256": self.content_sha256,
        }
        atomic_write_json(path, payload)

    @classmethod
    def load(cls, path: str | Path) -> TrainingMixtureManifest:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("invalid training mixture manifest")
        TRAINING_MIXTURE_SCHEMA.require(payload.get("schema_version"))
        try:
            manifest = cls(
                schema_version=int(payload["schema_version"]),
                components=tuple(
                    MixtureComponent(str(item["snapshot_id"]), float(item["weight"]))
                    for item in payload["components"]
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid training mixture manifest") from error
        if payload.get("content_sha256") != manifest.content_sha256:
            raise ValueError("training mixture manifest SHA-256 does not match")
        return manifest


class MixtureBatchSource:
    """Choose one snapshot source per microbatch using immutable weights."""

    def __init__(
        self,
        catalog_path: str | Path,
        store_root: str | Path,
        manifest: TrainingMixtureManifest,
        *,
        shard_cache_size: int = 4,
    ) -> None:
        self.manifest = manifest
        store = ShardStore(store_root)
        materializer = TrajectoryBatchMaterializer(store, shard_cache_size)
        sources = []
        try:
            for component in manifest.components:
                sources.append(
                    CatalogBatchSource(
                        catalog_path,
                        store_root,
                        component.snapshot_id,
                        shard_cache_size=shard_cache_size,
                        _store=store,
                        _materializer=materializer,
                    )
                )
        except BaseException:
            for source in sources:
                source.close()
            raise
        self.sources = tuple(sources)
        weights = np.asarray(
            [component.weight for component in manifest.components], dtype=np.float64
        )
        self.weights = weights / weights.sum()
        identity = {
            "manifest_sha256": manifest.content_sha256,
            "source_sampling": [source.sampling_config_sha256 for source in self.sources],
            "microbatch_component_selection": "weighted-with-replacement-v1",
        }
        self.sampling_config_sha256 = sha256_bytes(canonical_json_bytes(identity))
        self.component_snapshot_ids = tuple(
            component.snapshot_id for component in manifest.components
        )

    def next_batch(self, batch_size: int, rng: np.random.Generator) -> TrainBatch:
        index = int(rng.choice(len(self.sources), p=self.weights))
        return self.sources[index].next_batch(batch_size, rng)

    def close(self) -> None:
        for source in self.sources:
            source.close()

    def __enter__(self) -> MixtureBatchSource:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
