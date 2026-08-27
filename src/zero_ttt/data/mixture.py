"""Immutable weighted mixtures of snapshot-backed batch sources."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from zero_ttt.data.catalog_source import CatalogBatchSource
from zero_ttt.data.contracts import TrainBatch


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
        if self.schema_version != 1 or not self.components:
            raise ValueError("unsupported or empty training mixture")
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
        encoded = json.dumps(
            self._payload(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            **self._payload(),
            "content_sha256": self.content_sha256,
        }
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()

    @classmethod
    def load(cls, path: str | Path) -> "TrainingMixtureManifest":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
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
        sources = []
        try:
            for component in manifest.components:
                sources.append(
                    CatalogBatchSource(
                        catalog_path,
                        store_root,
                        component.snapshot_id,
                        shard_cache_size=shard_cache_size,
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
        identity = json.dumps(
            {
                "manifest_sha256": manifest.content_sha256,
                "source_sampling": [source.sampling_config_sha256 for source in self.sources],
                "microbatch_component_selection": "weighted-with-replacement-v1",
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.sampling_config_sha256 = hashlib.sha256(identity).hexdigest()
        self.component_snapshot_ids = tuple(
            component.snapshot_id for component in manifest.components
        )

    def next_batch(self, batch_size: int, rng: np.random.Generator) -> TrainBatch:
        index = int(rng.choice(len(self.sources), p=self.weights))
        return self.sources[index].next_batch(batch_size, rng)

    def close(self) -> None:
        for source in self.sources:
            source.close()

    def __enter__(self) -> "MixtureBatchSource":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
