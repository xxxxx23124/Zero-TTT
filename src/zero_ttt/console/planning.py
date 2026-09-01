"""Training-data planning for cold-start and self-play mixture phases."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from zero_ttt.config import ExperimentConfig
from zero_ttt.console.config import ConsoleConfig
from zero_ttt.console.state import TrainingPhase
from zero_ttt.data.catalog import Catalog
from zero_ttt.data.catalog_source import CatalogBatchSource
from zero_ttt.data.mixture import (
    MixtureBatchSource,
    MixtureComponent,
    TrainingMixtureManifest,
)
from zero_ttt.data.shards import ShardStore
from zero_ttt.training.contracts import LearnerDataIdentity
from zero_ttt.versioning import TRAINING_MIXTURE_SCHEMA

TrainingSource = CatalogBatchSource | MixtureBatchSource


@dataclass(slots=True)
class TrainingDataPlan:
    source: TrainingSource
    identity: LearnerDataIdentity
    target_phase: TrainingPhase
    mixture_manifest: TrainingMixtureManifest | None = None
    selfplay_snapshot_id: str = ""

    def close(self) -> None:
        self.source.close()

    def __enter__(self) -> TrainingDataPlan:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class TrainingDataPlanner:
    def __init__(
        self,
        settings: ConsoleConfig,
        config: ExperimentConfig,
        console_dir: str | Path,
    ) -> None:
        self.settings = settings
        self.config = config
        self.console_dir = Path(console_dir)

    def _catalog(self) -> Catalog:
        return Catalog(
            self.settings.catalog_path,
            ShardStore(self.settings.store_root),
        )

    def build(self, *, use_mixture: bool) -> TrainingDataPlan:
        return self._mixture() if use_mixture else self._cold_start()

    def _cold_start(self) -> TrainingDataPlan:
        source = CatalogBatchSource(
            self.settings.catalog_path,
            self.settings.store_root,
            self.settings.cold_start_snapshot_id,
        )
        return TrainingDataPlan(
            source=source,
            identity=LearnerDataIdentity(
                snapshot_id=self.settings.cold_start_snapshot_id,
                sampling_config_sha256=source.sampling_config_sha256,
            ),
            target_phase=TrainingPhase.COLD_START,
        )

    def _mixture(self) -> TrainingDataPlan:
        with self._catalog() as catalog:
            if catalog.selfplay_statistics().games <= 0:
                raise RuntimeError("mixture training requires at least one sealed self-play game")
            selfplay_snapshot = catalog.create_snapshot(
                self.config.seed,
                split="train",
                validation_fraction=0.0,
                source_kind="selfplay",
            )
        manifest = TrainingMixtureManifest(
            TRAINING_MIXTURE_SCHEMA.current,
            (
                MixtureComponent(
                    selfplay_snapshot,
                    self.config.training.mixture.selfplay_weight,
                ),
                MixtureComponent(
                    self.settings.cold_start_snapshot_id,
                    self.config.training.mixture.cold_start_weight,
                ),
            ),
        )
        manifest.save(self.console_dir / "mixtures" / f"{manifest.content_sha256}.json")
        source = MixtureBatchSource(
            self.settings.catalog_path,
            self.settings.store_root,
            manifest,
        )
        return TrainingDataPlan(
            source=source,
            identity=LearnerDataIdentity(
                snapshot_id=f"mixture:{manifest.content_sha256}",
                sampling_config_sha256=source.sampling_config_sha256,
                mixture_manifest_sha256=manifest.content_sha256,
                component_snapshot_ids=source.component_snapshot_ids,
            ),
            target_phase=TrainingPhase.MIXTURE,
            mixture_manifest=manifest,
            selfplay_snapshot_id=selfplay_snapshot,
        )
