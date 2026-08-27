"""Versioned training-data contracts, storage, and batch sources."""

from zero_ttt.data.catalog import Catalog, SelfPlayStatistics, SnapshotStatistics
from zero_ttt.data.catalog_source import CatalogBatchSource
from zero_ttt.data.contracts import BatchSource, TrainBatch
from zero_ttt.data.manifest import ManifestAsset, SourceManifest
from zero_ttt.data.mixture import (
    MixtureBatchSource,
    MixtureComponent,
    TrainingMixtureManifest,
)
from zero_ttt.data.records import AnnotationRecord, ImportEvent, TrajectoryRecord
from zero_ttt.data.shards import ShardStore
from zero_ttt.data.synthetic import SyntheticBatchSource

__all__ = [
    "AnnotationRecord",
    "BatchSource",
    "Catalog",
    "CatalogBatchSource",
    "ImportEvent",
    "ManifestAsset",
    "MixtureBatchSource",
    "MixtureComponent",
    "SelfPlayStatistics",
    "ShardStore",
    "SnapshotStatistics",
    "SourceManifest",
    "SyntheticBatchSource",
    "TrainBatch",
    "TrainingMixtureManifest",
    "TrajectoryRecord",
]
