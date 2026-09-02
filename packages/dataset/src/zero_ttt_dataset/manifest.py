"""Database-independent snapshot manifests."""

from __future__ import annotations

from dataclasses import asdict
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from zero_ttt_contracts.hashing import payload_sha256

from zero_ttt_dataset.locators import AnnotationLocator, TrajectoryLocator

SNAPSHOT_FORMAT_VERSION = 1


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TrajectoryEntry(_StrictModel):
    game_id: str
    content_sha256: str
    shard_sha256: str
    relative_path: str
    game_index: int = Field(ge=0)
    trainable_start_ply: int = Field(ge=0)
    trainable_positions: int = Field(gt=0)

    def locator(self) -> TrajectoryLocator:
        return TrajectoryLocator(**self.model_dump())


class AnnotationEntry(_StrictModel):
    game_id: str
    content_sha256: str
    ply: int = Field(ge=0)
    teacher_fingerprint: str
    shard_sha256: str
    relative_path: str
    record_index: int = Field(ge=0)

    def locator(self) -> AnnotationLocator:
        return AnnotationLocator(**self.model_dump())


class SnapshotManifest(_StrictModel):
    format_version: int = SNAPSHOT_FORMAT_VERSION
    snapshot_id: str
    seed: int
    split: Literal["train", "validation"]
    validation_fraction: float = Field(ge=0.0, lt=1.0)
    source_kind: Literal["external", "selfplay"]
    task_id: str = ""
    games: int = Field(gt=0)
    positions: int = Field(gt=0)
    trajectories: tuple[TrajectoryEntry, ...]
    annotations: tuple[AnnotationEntry, ...] = ()

    @model_validator(mode="after")
    def _validate_counts_and_identity(self) -> SnapshotManifest:
        if self.format_version != SNAPSHOT_FORMAT_VERSION:
            raise ValueError(f"expected snapshot format v{SNAPSHOT_FORMAT_VERSION}")
        if len(self.trajectories) != self.games:
            raise ValueError("snapshot game count does not match trajectory entries")
        if sum(item.trainable_positions for item in self.trajectories) != self.positions:
            raise ValueError("snapshot position count does not match trajectory entries")
        return self

    @property
    def content_sha256(self) -> str:
        return payload_sha256(self.model_dump(mode="json"))

    @classmethod
    def from_locators(
        cls,
        *,
        snapshot_id: str,
        seed: int,
        split: Literal["train", "validation"],
        validation_fraction: float,
        source_kind: Literal["external", "selfplay"],
        task_id: str,
        trajectories: tuple[TrajectoryLocator, ...],
        annotations: tuple[AnnotationLocator, ...] = (),
    ) -> SnapshotManifest:
        return cls(
            snapshot_id=snapshot_id,
            seed=seed,
            split=split,
            validation_fraction=validation_fraction,
            source_kind=source_kind,
            task_id=task_id,
            games=len(trajectories),
            positions=sum(item.trainable_positions for item in trajectories),
            trajectories=tuple(TrajectoryEntry(**asdict(item)) for item in trajectories),
            annotations=tuple(AnnotationEntry(**asdict(item)) for item in annotations),
        )
