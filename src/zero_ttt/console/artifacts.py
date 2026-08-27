"""Compatibility re-exports for the former console artifact module."""

from zero_ttt.training.artifacts import (
    ArtifactConsistency,
    ArtifactCoordinator,
    ArtifactInspection,
    LoadedCheckpoint,
    PublishedArtifacts,
)

__all__ = [
    "ArtifactConsistency",
    "ArtifactCoordinator",
    "ArtifactInspection",
    "LoadedCheckpoint",
    "PublishedArtifacts",
]
