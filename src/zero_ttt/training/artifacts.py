"""Typed inspection and recovery for training model artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from zero_ttt._io import sha256_file
from zero_ttt.config import ExperimentConfig
from zero_ttt.data.catalog import Catalog
from zero_ttt.data.shards import ShardStore
from zero_ttt.learner import Learner
from zero_ttt.training.checkpoint import CheckpointManager, checkpoint_metadata
from zero_ttt.training.contracts import CheckpointSummary, PublicationSummary


class ArtifactConsistency(StrEnum):
    NO_ARTIFACTS = "no_artifacts"
    PUBLICATION_ONLY = "publication_only"
    CHECKPOINT_ONLY = "checkpoint_only"
    ALIGNED = "aligned"
    CHECKPOINT_NEWER = "checkpoint_newer"
    PUBLICATION_AHEAD = "publication_ahead"
    IDENTITY_MISMATCH = "identity_mismatch"
    CONFLICTING_STEP = "conflicting_step"
    INVALID_PUBLICATION = "invalid_publication"


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    path: Path
    payload: dict[str, Any]
    summary: CheckpointSummary


@dataclass(frozen=True, slots=True)
class ArtifactInspection:
    checkpoint: LoadedCheckpoint | None
    publication_path: Path | None
    publication: PublicationSummary | None
    publication_error: str
    consistency: ArtifactConsistency


@dataclass(frozen=True, slots=True)
class PublishedArtifacts:
    checkpoint_path: Path
    publication_path: Path


class ArtifactCoordinator:
    def __init__(
        self,
        config: ExperimentConfig,
        manager: CheckpointManager,
        *,
        run_dir: str | Path,
        catalog_path: str | Path,
        store_root: str | Path,
    ) -> None:
        self.config = config
        self.manager = manager
        self.run_dir = Path(run_dir)
        self.catalog_path = Path(catalog_path)
        self.store_root = Path(store_root)

    def _load_checkpoint(self) -> LoadedCheckpoint | None:
        path = self.manager.latest_checkpoint()
        if path is None:
            return None
        payload = self.manager.load(path)
        return LoadedCheckpoint(path, payload, CheckpointSummary.from_payload(payload))

    def inspect(self) -> ArtifactInspection:
        checkpoint = self._load_checkpoint()
        publication_path = None
        publication = None
        publication_error = ""
        try:
            publication_path = self.manager.current_publication()
            if publication_path is not None:
                publication = PublicationSummary.from_payload(
                    self.manager.load_publication(publication_path)
                )
        except (
            OSError,
            EOFError,
            KeyError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            publication_error = str(error)

        consistency = self._consistency(
            checkpoint,
            publication,
            publication_error=publication_error,
        )
        return ArtifactInspection(
            checkpoint=checkpoint,
            publication_path=publication_path,
            publication=publication,
            publication_error=publication_error,
            consistency=consistency,
        )

    @staticmethod
    def _consistency(
        checkpoint: LoadedCheckpoint | None,
        publication: PublicationSummary | None,
        *,
        publication_error: str,
    ) -> ArtifactConsistency:
        if publication_error:
            return ArtifactConsistency.INVALID_PUBLICATION
        if checkpoint is None and publication is None:
            return ArtifactConsistency.NO_ARTIFACTS
        if checkpoint is None:
            return ArtifactConsistency.PUBLICATION_ONLY
        if publication is None:
            return ArtifactConsistency.CHECKPOINT_ONLY
        expected = checkpoint.summary.identity
        actual = publication.identity
        if actual == expected:
            return ArtifactConsistency.ALIGNED
        if actual.run_id != expected.run_id:
            return ArtifactConsistency.IDENTITY_MISMATCH
        if actual.optimizer_step > expected.optimizer_step:
            return ArtifactConsistency.PUBLICATION_AHEAD
        if actual.optimizer_step == expected.optimizer_step:
            return ArtifactConsistency.CONFLICTING_STEP
        return ArtifactConsistency.CHECKPOINT_NEWER

    def _validate_config(self, summary: CheckpointSummary | PublicationSummary) -> None:
        identity = summary.identity
        if (
            identity.config_sha256 != self.config.sha256
            or identity.config_json != self.config.canonical_json()
        ):
            raise ValueError("model artifact does not match the configured experiment")

    def validate_checkpoint(self, checkpoint: LoadedCheckpoint) -> None:
        self._validate_config(checkpoint.summary)

    def _record_publication(
        self,
        publication: Path,
        summary: PublicationSummary,
    ) -> None:
        relative = publication.relative_to(self.run_dir).as_posix()
        identity = summary.identity
        with Catalog(self.catalog_path, ShardStore(self.store_root)) as catalog:
            catalog.record_publication(
                identity.run_id,
                identity.optimizer_step,
                identity.samples_seen,
                relative,
                sha256_file(publication),
            )

    def _publish_checkpoint(self, checkpoint: LoadedCheckpoint) -> Path:
        identity = checkpoint.summary.identity
        publication = self.manager.save_publication(
            identity.run_id,
            identity.optimizer_step,
            identity.samples_seen,
            checkpoint.payload["slow_state"],
            checkpoint_metadata(identity.config_json, identity.config_sha256),
        )
        actual = PublicationSummary.from_payload(self.manager.load_publication(publication))
        if actual.identity != identity:
            raise ValueError("recovered publication identity does not match checkpoint")
        return publication

    def _repair_publication_state(self, checkpoint: LoadedCheckpoint) -> Path:
        identity = checkpoint.summary.identity
        state = dict(checkpoint.summary.learner_state)
        next_boundary = (
            identity.samples_seen // self.config.training.publish_interval_samples + 1
        ) * self.config.training.publish_interval_samples
        expected = {
            "last_published_step": identity.optimizer_step,
            "last_published_samples": identity.samples_seen,
            "next_publish_sample": next_boundary,
        }
        if all(state[name] == value for name, value in expected.items()):
            return checkpoint.path
        state.update(expected)
        payload = {**checkpoint.payload, "learner_state": state}
        return self.manager.save_full(identity.optimizer_step, payload)

    def reconcile(
        self,
        inspection: ArtifactInspection | None = None,
    ) -> Path | None:
        inspection = inspection or self.inspect()
        checkpoint = inspection.checkpoint
        publication = inspection.publication
        publication_path = inspection.publication_path

        if checkpoint is None:
            if inspection.publication_error:
                raise ValueError(f"current publication is invalid: {inspection.publication_error}")
            if publication is None or publication_path is None:
                return None
            self._validate_config(publication)
            self._record_publication(publication_path, publication)
            return publication_path

        self._validate_config(checkpoint.summary)
        desired = checkpoint.summary.identity
        if publication is not None and publication.identity.run_id == desired.run_id:
            actual = publication.identity
            if actual.optimizer_step > desired.optimizer_step:
                raise ValueError("publication is ahead of the latest resumable checkpoint")
            if actual.optimizer_step == desired.optimizer_step and actual != desired:
                raise ValueError("publication conflicts with the latest resumable checkpoint")

        if publication is None or publication.identity != desired:
            publication_path = self._publish_checkpoint(checkpoint)
            publication = PublicationSummary.from_payload(
                self.manager.load_publication(publication_path)
            )
        if publication_path is None:
            raise RuntimeError("publication recovery did not produce an artifact")
        self._record_publication(publication_path, publication)
        self._repair_publication_state(checkpoint)
        return publication_path

    def publish_learner(
        self,
        learner: Learner,
        rng: np.random.Generator,
    ) -> PublishedArtifacts:
        if (
            learner.config.sha256 != self.config.sha256
            or learner.config.canonical_json() != self.config.canonical_json()
        ):
            raise ValueError("learner does not match the configured experiment")
        learner.save_checkpoint(rng)
        publication_path = learner.publish()
        summary = PublicationSummary.from_payload(self.manager.load_publication(publication_path))
        self._record_publication(publication_path, summary)
        checkpoint_path = learner.save_checkpoint(rng)
        return PublishedArtifacts(checkpoint_path, publication_path)
