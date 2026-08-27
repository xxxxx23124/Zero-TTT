"""Shared application service for resumable learner workflows."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from zero_ttt.config import ExperimentConfig
from zero_ttt.data.contracts import BatchSource
from zero_ttt.learner import Learner, StepMetrics
from zero_ttt.training.artifacts import ArtifactCoordinator, PublishedArtifacts
from zero_ttt.training.checkpoint import CheckpointManager
from zero_ttt.training.contracts import LearnerDataIdentity


class TrainingSession:
    """Own learner construction, RNG, restore, stepping, and artifact commits."""

    def __init__(
        self,
        config: ExperimentConfig,
        manager: CheckpointManager,
        *,
        data_identity: LearnerDataIdentity | None = None,
        run_id: str | None = None,
        artifacts: ArtifactCoordinator | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.config = config
        self.manager = manager
        self.artifacts = artifacts
        self.rng = rng or np.random.default_rng(config.seed)
        self.learner = Learner(
            config,
            manager,
            data_identity=data_identity,
            run_id=run_id,
        )

    def restore(
        self,
        checkpoint: str | Path,
        *,
        allow_data_transition: bool = False,
    ) -> LearnerDataIdentity | None:
        if allow_data_transition:
            return self.learner.restore_for_data_transition(checkpoint, self.rng)
        self.learner.restore(checkpoint, self.rng)
        return None

    def restore_requested(self, resume: str) -> Path:
        checkpoint = self.manager.latest_checkpoint() if resume == "latest" else Path(resume)
        if checkpoint is None:
            raise FileNotFoundError("no checkpoint is available to resume")
        self.restore(checkpoint)
        return checkpoint

    def step(self, source: BatchSource) -> StepMetrics:
        return self.learner.train_optimizer_step(source, self.rng)

    @property
    def publication_due(self) -> bool:
        return self.learner.publication_due

    def publish(self) -> PublishedArtifacts:
        if self.artifacts is None:
            checkpoint = self.learner.save_checkpoint(self.rng)
            publication = self.learner.publish()
            checkpoint = self.learner.save_checkpoint(self.rng)
            return PublishedArtifacts(checkpoint, publication)
        return self.artifacts.publish_learner(self.learner, self.rng)

    def publish_if_due(self) -> PublishedArtifacts | None:
        return self.publish() if self.publication_due else None

    def save_checkpoint(self) -> Path:
        return self.learner.save_checkpoint(self.rng)
