"""Finite, resumable training use cases; no control-plane or data-database imports."""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from torch.utils.tensorboard import SummaryWriter
from zero_ttt.config import config_from_mapping
from zero_ttt_contracts import ArtifactKind, ArtifactRef, JobEnvelope, RunSpec
from zero_ttt_contracts.hashing import sha256_file
from zero_ttt_dataset import (
    LocalArtifactStore,
    PortableMixtureBatchSource,
    PortableSnapshotBatchSource,
    SnapshotManifest,
)
from zero_ttt_worker import JobContext, JobResult

from zero_ttt_trainer.checkpoint import CheckpointManager
from zero_ttt_trainer.contracts import LearnerDataIdentity
from zero_ttt_trainer.learner import Learner
from zero_ttt_trainer.settings import TrainerSettings


class TrainingJobHandler:
    def __init__(self, settings: TrainerSettings) -> None:
        self.settings = settings
        self.artifacts = LocalArtifactStore(settings.artifact_root)
        self.settings.work_root.mkdir(parents=True, exist_ok=True)

    def mapping(self):
        return {
            "trainer.cold-start": self.execute,
            "trainer.mixture": self.execute,
        }

    def _load_snapshot(self, reference: ArtifactRef) -> SnapshotManifest:
        path = self.artifacts.verify(reference)
        return SnapshotManifest.model_validate_json(path.read_text(encoding="utf-8"))

    def _source(self, job: JobEnvelope, run: RunSpec):
        snapshots = [item for item in job.inputs if item.kind is ArtifactKind.DATASET_SNAPSHOT]
        cold_ref = next(
            (item for item in snapshots if item.artifact_id == run.cold_snapshot.artifact_id),
            None,
        )
        if cold_ref is None:
            raise ValueError("training job is missing its frozen cold-start snapshot")
        cold = PortableSnapshotBatchSource(
            self._load_snapshot(cold_ref), self.settings.artifact_root / "data" / "shards"
        )
        if job.kind == "trainer.cold-start":
            identity = LearnerDataIdentity(
                snapshot_id=cold.manifest.snapshot_id,
                sampling_config_sha256=cold.sampling_config_sha256,
            )
            return cold, identity
        selfplay_ref = next(
            (
                item
                for item in snapshots
                if item.artifact_id != run.cold_snapshot.artifact_id
                and self._load_snapshot(item).source_kind == "selfplay"
            ),
            None,
        )
        if selfplay_ref is None:
            cold.close()
            raise ValueError("mixture training requires a self-play snapshot")
        selfplay = PortableSnapshotBatchSource(
            self._load_snapshot(selfplay_ref), self.settings.artifact_root / "data" / "shards"
        )
        training = run.profile["training"]
        mixture = training["mixture"]
        source = PortableMixtureBatchSource(
            (
                (selfplay, float(mixture["selfplay_weight"])),
                (cold, float(mixture["cold_start_weight"])),
            )
        )
        identity = LearnerDataIdentity(
            snapshot_id=f"mixture:{source.sampling_config_sha256}",
            sampling_config_sha256=source.sampling_config_sha256,
            mixture_manifest_sha256=source.sampling_config_sha256,
            component_snapshot_ids=source.component_snapshot_ids,
        )
        return source, identity

    def execute(self, job: JobEnvelope, context: JobContext) -> JobResult:
        raw_run = job.payload.get("run_spec")
        if not isinstance(raw_run, dict):
            raise ValueError("training job is missing a frozen run specification")
        run = RunSpec.model_validate(raw_run)
        config = config_from_mapping(run.profile)
        run_root = self.settings.artifact_root / "models" / "runs" / run.run_id
        manager = CheckpointManager(run_root, keep=config.training.checkpoint_keep)
        source, identity = self._source(job, run)
        rng = np.random.default_rng(config.seed)
        learner = Learner(config, manager, data_identity=identity, run_id=run.run_id)
        checkpoint_ref = next(
            (item for item in job.inputs if item.kind is ArtifactKind.CHECKPOINT), None
        )
        checkpoint_path: Path | None = None
        if checkpoint_ref is not None:
            checkpoint_path = self.artifacts.verify(checkpoint_ref)
            if job.kind == "trainer.mixture":
                learner.restore_for_data_transition(checkpoint_path, rng)
            else:
                learner.restore(checkpoint_path, rng)
        parameters = job.payload.get("workflow_input", {})
        max_steps = int(parameters.get("steps", 1))
        if max_steps <= 0:
            raise ValueError("training steps must be positive")
        max_runtime_seconds = float(parameters.get("max_runtime_seconds", 0.0))
        if max_runtime_seconds < 0:
            raise ValueError("max_runtime_seconds cannot be negative")
        deadline = None if max_runtime_seconds == 0 else time.monotonic() + max_runtime_seconds
        writer = SummaryWriter(run_root / "tensorboard")
        steps = 0
        recovered_checkpoint = manager.latest_checkpoint()
        try:
            if recovered_checkpoint is not None and (
                checkpoint_path is None
                or recovered_checkpoint.resolve() != checkpoint_path.resolve()
            ):
                learner.restore(recovered_checkpoint, rng)
                checkpoint_path = recovered_checkpoint
                context.emit(
                    "training.recovered-final-artifact",
                    {"optimizer_step": learner.state.optimizer_step},
                )
            else:
                while steps < max_steps and not context.cancel_requested:
                    if deadline is not None and time.monotonic() >= deadline:
                        break
                    started = time.perf_counter()
                    metrics = learner.train_optimizer_step(source, rng)
                    elapsed = time.perf_counter() - started
                    steps += 1
                    values = asdict(metrics)
                    values["elapsed_seconds"] = elapsed
                    values["samples_seen"] = learner.state.samples_seen
                    for name, value in values.items():
                        if isinstance(value, int | float):
                            writer.add_scalar(
                                f"training/{name}", value, learner.state.optimizer_step
                            )
                    writer.flush()
                    context.emit("training.step", values)
                checkpoint_path = learner.save_checkpoint(rng)
            publication_path = learner.publish()
        finally:
            writer.close()
            source.close()
        checkpoint = self._artifact_ref(
            ArtifactKind.CHECKPOINT,
            f"checkpoint.{run.run_id}.{learner.state.optimizer_step}",
            checkpoint_path,
        )
        publication = self._artifact_ref(
            ArtifactKind.PUBLICATION,
            f"publication.{run.run_id}.{learner.state.optimizer_step}",
            publication_path,
        )
        result: dict[str, object] = {
            "run_id": run.run_id,
            "optimizer_step": learner.state.optimizer_step,
            "samples_seen": learner.state.samples_seen,
            "steps_executed": steps,
        }
        context.emit("training.completed", result)
        return JobResult(result, (checkpoint, publication))

    def _artifact_ref(self, kind: ArtifactKind, artifact_id: str, path: str | Path) -> ArtifactRef:
        resolved = Path(path).resolve()
        relative = resolved.relative_to(self.settings.artifact_root.resolve()).as_posix()
        return ArtifactRef(
            kind=kind,
            artifact_id=artifact_id,
            format_version=1,
            sha256=sha256_file(resolved),
            uri=f"artifact://{relative}",
            size_bytes=resolved.stat().st_size,
        )
