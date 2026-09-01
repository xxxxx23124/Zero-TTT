"""Pure event-payload projections for the console orchestration layer."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch

from zero_ttt.config import ExperimentConfig
from zero_ttt.console.config import ConsoleConfig
from zero_ttt.console.planning import TrainingDataPlan
from zero_ttt.training.artifacts import (
    ArtifactConsistency,
    ArtifactInspection,
    PublishedArtifacts,
)
from zero_ttt.training.checkpoint import CheckpointManager
from zero_ttt.training.contracts import PublicationSummary
from zero_ttt.training.session import TrainingSession


def configuration_summary(
    settings: ConsoleConfig,
    config: ExperimentConfig,
    run_dir: Path,
) -> dict[str, object]:
    return {
        "experiment_config": str(settings.experiment_config),
        "config_sha256": config.sha256,
        "run_name": config.run_name,
        "run_dir": str(run_dir),
        "max_runtime_hours": settings.max_runtime_hours,
        "cold_start_snapshot_id": settings.cold_start_snapshot_id,
        "mixture_selfplay_weight": settings.mixture.selfplay,
        "mixture_cold_start_weight": settings.mixture.cold_start,
        "model": {
            "d_model": config.model.d_model,
            "n_layers": config.model.n_layers,
            "n_heads": config.model.n_heads,
        },
        "training": {
            "batch_size": config.training.batch_size,
            "accumulation_steps": config.training.accumulation_steps,
            "learning_rate": config.training.learning_rate,
        },
        "selfplay": {
            "actor_count": config.selfplay.actor_count,
            "max_simulations": config.search.max_simulations,
        },
    }


def reconciled_inspection(
    inspection: ArtifactInspection,
    publication_path: Path | None,
) -> ArtifactInspection:
    checkpoint = inspection.checkpoint
    if checkpoint is None or publication_path is None:
        return inspection
    return ArtifactInspection(
        checkpoint=checkpoint,
        publication_path=publication_path,
        publication=PublicationSummary(checkpoint.summary.identity),
        publication_error="",
        consistency=ArtifactConsistency.ALIGNED,
    )


def publication_identity(manager: CheckpointManager) -> dict[str, object]:
    pointer = manager.publication_dir / "current.json"
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    return {
        "run_id": str(payload["run_id"]),
        "optimizer_step": int(payload["optimizer_step"]),
        "samples_seen": int(payload["samples_seen"]),
    }


def training_step_payload(
    config: ExperimentConfig,
    session: TrainingSession,
    metrics,
    elapsed: float,
) -> dict[str, object]:
    state = session.learner.state
    effective_batch = config.training.effective_batch_size
    payload = {
        **asdict(metrics),
        "run_id": state.run_id,
        "samples_seen": state.samples_seen,
        "step_seconds": elapsed,
        "positions_per_second": 0.0 if elapsed == 0.0 else effective_batch / elapsed,
        "allocated_gib": None,
        "reserved_gib": None,
        "max_allocated_gib": None,
    }
    if session.learner.device.type == "cuda" and torch.cuda.is_available():
        gib = float(1024**3)
        device = session.learner.device
        payload.update(
            allocated_gib=torch.cuda.memory_allocated(device) / gib,
            reserved_gib=torch.cuda.memory_reserved(device) / gib,
            max_allocated_gib=torch.cuda.max_memory_allocated(device) / gib,
        )
    return payload


def training_finished_payload(
    session: TrainingSession,
    plan: TrainingDataPlan,
    steps: int,
    phase: str,
    outcome: str,
    published: PublishedArtifacts | None,
) -> dict[str, object]:
    state = session.learner.state
    return {
        "run_id": state.run_id,
        "optimizer_step": state.optimizer_step,
        "samples_seen": state.samples_seen,
        "phase": phase,
        "steps": steps,
        "outcome": outcome,
        "checkpoint_path": None if published is None else str(published.checkpoint_path),
        "publication_path": None if published is None else str(published.publication_path),
        "selfplay_snapshot_id": plan.selfplay_snapshot_id,
        "mixture_manifest_sha256": (
            "" if plan.mixture_manifest is None else plan.mixture_manifest.content_sha256
        ),
    }
