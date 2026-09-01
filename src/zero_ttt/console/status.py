"""Read-only status projection for the interactive console."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from zero_ttt.console.artifacts import ArtifactConsistency, ArtifactInspection
from zero_ttt.console.config import ConsoleConfig
from zero_ttt.console.state import ConsoleState
from zero_ttt.data.catalog import Catalog, SelfPlayStatistics
from zero_ttt.data.shards import ShardStore


@dataclass(frozen=True, slots=True)
class ConsoleStatus:
    phase: str
    operation: str
    run_id: str
    optimizer_step: int
    samples_seen: int
    checkpoint_path: Path | None
    publication_path: Path | None
    publication_step: int | None
    artifact_consistency: str
    selfplay: SelfPlayStatistics
    pending_games: int
    pending_positions: int
    cold_snapshot_id: str
    selfplay_snapshot_id: str
    mixture_manifest_sha256: str
    last_operation: str
    last_outcome: str


def status_payload(status: ConsoleStatus) -> dict[str, object]:
    payload = asdict(status)
    for field in ("checkpoint_path", "publication_path"):
        value = payload[field]
        payload[field] = None if value is None else str(value)
    return payload


_CONSISTENCY_MESSAGES = {
    ArtifactConsistency.NO_ARTIFACTS: "no model artifacts",
    ArtifactConsistency.PUBLICATION_ONLY: "publication exists without resumable checkpoint",
    ArtifactConsistency.CHECKPOINT_ONLY: "checkpoint is newer; publication is missing",
    ArtifactConsistency.ALIGNED: "checkpoint and publication aligned",
    ArtifactConsistency.CHECKPOINT_NEWER: "checkpoint is newer than publication",
    ArtifactConsistency.PUBLICATION_AHEAD: "publication is ahead of resumable checkpoint",
    ArtifactConsistency.IDENTITY_MISMATCH: ("checkpoint and publication belong to different runs"),
    ArtifactConsistency.CONFLICTING_STEP: ("checkpoint and publication conflict at the same step"),
}


@dataclass(frozen=True, slots=True)
class _DataStatus:
    selfplay: SelfPlayStatistics
    pending_games: int
    pending_positions: int
    selfplay_snapshot: str
    mixture_sha: str


def _data_status(settings: ConsoleConfig, identity) -> _DataStatus:
    selfplay_snapshot = ""
    mixture_sha = "" if identity is None else identity.mixture_manifest_sha256
    with Catalog(settings.catalog_path, ShardStore(settings.store_root)) as catalog:
        selfplay = catalog.selfplay_statistics()
        candidates = (
            () if identity is None else (identity.component_snapshot_ids or (identity.snapshot_id,))
        )
        for snapshot_id in candidates:
            if snapshot_id.startswith("mixture:"):
                continue
            if catalog.snapshot_statistics(snapshot_id).source_kind != "selfplay":
                continue
            if selfplay_snapshot:
                raise ValueError("checkpoint contains multiple self-play snapshots")
            selfplay_snapshot = snapshot_id
        pending = catalog.selfplay_outside_snapshot(selfplay_snapshot or None)
    return _DataStatus(selfplay, *pending, selfplay_snapshot, mixture_sha)


def _consistency_message(artifacts: ArtifactInspection) -> str:
    if artifacts.consistency is ArtifactConsistency.INVALID_PUBLICATION:
        return f"invalid publication: {artifacts.publication_error}"
    return _CONSISTENCY_MESSAGES[artifacts.consistency]


def inspect_status(
    settings: ConsoleConfig,
    state: ConsoleState,
    artifacts: ArtifactInspection,
) -> ConsoleStatus:
    checkpoint = artifacts.checkpoint
    identity = None if checkpoint is None else checkpoint.summary.data_identity
    learner_state = {} if checkpoint is None else checkpoint.summary.learner_state
    publication = artifacts.publication_path
    publication_step = (
        None if artifacts.publication is None else artifacts.publication.identity.optimizer_step
    )
    checkpoint_step = int(learner_state.get("optimizer_step", 0))
    data = _data_status(settings, identity)
    return ConsoleStatus(
        phase=state.phase.value,
        operation=state.operation.value,
        run_id=str(learner_state.get("run_id", "")),
        optimizer_step=checkpoint_step,
        samples_seen=int(learner_state.get("samples_seen", 0)),
        checkpoint_path=None if checkpoint is None else checkpoint.path,
        publication_path=publication,
        publication_step=publication_step,
        artifact_consistency=_consistency_message(artifacts),
        selfplay=data.selfplay,
        pending_games=data.pending_games,
        pending_positions=data.pending_positions,
        cold_snapshot_id=settings.cold_start_snapshot_id,
        selfplay_snapshot_id=data.selfplay_snapshot,
        mixture_manifest_sha256=data.mixture_sha,
        last_operation=state.last_operation,
        last_outcome=state.last_outcome,
    )
