"""Manual state-machine orchestration for Docker training workflows."""

from __future__ import annotations

import dataclasses
import hashlib
import time
from collections.abc import Callable
from dataclasses import asdict
from typing import TYPE_CHECKING

from zero_ttt.config import ExperimentConfig, load_config
from zero_ttt.console.config import ConsoleConfig
from zero_ttt.console.events import ConsoleEventSink, NullEventSink
from zero_ttt.console.planning import TrainingDataPlan, TrainingDataPlanner
from zero_ttt.console.runtime import RuntimeBudget, SoftStopSignals
from zero_ttt.console.state import (
    ConsoleLock,
    ConsoleState,
    Operation,
    StateStore,
    TrainingPhase,
    migration_record,
    transition,
)
from zero_ttt.console.status import (
    ConsoleStatus,
    inspect_status,
    status_payload,
)
from zero_ttt.console.telemetry import (
    configuration_summary,
    publication_identity,
    reconciled_inspection,
    training_finished_payload,
    training_step_payload,
)
from zero_ttt.data.catalog import Catalog
from zero_ttt.data.shards import ShardStore
from zero_ttt.training.artifacts import ArtifactCoordinator, PublishedArtifacts
from zero_ttt.training.checkpoint import CheckpointManager
from zero_ttt.training.contracts import LearnerDataIdentity
from zero_ttt.training.session import TrainingSession

if TYPE_CHECKING:
    from zero_ttt.selfplay.collector import CollectionSummary
    from zero_ttt.selfplay.service import SelfPlayService


class TrainingConsole:
    def __init__(
        self,
        settings: ConsoleConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
        output: Callable[[str], None] = print,
        events: ConsoleEventSink | None = None,
    ) -> None:
        self.settings = settings
        self.config: ExperimentConfig = load_config(settings.experiment_config)
        self.run_dir = settings.run_dir.resolve()
        self.manager = CheckpointManager(
            self.run_dir,
            keep=self.config.training.checkpoint_keep,
        )
        self.console_dir = self.run_dir / "console"
        self.state_store = StateStore(self.console_dir / "state.json")
        self.lock = ConsoleLock(self.console_dir / "console.lock")
        self.artifacts = ArtifactCoordinator(
            self.config,
            self.manager,
            run_dir=self.run_dir,
            catalog_path=self.settings.catalog_path,
            store_root=self.settings.store_root,
        )
        self.data_planner = TrainingDataPlanner(
            self.settings,
            self.config,
            self.console_dir,
        )
        self.state = self.state_store.load()
        self.clock = clock
        self.output = output
        self.events = events or NullEventSink()

    def _save_state(self, state: ConsoleState) -> None:
        self.state_store.save(state)
        self.state = state

    def _begin(self, operation: Operation) -> None:
        state = transition(self.state, operation)
        self._save_state(dataclasses.replace(state, last_operation=operation.value))
        self.events.emit(
            "operation_started",
            {"operation": operation.value, "phase": self.state.phase.value},
        )

    def _finish(self, outcome: str) -> None:
        operation = self.state.operation.value
        stopping = transition(self.state, Operation.SOFT_STOPPING)
        self._save_state(stopping)
        ready = transition(stopping, Operation.READY)
        self._save_state(dataclasses.replace(ready, last_outcome=outcome))
        self.events.emit(
            "operation_finished",
            {"operation": operation, "phase": self.state.phase.value, "outcome": outcome},
        )

    def _fail(self, error: Exception) -> None:
        try:
            failed = transition(self.state, Operation.FAILED)
        except ValueError:
            failed = dataclasses.replace(self.state, operation=Operation.FAILED)
        self._save_state(
            dataclasses.replace(failed, last_outcome=f"{type(error).__name__}: {error}")
        )
        self.events.emit(
            "operation_failed",
            {
                "operation": self.state.last_operation,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )

    def _catalog(self) -> Catalog:
        return Catalog(self.settings.catalog_path, ShardStore(self.settings.store_root))

    def _validate_reconcile_inputs(self) -> None:
        if not self.settings.catalog_path.is_file():
            raise FileNotFoundError(f"catalog does not exist: {self.settings.catalog_path}")
        if not self.settings.store_root.is_dir():
            raise FileNotFoundError(f"shard store does not exist: {self.settings.store_root}")
        with self._catalog() as catalog:
            cold = catalog.snapshot_statistics(self.settings.cold_start_snapshot_id)
        if cold.games <= 0 or cold.positions <= 0:
            raise ValueError("configured cold-start snapshot contains no trainable data")

    def _checkpoint_phase(self, checkpoint) -> TrainingPhase:
        if checkpoint is None:
            return TrainingPhase.COLD_START
        self.artifacts.validate_checkpoint(checkpoint)
        if checkpoint.summary.identity.run_id != self.settings.run_id:
            raise ValueError("checkpoint belongs to a different web training run")
        identity = checkpoint.summary.data_identity
        if identity is None:
            raise ValueError("console checkpoint must have a training data identity")
        if not identity.mixture_manifest_sha256:
            if identity.snapshot_id != self.settings.cold_start_snapshot_id:
                raise ValueError("cold-start checkpoint uses a different snapshot")
            return TrainingPhase.COLD_START
        if self.settings.cold_start_snapshot_id not in identity.component_snapshot_ids:
            raise ValueError("mixture checkpoint does not contain the configured cold snapshot")
        with self._catalog() as catalog:
            selfplay_components = [
                snapshot_id
                for snapshot_id in identity.component_snapshot_ids
                if catalog.snapshot_statistics(snapshot_id).source_kind == "selfplay"
            ]
        if len(selfplay_components) != 1:
            raise ValueError("mixture checkpoint must contain one self-play snapshot")
        return TrainingPhase.MIXTURE

    def _recovery_outcome(self) -> str:
        outcome = self.state.last_outcome
        if self.state.operation not in {Operation.READY, Operation.FAILED}:
            return (f"recovered interrupted {self.state.operation.value}; {outcome}").strip("; ")
        if self.state.operation is Operation.FAILED:
            return f"validated after previous failure; {outcome}".strip("; ")
        return outcome

    def reconcile(self) -> ConsoleStatus:
        try:
            self._validate_reconcile_inputs()
            inspection = self.artifacts.inspect()
            identities = [
                summary.identity
                for summary in (
                    None if inspection.checkpoint is None else inspection.checkpoint.summary,
                    inspection.publication,
                )
                if summary is not None
            ]
            if any(identity.run_id != self.settings.run_id for identity in identities):
                raise ValueError("model artifact belongs to a different web training run")
            inferred_phase = self._checkpoint_phase(inspection.checkpoint)
            publication_path = self.artifacts.reconcile(inspection)
            reconciled = dataclasses.replace(
                self.state,
                phase=inferred_phase,
                operation=Operation.READY,
                last_outcome=self._recovery_outcome(),
            )
            self._save_state(reconciled)
            current = reconciled_inspection(inspection, publication_path)
            status = inspect_status(self.settings, self.state, current)
            payload = status_payload(status)
            payload["validated"] = True
            payload["configuration"] = configuration_summary(
                self.settings, self.config, self.run_dir
            )
            self.events.emit("status", payload)
            return status
        except Exception as error:
            self._fail(error)
            raise

    def status(self) -> ConsoleStatus:
        return inspect_status(self.settings, self.state, self.artifacts.inspect())

    def print_status(self) -> None:
        status = self.status()
        self.output("\n=== Zero-TTT 训练控制台 ===")
        self.output(f"阶段/状态: {status.phase} / {status.operation}")
        self.output(
            f"模型: run={status.run_id or '-'} step={status.optimizer_step} "
            f"samples={status.samples_seen}"
        )
        self.output(f"产物: {status.artifact_consistency}")
        self.output(f"checkpoint: {status.checkpoint_path or '-'}")
        self.output(
            f"publication: {status.publication_path or '-'} "
            "(step="
            f"{status.publication_step if status.publication_step is not None else '-'}"
            ")"
        )
        self.output(
            "自博弈: "
            f"sealed rounds={status.selfplay.sealed_tasks}, "
            f"games={status.selfplay.games}, positions={status.selfplay.positions}, "
            f"collecting={status.selfplay.collecting_tasks}, "
            f"failed={status.selfplay.failed_tasks}"
        )
        self.output(
            f"未纳入最新训练 snapshot: games={status.pending_games}, "
            f"positions={status.pending_positions}"
        )
        self.output(f"cold snapshot: {status.cold_snapshot_id}")
        self.output(f"self-play snapshot: {status.selfplay_snapshot_id or '-'}")
        self.output(f"mixture: {status.mixture_manifest_sha256 or '-'}")
        if status.last_operation or status.last_outcome:
            self.output(f"上一轮: {status.last_operation or '-'} / {status.last_outcome or '-'}")

    @staticmethod
    def _round_seed(base_seed: int, round_number: int) -> int:
        digest = hashlib.sha256(
            f"zero-ttt-console-round-v1:{base_seed}:{round_number}".encode("ascii")
        ).digest()
        return int.from_bytes(digest[:8], "big")

    def collect(self) -> None:
        self._begin(Operation.COLLECTING)
        rounds = 0
        games = 0
        try:
            budget = RuntimeBudget(self.settings.max_runtime_seconds, self.clock)
            publication = self.artifacts.reconcile()
            if publication is None:
                raise RuntimeError("data collection requires a published model")
            current_publication = publication_identity(self.manager)
            self.events.emit(
                "collection_started",
                {
                    **current_publication,
                    "round_number": self.state.next_collection_round,
                },
            )

            from zero_ttt.selfplay.service import SelfPlayService

            service = SelfPlayService(
                self.config,
                publication,
                store_root=self.settings.store_root,
                catalog_path=self.settings.catalog_path,
            )
            with (
                SoftStopSignals() as signals,
                service,
            ):
                while not budget.expired and not signals.requested:
                    summary = self._collect_round(service, current_publication)
                    rounds += 1
                    games += summary.collected_games
            outcome = f"soft-stopped after {rounds} rounds and {games} new games"
            self.events.emit(
                "collection_finished",
                {
                    **current_publication,
                    "rounds": rounds,
                    "new_games": games,
                    "next_collection_round": self.state.next_collection_round,
                },
            )
            self._finish(outcome)
        except Exception as error:
            self._fail(error)
            raise

    def _collect_round(
        self,
        service: SelfPlayService,
        current_publication: dict[str, object],
    ) -> CollectionSummary:
        round_number = self.state.next_collection_round
        summary = service.collect(
            games=self.config.selfplay.actor_count,
            seed=self._round_seed(self.config.seed, round_number),
        )
        if summary.collected_games + summary.skipped_games != summary.requested_games:
            raise RuntimeError("self-play round did not finish all requested games")
        self.events.emit(
            "collection_round",
            {
                **asdict(summary),
                "run_id": current_publication["run_id"],
                "round_number": round_number,
            },
        )
        self._save_state(
            dataclasses.replace(
                self.state,
                next_collection_round=round_number + 1,
                last_outcome=(
                    f"round {round_number} sealed: new={summary.collected_games}, "
                    f"skipped={summary.skipped_games}"
                ),
            )
        )
        return summary

    def _validate_warm_start(self, warm_start: bool) -> None:
        if warm_start and self.state.phase is not TrainingPhase.COLD_START:
            raise RuntimeError("warm-start is only available in COLD_START")
        if not warm_start:
            return
        if self.manager.latest_checkpoint() is None:
            raise RuntimeError("warm-start requires a full cold-start checkpoint")
        with self._catalog() as catalog:
            games = catalog.selfplay_statistics().games
        if games <= 0:
            raise RuntimeError("warm-start requires at least one sealed self-play game")

    def _training_session(
        self, plan: TrainingDataPlan, use_mixture: bool
    ) -> tuple[TrainingSession, LearnerDataIdentity | None]:
        session = TrainingSession(
            self.config,
            self.manager,
            data_identity=plan.identity,
            run_id=self.settings.run_id,
            artifacts=self.artifacts,
        )
        checkpoint = self.artifacts.inspect().checkpoint
        if checkpoint is None:
            if use_mixture:
                raise RuntimeError("warm-start/mixture training requires a full checkpoint")
            return session, None
        self.artifacts.validate_checkpoint(checkpoint)
        if checkpoint.summary.data_identity == plan.identity:
            session.restore(checkpoint.path)
            return session, None
        if not use_mixture:
            raise ValueError("cold-start checkpoint data identity changed unexpectedly")
        previous = session.restore(checkpoint.path, allow_data_transition=True)
        return session, previous

    def _train_until_stop(
        self,
        session: TrainingSession,
        plan: TrainingDataPlan,
        budget: RuntimeBudget,
    ) -> tuple[int, PublishedArtifacts | None]:
        steps = 0
        last_published = None
        with SoftStopSignals() as signals:
            while not budget.expired and not signals.requested:
                started = time.perf_counter()
                metrics = session.step(plan.source)
                elapsed = time.perf_counter() - started
                steps += 1
                self.events.emit(
                    "training_step",
                    training_step_payload(self.config, session, metrics, elapsed),
                )
                if session.publication_due:
                    last_published = session.publish()
        return steps, last_published

    @staticmethod
    def _current_publication(
        session: TrainingSession, last_published: PublishedArtifacts | None
    ) -> PublishedArtifacts:
        state = session.learner.state
        if (
            last_published is not None
            and state.last_published_step == state.optimizer_step
            and state.last_published_samples == state.samples_seen
        ):
            return last_published
        return session.publish()

    def _record_training_transition(
        self,
        plan: TrainingDataPlan,
        previous: LearnerDataIdentity | None,
        warm_start: bool,
    ) -> None:
        state = dataclasses.replace(self.state, phase=plan.target_phase)
        if previous is not None and plan.mixture_manifest is not None:
            reason = "warm_start" if warm_start else "mixture_rollover"
            record = migration_record(
                reason,
                previous.snapshot_id,
                plan.selfplay_snapshot_id,
                plan.mixture_manifest.content_sha256,
            )
            state = dataclasses.replace(state, migrations=(*state.migrations, record))
        self._save_state(state)

    def train(self, *, warm_start: bool = False) -> None:
        self._validate_warm_start(warm_start)
        operation = Operation.WARM_STARTING if warm_start else Operation.TRAINING
        self._begin(operation)
        plan: TrainingDataPlan | None = None
        try:
            budget = RuntimeBudget(self.settings.max_runtime_seconds, self.clock)
            use_mixture = warm_start or self.state.phase is TrainingPhase.MIXTURE
            plan = self.data_planner.build(use_mixture=use_mixture)
            session, previous_identity = self._training_session(plan, use_mixture)
            learner_state = session.learner.state
            self.events.emit(
                "training_started",
                {
                    "run_id": learner_state.run_id,
                    "optimizer_step": learner_state.optimizer_step,
                    "samples_seen": learner_state.samples_seen,
                    "phase": plan.target_phase.value,
                    "data_identity": asdict(plan.identity),
                    "config_json": self.config.canonical_json(),
                    "config_sha256": self.config.sha256,
                },
            )
            steps, last_published = self._train_until_stop(session, plan, budget)
            published: PublishedArtifacts | None = None
            if steps > 0:
                published = self._current_publication(session, last_published)
                self._record_training_transition(plan, previous_identity, warm_start)
                outcome = (
                    f"soft-stopped after {steps} optimizer steps; "
                    f"checkpoint={published.checkpoint_path.name}; "
                    f"publication={published.publication_path.parent.name}"
                )
            else:
                outcome = "runtime budget expired before an optimizer step started"
            self.events.emit(
                "training_finished",
                training_finished_payload(
                    session,
                    plan,
                    steps,
                    self.state.phase.value,
                    outcome,
                    published,
                ),
            )
            self._finish(outcome)
        except Exception as error:
            self._fail(error)
            raise
        finally:
            if plan is not None:
                plan.close()
