"""Manual state-machine orchestration for Docker training workflows."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from zero_ttt.config import ExperimentConfig, load_config
from zero_ttt.console.artifacts import ArtifactCoordinator, PublishedArtifacts
from zero_ttt.console.config import ConsoleConfig
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
)
from zero_ttt.data.catalog import Catalog
from zero_ttt.data.shards import ShardStore
from zero_ttt.learner import Learner
from zero_ttt.training.checkpoint import CheckpointManager


class TrainingConsole:
    def __init__(
        self,
        settings: ConsoleConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
        output: Callable[[str], None] = print,
        input_fn: Callable[[str], str] = input,
    ) -> None:
        self.settings = settings
        self.config: ExperimentConfig = load_config(settings.experiment_config)
        configured_run_dir = self.config.run_dir
        self.run_dir = (
            configured_run_dir.resolve()
            if configured_run_dir.is_absolute()
            else (Path.cwd() / configured_run_dir).resolve()
        )
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
        self.input = input_fn

    def _save_state(self, state: ConsoleState) -> None:
        self.state_store.save(state)
        self.state = state

    def _begin(self, operation: Operation) -> None:
        state = transition(self.state, operation)
        self._save_state(dataclasses.replace(state, last_operation=operation.value))

    def _finish(self, outcome: str) -> None:
        stopping = transition(self.state, Operation.SOFT_STOPPING)
        self._save_state(stopping)
        ready = transition(stopping, Operation.READY)
        self._save_state(dataclasses.replace(ready, last_outcome=outcome))

    def _fail(self, error: BaseException) -> None:
        try:
            failed = transition(self.state, Operation.FAILED)
        except ValueError:
            failed = dataclasses.replace(self.state, operation=Operation.FAILED)
        self._save_state(
            dataclasses.replace(failed, last_outcome=f"{type(error).__name__}: {error}")
        )

    def _catalog(self) -> Catalog:
        return Catalog(self.settings.catalog_path, ShardStore(self.settings.store_root))

    def reconcile(self) -> None:
        try:
            if not self.settings.catalog_path.is_file():
                raise FileNotFoundError(
                    f"catalog does not exist: {self.settings.catalog_path}"
                )
            if not self.settings.store_root.is_dir():
                raise FileNotFoundError(
                    f"shard store does not exist: {self.settings.store_root}"
                )
            with self._catalog() as catalog:
                cold = catalog.snapshot_statistics(
                    self.settings.cold_start_snapshot_id
                )
                if cold.games <= 0 or cold.positions <= 0:
                    raise ValueError(
                        "configured cold-start snapshot contains no trainable data"
                    )

            inspection = self.artifacts.inspect()
            inferred_phase = TrainingPhase.COLD_START
            if inspection.checkpoint is not None:
                self.artifacts.validate_checkpoint(inspection.checkpoint)
                identity = inspection.checkpoint.summary.data_identity
                if identity is None:
                    raise ValueError(
                        "console checkpoint must have a training data identity"
                    )
                if identity.mixture_manifest_sha256:
                    if (
                        self.settings.cold_start_snapshot_id
                        not in identity.component_snapshot_ids
                    ):
                        raise ValueError(
                            "mixture checkpoint does not contain the configured "
                            "cold snapshot"
                        )
                    with self._catalog() as catalog:
                        selfplay_components = [
                            snapshot_id
                            for snapshot_id in identity.component_snapshot_ids
                            if catalog.snapshot_statistics(snapshot_id).source_kind
                            == "selfplay"
                        ]
                    if len(selfplay_components) != 1:
                        raise ValueError(
                            "mixture checkpoint must contain one self-play snapshot"
                        )
                    inferred_phase = TrainingPhase.MIXTURE
                elif identity.snapshot_id != self.settings.cold_start_snapshot_id:
                    raise ValueError("cold-start checkpoint uses a different snapshot")

            self.artifacts.reconcile(inspection)
            interrupted = self.state.operation not in {
                Operation.READY,
                Operation.FAILED,
            }
            recovered_failure = self.state.operation is Operation.FAILED
            outcome = self.state.last_outcome
            if interrupted:
                outcome = (
                    f"recovered interrupted {self.state.operation.value}; "
                    f"{outcome}"
                )
                outcome = outcome.strip("; ")
            elif recovered_failure:
                outcome = f"validated after previous failure; {outcome}".strip("; ")
            reconciled = dataclasses.replace(
                self.state,
                phase=inferred_phase,
                operation=Operation.READY,
                last_outcome=outcome,
            )
            self._save_state(reconciled)
        except BaseException as error:
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
            self.output(
                f"上一轮: {status.last_operation or '-'} / {status.last_outcome or '-'}"
            )

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

            from zero_ttt.game.features import FEATURE_SCHEMA_ID
            from zero_ttt.game.rules import RULES_ID
            from zero_ttt.inference import (
                BatchedInferenceBroker,
                PublicationPositionEvaluator,
            )
            from zero_ttt.selfplay.collector import (
                SelfPlayCollector,
                search_config_sha256,
            )

            evaluator = PublicationPositionEvaluator(
                publication,
                device=self.config.runtime.device,
                inference_batch_size=self.config.selfplay.inference_batch_size,
                compile_model=self.config.selfplay.compile_inference,
                compile_mode=self.config.execution.compile_mode,
            )
            search_hash = search_config_sha256(self.config)
            evaluator_id = hashlib.sha256(
                json.dumps(
                    [evaluator.model_version, FEATURE_SCHEMA_ID, RULES_ID, search_hash],
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            with (
                SoftStopSignals() as signals,
                BatchedInferenceBroker(
                    evaluator,
                    batch_size=self.config.selfplay.inference_batch_size,
                    batch_wait_ms=self.config.selfplay.batch_wait_ms,
                    cache_size=self.config.selfplay.inference_cache_size,
                ) as broker,
            ):
                while not budget.expired and not signals.requested:
                    round_number = self.state.next_collection_round
                    summary = SelfPlayCollector(
                        self.config,
                        broker,
                        publication_sha256=evaluator.publication_sha256,
                        evaluator_id=evaluator_id,
                        store_root=self.settings.store_root,
                        catalog_path=self.settings.catalog_path,
                        games=self.config.selfplay.actor_count,
                        seed=self._round_seed(self.config.seed, round_number),
                    ).collect()
                    if (
                        summary.collected_games + summary.skipped_games
                        != summary.requested_games
                    ):
                        raise RuntimeError(
                            "self-play round did not finish all requested games"
                        )
                    rounds += 1
                    games += summary.collected_games
                    self._save_state(
                        dataclasses.replace(
                            self.state,
                            next_collection_round=round_number + 1,
                            last_outcome=(
                                f"round {round_number} sealed: "
                                f"new={summary.collected_games}, "
                                f"skipped={summary.skipped_games}"
                            ),
                        )
                    )
            self._finish(f"soft-stopped after {rounds} rounds and {games} new games")
        except BaseException as error:
            self._fail(error)
            raise

    def train(self, *, warm_start: bool = False) -> None:
        if warm_start and self.state.phase is not TrainingPhase.COLD_START:
            raise RuntimeError("warm-start is only available in COLD_START")
        if warm_start:
            if self.manager.latest_checkpoint() is None:
                raise RuntimeError("warm-start requires a full cold-start checkpoint")
            with self._catalog() as catalog:
                if catalog.selfplay_statistics().games <= 0:
                    raise RuntimeError(
                        "warm-start requires at least one sealed self-play game"
                    )
        operation = Operation.WARM_STARTING if warm_start else Operation.TRAINING
        self._begin(operation)
        steps = 0
        plan: TrainingDataPlan | None = None
        last_published: PublishedArtifacts | None = None
        try:
            budget = RuntimeBudget(self.settings.max_runtime_seconds, self.clock)
            use_mixture = warm_start or self.state.phase is TrainingPhase.MIXTURE
            plan = self.data_planner.build(use_mixture=use_mixture)

            rng = np.random.default_rng(self.config.seed)
            learner = Learner(
                self.config,
                self.manager,
                data_identity=plan.identity,
            )
            inspection = self.artifacts.inspect()
            checkpoint = inspection.checkpoint
            previous_identity = None
            if checkpoint is not None:
                self.artifacts.validate_checkpoint(checkpoint)
                stored_identity = checkpoint.summary.data_identity
                if stored_identity == plan.identity:
                    learner.restore(checkpoint.path, rng)
                else:
                    if not use_mixture:
                        raise ValueError(
                            "cold-start checkpoint data identity changed unexpectedly"
                        )
                    previous_identity = learner.restore_for_data_transition(
                        checkpoint.path, rng
                    )
            elif use_mixture:
                raise RuntimeError(
                    "warm-start/mixture training requires a full checkpoint"
                )

            with SoftStopSignals() as signals:
                while not budget.expired and not signals.requested:
                    learner.train_optimizer_step(plan.source, rng)
                    steps += 1
                    if learner.publication_due:
                        last_published = self.artifacts.publish_learner(learner, rng)

            if steps > 0:
                if (
                    last_published is not None
                    and learner.state.last_published_step
                    == learner.state.optimizer_step
                    and learner.state.last_published_samples
                    == learner.state.samples_seen
                ):
                    published = last_published
                else:
                    published = self.artifacts.publish_learner(learner, rng)
                state = dataclasses.replace(
                    self.state,
                    phase=plan.target_phase,
                )
                if (
                    previous_identity is not None
                    and plan.mixture_manifest is not None
                ):
                    reason = "warm_start" if warm_start else "mixture_rollover"
                    state = dataclasses.replace(
                        state,
                        migrations=state.migrations
                        + (
                            migration_record(
                                reason,
                                previous_identity.snapshot_id,
                                plan.selfplay_snapshot_id,
                                plan.mixture_manifest.content_sha256,
                            ),
                        ),
                    )
                self._save_state(state)
                outcome = (
                    f"soft-stopped after {steps} optimizer steps; "
                    f"checkpoint={published.checkpoint_path.name}; "
                    f"publication={published.publication_path.parent.name}"
                )
            else:
                outcome = "runtime budget expired before an optimizer step started"
            self._finish(outcome)
        except BaseException as error:
            self._fail(error)
            raise
        finally:
            if plan is not None:
                plan.close()

    def run_interactive(self) -> int:
        with self.lock:
            self.reconcile()
            while True:
                self.print_status()
                self.output(
                    "\n1) 刷新状态  2) 收集数据  3) 开始训练  4) warm-start  5) 退出"
                )
                try:
                    choice = self.input("请选择: ").strip()
                except (EOFError, KeyboardInterrupt):
                    self.output("\n控制台已退出。")
                    return 0
                try:
                    if choice == "1":
                        self.reconcile()
                    elif choice == "2":
                        self.collect()
                    elif choice == "3":
                        self.train()
                    elif choice == "4":
                        self.train(warm_start=True)
                    elif choice == "5":
                        return 0
                    else:
                        self.output("无效选择。")
                except BaseException as error:
                    self.output(f"控制台操作失败: {type(error).__name__}: {error}")
                    return 1
