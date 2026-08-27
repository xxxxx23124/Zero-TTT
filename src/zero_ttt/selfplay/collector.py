"""Concurrent OpenSpiel self-play with resumable, whole-game persistence."""

from __future__ import annotations

import dataclasses
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from zero_ttt._io import atomic_write_bytes, canonical_json_bytes, sha256_bytes
from zero_ttt.config import ExperimentConfig
from zero_ttt.data.catalog import Catalog
from zero_ttt.data.ingestion import DEFAULT_TARGET_SHARD_BYTES, TrajectoryShardSink
from zero_ttt.data.records import TrajectoryRecord
from zero_ttt.data.shards import ShardStore
from zero_ttt.game.features import FEATURE_SCHEMA_ID
from zero_ttt.game.rules import RULES_ID, Color
from zero_ttt.inference.batching import BatchedInferenceBroker, BatchingStats
from zero_ttt.search.open_spiel import (
    OpenSpielEvaluator,
    OpenSpielGoGame,
    search_position,
)
from zero_ttt.versioning import RECORD_SCHEMA, SELFPLAY_TASK_SCHEMA


def _derived_seed(task_id: str, ordinal: int, ply: int, kind: str) -> int:
    payload = f"{task_id}:{ordinal}:{ply}:{kind}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _game_id(task_id: str, ordinal: int) -> str:
    return sha256_bytes(canonical_json_bytes(["zero-ttt-selfplay-game-v1", task_id, ordinal]))


def search_config_sha256(config: ExperimentConfig) -> str:
    payload = {
        "search": dataclasses.asdict(config.search),
        "game": dataclasses.asdict(config.game),
        "rules_id": RULES_ID,
        "feature_schema_id": FEATURE_SCHEMA_ID,
    }
    return sha256_bytes(canonical_json_bytes(payload))


@dataclass(frozen=True, slots=True)
class CollectionSummary:
    task_id: str
    requested_games: int
    collected_games: int
    skipped_games: int
    new_positions: int
    new_shards: int
    simulations: int
    wall_seconds: float
    simulations_per_second: float
    rules_seconds: float
    batching: BatchingStats


@dataclass(frozen=True, slots=True)
class _CompletedGame:
    record: TrajectoryRecord
    simulations: int
    rules_seconds: float


@dataclass(slots=True)
class _GameTrace:
    moves: list[int] = field(default_factory=list)
    offsets: list[int] = field(default_factory=lambda: [0])
    policy_actions: list[int] = field(default_factory=list)
    policy_values: list[float] = field(default_factory=list)
    budgets: list[int] = field(default_factory=list)
    root_values: list[float] = field(default_factory=list)
    root_scores: list[float] = field(default_factory=list)
    root_score_mask: list[bool] = field(default_factory=list)
    temperatures: list[float] = field(default_factory=list)
    search_seeds: list[int] = field(default_factory=list)

    def append(self, result) -> None:
        self.moves.append(result.action)
        self.policy_actions.extend(result.policy_actions)
        self.policy_values.extend(result.policy_values)
        self.offsets.append(len(self.policy_actions))
        self.budgets.append(result.simulations)
        self.root_values.append(result.root_value)
        self.root_scores.append(result.root_score_margin)
        self.root_score_mask.append(result.root_score_available)
        self.temperatures.append(result.temperature)
        self.search_seeds.append(result.search_seed)


@dataclass(slots=True)
class _CollectionTotals:
    collected: int = 0
    skipped: int = 0
    positions: int = 0
    simulations: int = 0
    rules_seconds: float = 0.0

    def add(self, completed: _CompletedGame) -> None:
        self.collected += 1
        self.positions += completed.record.trainable_position_count
        self.simulations += completed.simulations
        self.rules_seconds += completed.rules_seconds


class SelfPlayCollector:
    def __init__(
        self,
        config: ExperimentConfig,
        broker: BatchedInferenceBroker,
        *,
        publication_sha256: str,
        evaluator_id: str,
        store_root: str | Path,
        catalog_path: str | Path,
        games: int,
        seed: int,
        target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
    ) -> None:
        if games <= 0 or target_shard_bytes <= 0:
            raise ValueError("games and target_shard_bytes must be positive")
        self.config = config
        self.broker = broker
        self.publication_sha256 = publication_sha256
        self.evaluator_id = evaluator_id
        self.store = ShardStore(store_root)
        self.catalog_path = Path(catalog_path)
        self.games = games
        self.seed = seed
        self.target_shard_bytes = target_shard_bytes

        self.search_config_sha256 = search_config_sha256(config)
        identity_payload = {
            "schema_version": SELFPLAY_TASK_SCHEMA.current,
            "publication_sha256": publication_sha256,
            "evaluator_id": evaluator_id,
            "search_config_sha256": self.search_config_sha256,
            "search": dataclasses.asdict(config.search),
            "selfplay": dataclasses.asdict(config.selfplay),
            "game": dataclasses.asdict(config.game),
            "rules_id": RULES_ID,
            "feature_schema_id": FEATURE_SCHEMA_ID,
            "requested_games": games,
            "seed": seed,
        }
        self.task_id = sha256_bytes(canonical_json_bytes(identity_payload))
        self._manifest = {"task_id": self.task_id, **identity_payload}

    def _write_manifest(self) -> tuple[str, str, int]:
        directory = self.store.root / "metadata" / "selfplay"
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"{self.task_id}.json"
        payload = canonical_json_bytes(self._manifest) + b"\n"
        if destination.exists():
            if destination.read_bytes() != payload:
                raise ValueError("conflicting self-play task manifest")
        else:
            atomic_write_bytes(destination, payload)
        relative = destination.relative_to(self.store.root).as_posix()
        return relative, sha256_bytes(payload), len(payload)

    def _play_game(self, ordinal: int, manifest_sha256: str) -> _CompletedGame:
        game = OpenSpielGoGame(self.config.game)
        state = game.new_initial_state()
        evaluator = OpenSpielEvaluator(self.broker)
        trace = _GameTrace()
        while not state.is_terminal():
            ply = state.local_state.move_number
            result = search_position(
                game,
                state,
                evaluator,
                self.config.search,
                search_seed=_derived_seed(self.task_id, ordinal, ply, "search"),
                selection_seed=_derived_seed(self.task_id, ordinal, ply, "selection"),
            )
            trace.append(result)
            state.apply_action(result.action)
        final = state.local_score()
        record = self._trajectory_record(ordinal, manifest_sha256, trace, final)
        return _CompletedGame(record, sum(trace.budgets), game.rules_seconds)

    def _trajectory_record(self, ordinal, manifest_sha256, trace, final):
        winner = final.winner
        value_black = 0.0 if winner is None else (1.0 if winner is Color.BLACK else -1.0)
        ownership = tuple(
            float(value) for value in np.frombuffer(final.score.ownership, dtype=np.int8)
        )
        length = len(trace.moves)
        return TrajectoryRecord(
            schema_version=RECORD_SCHEMA.current,
            game_id=_game_id(self.task_id, ordinal),
            content_sha256="",
            dataset_id=f"selfplay/{self.task_id}",
            asset_sha256=manifest_sha256,
            member_path=f"game-{ordinal:08d}",
            ordinal=ordinal,
            rules=RULES_ID,
            komi_half_points=self.config.game.komi_half_points,
            max_moves=self.config.game.max_moves,
            moves=tuple(trace.moves),
            trainable_start_ply=0,
            policy_row_offsets=tuple(trace.offsets),
            policy_actions=tuple(trace.policy_actions),
            policy_values=tuple(trace.policy_values),
            value_black=value_black,
            value_available=True,
            score_margin_black=final.score.margin_half_points / 2.0,
            score_available=True,
            ownership_black=ownership,
            ownership_available=True,
            source_kind="selfplay/mcts",
            task_id=self.task_id,
            termination=final.termination,
            game_seed=_derived_seed(self.task_id, ordinal, 0, "game"),
            black_agent_id=self.evaluator_id,
            white_agent_id=self.evaluator_id,
            publication_sha256=self.publication_sha256,
            feature_schema_id=FEATURE_SCHEMA_ID,
            search_config_sha256=self.search_config_sha256,
            search_budgets=tuple(trace.budgets),
            root_values=tuple(trace.root_values),
            root_score_margins=tuple(trace.root_scores),
            temperatures=tuple(trace.temperatures),
            search_seeds=tuple(trace.search_seeds),
            root_noise_mask=(self.config.search.dirichlet_epsilon > 0,) * length,
            search_metadata_mask=(True,) * length,
            root_score_mask=tuple(trace.root_score_mask),
        )

    def _missing_ordinals(self, catalog: Catalog) -> tuple[list[int], int]:
        missing = []
        skipped = 0
        for ordinal in range(self.games):
            if catalog.has_trajectory(_game_id(self.task_id, ordinal)):
                skipped += 1
            else:
                missing.append(ordinal)
        return missing, skipped

    def _collect_missing(
        self,
        missing: list[int],
        manifest_sha256: str,
        sink: TrajectoryShardSink,
        totals: _CollectionTotals,
    ) -> None:
        width = self.config.selfplay.actor_count
        for start in range(0, len(missing), width):
            ordinals = missing[start : start + width]
            with ThreadPoolExecutor(
                max_workers=width, thread_name_prefix="zero-ttt-selfplay"
            ) as pool:
                completed_games = list(
                    pool.map(
                        lambda value: self._play_game(value, manifest_sha256),
                        ordinals,
                    )
                )
            for completed in sorted(completed_games, key=lambda item: item.record.ordinal):
                sink.append(completed.record)
                totals.add(completed)

    def collect(self) -> CollectionSummary:
        started = time.perf_counter()
        relative, manifest_sha256, manifest_size = self._write_manifest()
        totals = _CollectionTotals()
        with Catalog(self.catalog_path, self.store) as catalog:
            sink = TrajectoryShardSink(
                self.store,
                catalog,
                self.target_shard_bytes,
            )
            catalog.register_selfplay_task(
                task_id=self.task_id,
                manifest_relative_path=relative,
                manifest_sha256=manifest_sha256,
                manifest_size_bytes=manifest_size,
                publication_sha256=self.publication_sha256,
                evaluator_id=self.evaluator_id,
                search_config_sha256=self.search_config_sha256,
                requested_games=self.games,
            )
            catalog.set_selfplay_task_status(self.task_id, "collecting")
            missing, totals.skipped = self._missing_ordinals(catalog)
            try:
                self._collect_missing(missing, manifest_sha256, sink, totals)
                sink.flush()
                catalog.set_selfplay_task_status(self.task_id, "sealed")
            except Exception:
                catalog.set_selfplay_task_status(self.task_id, "failed")
                raise

        wall_seconds = time.perf_counter() - started
        return CollectionSummary(
            task_id=self.task_id,
            requested_games=self.games,
            collected_games=totals.collected,
            skipped_games=totals.skipped,
            new_positions=totals.positions,
            new_shards=sink.shard_count,
            simulations=totals.simulations,
            wall_seconds=wall_seconds,
            simulations_per_second=(
                0.0 if wall_seconds == 0.0 else totals.simulations / wall_seconds
            ),
            rules_seconds=totals.rules_seconds,
            batching=self.broker.stats,
        )
