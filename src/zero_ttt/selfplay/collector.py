"""Concurrent OpenSpiel self-play with resumable, whole-game persistence."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from zero_ttt.config import ExperimentConfig
from zero_ttt.data.catalog import Catalog
from zero_ttt.data.records import RECORD_SCHEMA_VERSION, TrajectoryRecord
from zero_ttt.data.shards import ShardStore
from zero_ttt.game.features import FEATURE_SCHEMA_ID
from zero_ttt.game.rules import RULES_ID, Color
from zero_ttt.inference.batching import BatchedInferenceBroker, BatchingStats
from zero_ttt.search.open_spiel import (
    OpenSpielEvaluator,
    OpenSpielGoGame,
    search_position,
)


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _derived_seed(task_id: str, ordinal: int, ply: int, kind: str) -> int:
    payload = f"{task_id}:{ordinal}:{ply}:{kind}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _game_id(task_id: str, ordinal: int) -> str:
    return _sha256(_canonical_json(["zero-ttt-selfplay-game-v1", task_id, ordinal]))


def _estimated_record_bytes(record: TrajectoryRecord) -> int:
    return (
        len(record.moves) * 2
        + len(record.policy_actions) * 6
        + record.trainable_position_count * 48
        + 4096
    )


def search_config_sha256(config: ExperimentConfig) -> str:
    payload = {
        "search": dataclasses.asdict(config.search),
        "game": dataclasses.asdict(config.game),
        "rules_id": RULES_ID,
        "feature_schema_id": FEATURE_SCHEMA_ID,
    }
    return _sha256(_canonical_json(payload))


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
        target_shard_bytes: int = 128 * 1024 * 1024,
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
            "schema_version": 1,
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
        self.task_id = _sha256(_canonical_json(identity_payload))
        self._manifest = {"task_id": self.task_id, **identity_payload}

    def _write_manifest(self) -> tuple[str, str, int]:
        directory = self.store.root / "metadata" / "selfplay"
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"{self.task_id}.json"
        payload = _canonical_json(self._manifest) + b"\n"
        if destination.exists():
            if destination.read_bytes() != payload:
                raise ValueError("conflicting self-play task manifest")
        else:
            temporary = directory / f".{self.task_id}.{uuid.uuid4().hex}.tmp"
            try:
                with temporary.open("wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, destination)
            finally:
                if temporary.exists():
                    temporary.unlink()
        relative = destination.relative_to(self.store.root).as_posix()
        return relative, _sha256(payload), len(payload)

    def _play_game(self, ordinal: int, manifest_sha256: str) -> _CompletedGame:
        game = OpenSpielGoGame(self.config.game)
        state = game.new_initial_state()
        evaluator = OpenSpielEvaluator(self.broker)
        moves: list[int] = []
        offsets = [0]
        policy_actions: list[int] = []
        policy_values: list[float] = []
        budgets: list[int] = []
        root_values: list[float] = []
        root_scores: list[float] = []
        root_score_mask: list[bool] = []
        temperatures: list[float] = []
        search_seeds: list[int] = []
        game_seed = _derived_seed(self.task_id, ordinal, 0, "game")
        while not state.is_terminal():
            ply = state.local_state.move_number
            search_seed = _derived_seed(self.task_id, ordinal, ply, "search")
            selection_seed = _derived_seed(self.task_id, ordinal, ply, "selection")
            result = search_position(
                game,
                state,
                evaluator,
                self.config.search,
                search_seed=search_seed,
                selection_seed=selection_seed,
            )
            moves.append(result.action)
            policy_actions.extend(result.policy_actions)
            policy_values.extend(result.policy_values)
            offsets.append(len(policy_actions))
            budgets.append(result.simulations)
            root_values.append(result.root_value)
            root_scores.append(result.root_score_margin)
            root_score_mask.append(result.root_score_available)
            temperatures.append(result.temperature)
            search_seeds.append(result.search_seed)
            state.apply_action(result.action)

        final = state.local_score()
        winner = final.winner
        value_black = 0.0 if winner is None else (1.0 if winner is Color.BLACK else -1.0)
        ownership = tuple(
            float(value)
            for value in np.frombuffer(final.score.ownership, dtype=np.int8)
        )
        length = len(moves)
        record = TrajectoryRecord(
            schema_version=RECORD_SCHEMA_VERSION,
            game_id=_game_id(self.task_id, ordinal),
            content_sha256="",
            dataset_id=f"selfplay/{self.task_id}",
            asset_sha256=manifest_sha256,
            member_path=f"game-{ordinal:08d}",
            ordinal=ordinal,
            rules=RULES_ID,
            komi_half_points=self.config.game.komi_half_points,
            max_moves=self.config.game.max_moves,
            moves=tuple(moves),
            trainable_start_ply=0,
            policy_row_offsets=tuple(offsets),
            policy_actions=tuple(policy_actions),
            policy_values=tuple(policy_values),
            value_black=value_black,
            value_available=True,
            score_margin_black=final.score.margin_half_points / 2.0,
            score_available=True,
            ownership_black=ownership,
            ownership_available=True,
            source_kind="selfplay/mcts",
            task_id=self.task_id,
            termination=final.termination,
            game_seed=game_seed,
            black_agent_id=self.evaluator_id,
            white_agent_id=self.evaluator_id,
            publication_sha256=self.publication_sha256,
            feature_schema_id=FEATURE_SCHEMA_ID,
            search_config_sha256=self.search_config_sha256,
            search_budgets=tuple(budgets),
            root_values=tuple(root_values),
            root_score_margins=tuple(root_scores),
            temperatures=tuple(temperatures),
            search_seeds=tuple(search_seeds),
            root_noise_mask=(self.config.search.dirichlet_epsilon > 0,) * length,
            search_metadata_mask=(True,) * length,
            root_score_mask=tuple(root_score_mask),
        )
        return _CompletedGame(
            record=record,
            simulations=sum(budgets),
            rules_seconds=game.rules_seconds,
        )

    def collect(self) -> CollectionSummary:
        started = time.perf_counter()
        relative, manifest_sha256, manifest_size = self._write_manifest()
        collected = 0
        skipped = 0
        positions = 0
        shards = 0
        simulations = 0
        rules_seconds = 0.0
        pending: list[TrajectoryRecord] = []
        pending_bytes = 0

        with Catalog(self.catalog_path, self.store) as catalog:
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
            missing = []
            for ordinal in range(self.games):
                if catalog.has_trajectory(_game_id(self.task_id, ordinal)):
                    skipped += 1
                else:
                    missing.append(ordinal)

            def flush() -> None:
                nonlocal pending, pending_bytes, shards
                if not pending:
                    return
                info = self.store.write_trajectories(pending)
                catalog.commit_trajectory_shard(info, pending)
                shards += 1
                pending = []
                pending_bytes = 0

            try:
                width = self.config.selfplay.actor_count
                for start in range(0, len(missing), width):
                    ordinals = missing[start : start + width]
                    with ThreadPoolExecutor(
                        max_workers=width,
                        thread_name_prefix="zero-ttt-selfplay",
                    ) as pool:
                        completed_games = list(
                            pool.map(
                                lambda value: self._play_game(value, manifest_sha256),
                                ordinals,
                            )
                        )
                    for completed in sorted(
                        completed_games,
                        key=lambda item: item.record.ordinal,
                    ):
                        record = completed.record
                        pending.append(record)
                        size = _estimated_record_bytes(record)
                        pending_bytes += size
                        collected += 1
                        positions += record.trainable_position_count
                        simulations += completed.simulations
                        rules_seconds += completed.rules_seconds
                        if pending_bytes >= self.target_shard_bytes:
                            flush()
                flush()
                catalog.set_selfplay_task_status(self.task_id, "sealed")
            except BaseException:
                catalog.set_selfplay_task_status(self.task_id, "failed")
                raise

        wall_seconds = time.perf_counter() - started
        return CollectionSummary(
            task_id=self.task_id,
            requested_games=self.games,
            collected_games=collected,
            skipped_games=skipped,
            new_positions=positions,
            new_shards=shards,
            simulations=simulations,
            wall_seconds=wall_seconds,
            simulations_per_second=(
                0.0 if wall_seconds == 0.0 else simulations / wall_seconds
            ),
            rules_seconds=rules_seconds,
            batching=self.broker.stats,
        )
