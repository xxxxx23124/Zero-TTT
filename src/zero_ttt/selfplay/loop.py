"""Single-GPU phased self-play, training, EMA, and publication controller."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from zero_ttt.config import ExperimentConfig
from zero_ttt.model.transformer import PolicyValueTransformer
from zero_ttt.replay.sampler import ReplaySampler
from zero_ttt.replay.sqlite_store import ReplayStore
from zero_ttt.search.inference import InferenceServer, TorchBatchEvaluator
from zero_ttt.selfplay.actor import SelfPlayActor
from zero_ttt.training.checkpoint import CheckpointManager
from zero_ttt.training.trainer import StepMetrics, Trainer


@dataclass(frozen=True, slots=True)
class CycleResult:
    games: int
    new_positions: int
    replay_positions: int
    optimizer_steps: int
    final_optimizer_step: int


class CoreLoop:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        config.run_dir.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        self.rng = np.random.default_rng(config.seed)
        self.checkpoints = CheckpointManager(
            config.run_dir,
            keep=config.training.checkpoint_keep,
        )
        self.replay = ReplayStore(
            config.run_dir / config.replay.database_name,
            capacity_positions=config.replay.capacity_positions,
            decoded_cache_games=config.replay.decoded_cache_games,
        )
        self.sampler = ReplaySampler(self.replay, config.replay.decoded_cache_games)
        self.trainer = Trainer(config, self.checkpoints)
        latest = self.checkpoints.latest_checkpoint()
        if latest is not None:
            self.trainer.restore(latest, self.rng)
        else:
            self.trainer.publish()
            self.save()
        current_publication = self.checkpoints.current_publication()
        if current_publication is None:
            raise RuntimeError("checkpoint exists without a current publication")
        publication = self.checkpoints.load_publication(current_publication)
        if publication["config_sha256"] != config.sha256:
            raise ValueError("publication configuration does not match this run")
        if publication["model_version"] != self.trainer.state.last_published_step:
            raise ValueError("publication version does not match trainer state")
        inference_model = PolicyValueTransformer(config.model)
        inference_model.load_state_dict(publication["slow_state"])
        self.batch_backend = TorchBatchEvaluator(
            inference_model,
            config.runtime,
            config.search.max_batch_size,
            publication["model_version"],
        )
        self.inference = InferenceServer(self.batch_backend, config.search)
        self.actor = SelfPlayActor(config, self.inference)
        self.metrics_path = config.run_dir / "metrics.jsonl"

    def _log(self, kind: str, payload: dict) -> None:
        record = {"kind": kind, **payload}
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def selfplay_phase(self, games: int | None = None) -> tuple[int, int]:
        count = self.config.selfplay.games_per_cycle if games is None else games
        model_version = self.batch_backend.model_version
        positions = 0
        for _ in range(count):
            game = self.actor.play_game(model_version, self.rng)
            self.replay.add_game(game)
            positions += game.length
            self._log(
                "selfplay_game",
                {
                    "model_version": model_version,
                    "positions": game.length,
                    "termination": game.termination,
                    "margin_half_points": game.final_margin_half_points,
                },
            )
        return count, positions

    def _train(self, steps: int) -> list[StepMetrics]:
        if steps <= 0:
            return []
        previous = self.trainer.state.optimizer_step
        metrics = self.trainer.train_steps(steps, self.sampler, self.rng)
        for item in metrics:
            self._log("train_step", asdict(item))
        interval = self.config.training.publish_interval
        if previous // interval < self.trainer.state.optimizer_step // interval:
            publication = self.trainer.publish()
            publication_payload = self.checkpoints.load_publication(publication)
            self._replace_publication(publication_payload)
            self.save()
            self._log(
                "publication",
                {"step": self.trainer.state.optimizer_step, "path": str(publication)},
            )
        return metrics

    def _replace_publication(self, publication: dict) -> None:
        if publication["config_sha256"] != self.config.sha256:
            raise ValueError("publication configuration does not match this run")
        self.inference.close()
        self.batch_backend.load_publication(
            publication["slow_state"],
            publication["model_version"],
        )
        self.inference = InferenceServer(self.batch_backend, self.config.search)
        self.actor = SelfPlayActor(self.config, self.inference)

    def train_for_new_positions(self, new_positions: int) -> list[StepMetrics]:
        if self.replay.position_count < self.config.selfplay.minimum_replay_positions:
            return []
        effective_batch = (
            self.config.training.batch_size * self.config.training.accumulation_steps
        )
        target_samples = new_positions * self.config.selfplay.train_samples_per_new_position
        steps = math.ceil(target_samples / effective_batch)
        return self._train(steps)

    def train_replay_once(self) -> list[StepMetrics]:
        if self.replay.position_count == 0:
            raise RuntimeError("cannot train with an empty replay")
        effective_batch = (
            self.config.training.batch_size * self.config.training.accumulation_steps
        )
        return self._train(math.ceil(self.replay.position_count / effective_batch))

    def run_cycle(self) -> CycleResult:
        games, positions = self.selfplay_phase()
        metrics = self.train_for_new_positions(positions)
        result = CycleResult(
            games=games,
            new_positions=positions,
            replay_positions=self.replay.position_count,
            optimizer_steps=len(metrics),
            final_optimizer_step=self.trainer.state.optimizer_step,
        )
        self._log("cycle", asdict(result))
        return result

    def save(self) -> Path:
        return self.trainer.save_checkpoint(
            self.rng,
            replay_metadata={
                "database_name": self.config.replay.database_name,
                "game_count": self.replay.game_count,
                "position_count": self.replay.position_count,
            },
        )

    def close(self) -> None:
        self.inference.close()
        self.replay.close()

    def __enter__(self) -> "CoreLoop":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
