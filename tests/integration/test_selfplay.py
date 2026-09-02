from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
import torch
from zero_ttt.config import load_config
from zero_ttt.versioning import SELFPLAY_TASK_SCHEMA
from zero_ttt_dataset import PortableSnapshotBatchSource, ShardStore, SnapshotManifest
from zero_ttt_dataset.locators import TrajectoryLocator
from zero_ttt_selfplay_worker.collector import SelfPlayCollector
from zero_ttt_selfplay_worker.inference import (
    BatchedInferenceBroker,
    InferenceBatch,
    InferenceOutput,
)


class UniformEvaluator:
    model_version = "uniform-selfplay-v1"

    def evaluate(self, batch: InferenceBatch) -> InferenceOutput:
        size = batch.board.shape[0]
        return InferenceOutput(
            policy_logits=torch.zeros(size, 362),
            value=torch.zeros(size),
            ownership=torch.zeros(size, 361),
            score_margin=torch.zeros(size),
        )


def _records(store: ShardStore):
    result = []
    for path in sorted(store.trajectory_dir.glob("*.npz")):
        result.extend(store.read_trajectories(path))
    return tuple(result)


@pytest.mark.parametrize(("epsilon", "expected"), ((0.0, False), (0.25, True)))
def test_selfplay_root_noise_mask_tracks_nonzero_epsilon(tmp_path, epsilon, expected) -> None:
    base = load_config("configs/test.toml")
    config = dataclasses.replace(
        base,
        game=dataclasses.replace(base.game, max_moves=2),
        search=dataclasses.replace(
            base.search,
            max_simulations=2,
            dirichlet_epsilon=epsilon,
            temperature=0.0,
            temperature_drop_ply=0,
        ),
        selfplay=dataclasses.replace(base.selfplay, actor_count=1, inference_batch_size=1),
    )
    store_root = tmp_path / "shards"
    with BatchedInferenceBroker(
        UniformEvaluator(), batch_size=1, batch_wait_ms=0, cache_size=16
    ) as broker:
        summary = SelfPlayCollector(
            config,
            broker,
            publication_sha256="a" * 64,
            evaluator_id="b" * 64,
            store_root=store_root,
            games=1,
            seed=7,
            target_shard_bytes=1024 * 1024,
        ).collect()
    manifest = json.loads(
        (store_root / "metadata" / "selfplay" / f"{summary.task_id}.json").read_text()
    )
    assert manifest["schema_version"] == SELFPLAY_TASK_SCHEMA.current
    record = _records(ShardStore(store_root, read_only=True))[0]
    assert record.root_noise_mask == (expected,) * record.trainable_position_count
    assert all(np.isclose(sum(record.policy_at(index)[1]), 1.0) for index in range(2))


def test_selfplay_retry_is_idempotent_and_portable(tmp_path) -> None:
    base = load_config("configs/test.toml")
    config = dataclasses.replace(
        base,
        game=dataclasses.replace(base.game, max_moves=2),
        search=dataclasses.replace(
            base.search, max_simulations=2, temperature=0.0, temperature_drop_ply=0
        ),
    )
    store_root = tmp_path / "shards"
    with BatchedInferenceBroker(
        UniformEvaluator(), batch_size=16, batch_wait_ms=10, cache_size=2048
    ) as broker:
        collector = SelfPlayCollector(
            config,
            broker,
            publication_sha256="a" * 64,
            evaluator_id="b" * 64,
            store_root=store_root,
            games=16,
            seed=7,
            target_shard_bytes=1024 * 1024,
        )
        first = collector.collect()
        second = collector.collect()
    assert (first.collected_games, first.new_positions) == (16, 32)
    assert (second.collected_games, second.skipped_games) == (0, 16)

    store = ShardStore(store_root, read_only=True)
    locators = []
    for path in sorted(store.trajectory_dir.glob("*.npz")):
        records = store.read_trajectories(path)
        relative = path.relative_to(store.root).as_posix()
        for index, record in enumerate(records):
            locators.append(
                TrajectoryLocator(
                    record.game_id,
                    record.content_sha256,
                    path.stem,
                    relative,
                    index,
                    record.trainable_start_ply,
                    record.trainable_position_count,
                )
            )
    snapshot = SnapshotManifest.from_locators(
        snapshot_id="selfplay-test",
        seed=7,
        split="train",
        validation_fraction=0.0,
        source_kind="selfplay",
        task_id=first.task_id,
        trajectories=tuple(locators),
    )
    with PortableSnapshotBatchSource(snapshot, store_root) as source:
        batch = source.next_batch(4, np.random.default_rng(2))
    assert batch.board.shape == (4, 25, 19, 19)
    assert np.all(batch.value_mask)
