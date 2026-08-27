from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
import torch

from zero_ttt.config import load_config
from zero_ttt.data import (
    Catalog,
    CatalogBatchSource,
    MixtureBatchSource,
    MixtureComponent,
    ShardStore,
    TrainingMixtureManifest,
)
from zero_ttt.data.manifest import ManifestAsset, SourceManifest
from zero_ttt.inference import (
    BatchedInferenceBroker,
    InferenceBatch,
    InferenceOutput,
    PublicationPositionEvaluator,
)
from zero_ttt.learner import Learner, LearnerDataIdentity
from zero_ttt.model import PolicyValueTransformer
from zero_ttt.selfplay import SelfPlayCollector
from zero_ttt.training.checkpoint import CheckpointManager, checkpoint_metadata
from zero_ttt.versioning import (
    SELFPLAY_TASK_SCHEMA,
    SOURCE_MANIFEST_SCHEMA,
    TRAINING_MIXTURE_SCHEMA,
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


@pytest.mark.parametrize(
    ("dirichlet_epsilon", "expected_noise"),
    ((0.0, False), (0.25, True)),
)
def test_selfplay_root_noise_mask_tracks_nonzero_epsilon(
    tmp_path,
    dirichlet_epsilon: float,
    expected_noise: bool,
) -> None:
    base = load_config("configs/test.toml")
    config = dataclasses.replace(
        base,
        game=dataclasses.replace(base.game, max_moves=2),
        search=dataclasses.replace(
            base.search,
            max_simulations=2,
            dirichlet_epsilon=dirichlet_epsilon,
            temperature=0.0,
            temperature_drop_ply=0,
        ),
        selfplay=dataclasses.replace(
            base.selfplay,
            actor_count=1,
            inference_batch_size=1,
        ),
    )
    store_root = tmp_path / "processed"
    catalog_path = tmp_path / "catalog.sqlite"
    with BatchedInferenceBroker(
        UniformEvaluator(), batch_size=1, batch_wait_ms=0, cache_size=16
    ) as broker:
        summary = SelfPlayCollector(
            config,
            broker,
            publication_sha256="a" * 64,
            evaluator_id="b" * 64,
            store_root=store_root,
            catalog_path=catalog_path,
            games=1,
            seed=7,
            target_shard_bytes=1024 * 1024,
        ).collect()

    store = ShardStore(store_root)
    task_manifest = json.loads(
        (store_root / "metadata" / "selfplay" / f"{summary.task_id}.json").read_text(
            encoding="utf-8"
        )
    )
    assert task_manifest["schema_version"] == SELFPLAY_TASK_SCHEMA.current
    with Catalog(catalog_path, store) as catalog:
        snapshot = catalog.create_snapshot(
            5,
            validation_fraction=0.0,
            source_kind="selfplay",
            task_id=summary.task_id,
        )
        locator = catalog.snapshot_trajectories(snapshot)[0]
        record = store.read_verified_trajectories(
            locator.relative_path,
            locator.shard_sha256,
        )[locator.game_index]
    assert record.root_noise_mask == (expected_noise,) * record.trainable_position_count
    for position in range(record.trainable_position_count):
        _, policy = record.policy_at(position)
        assert np.isclose(sum(policy), 1.0)


def test_selfplay_collect_resume_snapshot_and_training_batch(tmp_path) -> None:
    base = load_config("configs/test.toml")
    config = dataclasses.replace(
        base,
        game=dataclasses.replace(base.game, max_moves=2),
        search=dataclasses.replace(
            base.search,
            max_simulations=2,
            temperature=0.0,
            temperature_drop_ply=0,
        ),
    )
    store_root = tmp_path / "processed"
    catalog_path = tmp_path / "catalog.sqlite"
    with BatchedInferenceBroker(
        UniformEvaluator(), batch_size=16, batch_wait_ms=10, cache_size=2048
    ) as broker:
        collector = SelfPlayCollector(
            config,
            broker,
            publication_sha256="a" * 64,
            evaluator_id="b" * 64,
            store_root=store_root,
            catalog_path=catalog_path,
            games=16,
            seed=7,
            target_shard_bytes=1024 * 1024,
        )
        first = collector.collect()
        second = collector.collect()
    assert first.collected_games == 16
    assert first.new_positions == 32
    assert second.collected_games == 0
    assert second.skipped_games == 16

    store = ShardStore(store_root)
    with Catalog(catalog_path, store) as catalog:
        snapshot = catalog.create_snapshot(
            5,
            validation_fraction=0.0,
            source_kind="selfplay",
            task_id=first.task_id,
        )
        records = catalog.snapshot_trajectories(snapshot)
        assert len(records) == 16
        first_locator = records[0]
        first_record = store.read_verified_trajectories(
            first_locator.relative_path,
            first_locator.shard_sha256,
        )[first_locator.game_index]
        assert first_record.source_kind == "selfplay/mcts"
        assert first_record.search_budgets == (2, 2)
        assert all(first_record.root_noise_mask)
        changed = dataclasses.replace(
            first_record,
            content_sha256="",
            root_values=(first_record.root_values[0] + 0.25, first_record.root_values[1]),
        )
        assert changed.content_sha256 != first_record.content_sha256
    with CatalogBatchSource(catalog_path, store_root, snapshot) as source:
        batch = source.next_batch(4, np.random.default_rng(2))
    assert batch.board.shape == (4, 25, 19, 19)
    assert np.all(batch.value_mask)


def test_tiny_publication_selfplay_mixture_and_learner_step(
    tmp_path,
) -> None:
    base = load_config("configs/test.toml")
    config = dataclasses.replace(
        base,
        game=dataclasses.replace(base.game, max_moves=2),
        search=dataclasses.replace(base.search, max_simulations=2),
    )
    publication_manager = CheckpointManager(tmp_path / "publication", keep=1)
    model = PolicyValueTransformer(config.model)
    publication = publication_manager.save_publication(
        "tiny-selfplay",
        1,
        4,
        model.state_dict(),
        checkpoint_metadata(config.canonical_json(), config.sha256),
    )
    evaluator = PublicationPositionEvaluator(
        publication,
        device="cpu",
        inference_batch_size=16,
    )
    store_root = tmp_path / "processed"
    catalog_path = tmp_path / "catalog.sqlite"
    with BatchedInferenceBroker(
        evaluator, batch_size=16, batch_wait_ms=50, cache_size=2048
    ) as broker:
        summary = SelfPlayCollector(
            config,
            broker,
            publication_sha256=evaluator.publication_sha256,
            evaluator_id="d" * 64,
            store_root=store_root,
            catalog_path=catalog_path,
            games=16,
            seed=13,
            target_shard_bytes=1024 * 1024,
        ).collect()
    assert summary.collected_games == 16

    store = ShardStore(store_root)
    with Catalog(catalog_path, store) as catalog:
        selfplay_snapshot = catalog.create_snapshot(
            3,
            validation_fraction=0.0,
            source_kind="selfplay",
            task_id=summary.task_id,
        )
        locator = catalog.snapshot_trajectories(selfplay_snapshot)[0]
        generated = store.read_verified_trajectories(
            locator.relative_path,
            locator.shard_sha256,
        )[locator.game_index]
        cold_start = dataclasses.replace(
            generated,
            game_id="c" * 64,
            content_sha256="",
            dataset_id="cold-start",
            asset_sha256="e" * 64,
            member_path="rehearsal/game-00000000",
            ordinal=0,
            source_kind="external/played_move",
            task_id="",
            termination="external",
            game_seed=0,
            black_agent_id="",
            white_agent_id="",
            publication_sha256="",
            feature_schema_id="",
            search_config_sha256="",
            search_budgets=(),
            root_values=(),
            root_score_margins=(),
            temperatures=(),
            search_seeds=(),
            root_noise_mask=(),
            search_metadata_mask=(),
            root_score_mask=(),
        )
        asset = ManifestAsset("cold-start.zip", cold_start.asset_sha256, 1)
        manifest = SourceManifest(
            schema_version=SOURCE_MANIFEST_SCHEMA.current,
            dataset_id="cold-start",
            source_type="test-rehearsal",
            license_id="test-only",
            license_url="https://example.invalid/license",
            assets=(asset,),
        )
        catalog.register_asset(manifest, asset)
        cold_info = store.write_trajectories([cold_start])
        catalog.commit_trajectory_shard(cold_info, [cold_start])
        cold_snapshot = catalog.create_snapshot(
            3,
            validation_fraction=0.0,
            source_kind="external",
        )

    mixture = TrainingMixtureManifest(
        TRAINING_MIXTURE_SCHEMA.current,
        (
            MixtureComponent(selfplay_snapshot, 0.8),
            MixtureComponent(cold_snapshot, 0.2),
        ),
    )
    with MixtureBatchSource(catalog_path, store_root, mixture) as source:
        learner = Learner(
            config,
            CheckpointManager(tmp_path / "learner", keep=1),
            data_identity=LearnerDataIdentity(
                snapshot_id=f"mixture:{mixture.content_sha256}",
                sampling_config_sha256=source.sampling_config_sha256,
                mixture_manifest_sha256=mixture.content_sha256,
                component_snapshot_ids=source.component_snapshot_ids,
            ),
        )
        metrics = learner.train_optimizer_step(source, np.random.default_rng(17))
    assert metrics.step == 1
