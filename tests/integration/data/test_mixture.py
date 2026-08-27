from __future__ import annotations

import dataclasses
import json
from argparse import Namespace

import numpy as np
import pytest

from zero_ttt.cli import _mixture_create
from zero_ttt.config import load_config
from zero_ttt.data import (
    Catalog,
    MixtureBatchSource,
    MixtureComponent,
    ShardStore,
    TrainingMixtureManifest,
)
from zero_ttt.data.manifest import ManifestAsset
from zero_ttt.learner import Learner, LearnerDataIdentity
from zero_ttt.training.checkpoint import CheckpointManager
from zero_ttt.versioning import TRAINING_MIXTURE_SCHEMA


@pytest.mark.parametrize("snapshot_id", ("z" * 64, "A" * 64, "a" * 63))
def test_mixture_component_requires_canonical_snapshot_sha(snapshot_id: str) -> None:
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        MixtureComponent(snapshot_id, 1.0)


@pytest.mark.parametrize("weight", (0.0, -1.0, float("nan"), float("inf")))
def test_mixture_component_requires_positive_finite_weight(weight: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        MixtureComponent("a" * 64, weight)


def test_mixture_cli_distinguishes_format_weight_and_snapshot_errors(tmp_path) -> None:
    output = tmp_path / "mixture.json"
    with pytest.raises(ValueError, match="SNAPSHOT=WEIGHT"):
        _mixture_create(Namespace(component=["missing-separator"], output=output))
    with pytest.raises(ValueError, match="weight must be numeric"):
        _mixture_create(Namespace(component=[f"{'a' * 64}=heavy"], output=output))
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        _mixture_create(Namespace(component=[f"{'z' * 64}=1"], output=output))


def test_previous_mixture_manifest_is_rejected_before_checksum(tmp_path) -> None:
    path = tmp_path / "mixture.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": TRAINING_MIXTURE_SCHEMA.current - 1,
                "components": [],
                "content_sha256": "",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"training mixture.*expected v2"):
        TrainingMixtureManifest.load(path)


@pytest.mark.parametrize("payload", ([], "mixture", 2, None))
def test_non_object_mixture_manifest_is_rejected(
    tmp_path,
    payload: object,
) -> None:
    path = tmp_path / "mixture.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid training mixture manifest"):
        TrainingMixtureManifest.load(path)


def test_mixture_manifest_source_and_learner_step(
    tmp_path,
    trajectory_factory,
    manifest_factory,
) -> None:
    first = trajectory_factory()
    second = dataclasses.replace(
        first,
        game_id="c" * 64,
        content_sha256="",
        ordinal=1,
    )
    store = ShardStore(tmp_path / "processed")
    catalog_path = tmp_path / "catalog.sqlite"
    asset = ManifestAsset("source.zip", first.asset_sha256, 1)
    with Catalog(catalog_path, store) as catalog:
        catalog.register_asset(manifest_factory(asset), asset)
        first_info = store.write_trajectories([first])
        catalog.commit_trajectory_shard(first_info, [first])
        first_snapshot = catalog.create_snapshot(1, validation_fraction=0.0)
        second_info = store.write_trajectories([second])
        catalog.commit_trajectory_shard(second_info, [second])
        second_snapshot = catalog.create_snapshot(1, validation_fraction=0.0)

    manifest = TrainingMixtureManifest(
        TRAINING_MIXTURE_SCHEMA.current,
        (
            MixtureComponent(first_snapshot, 0.8),
            MixtureComponent(second_snapshot, 0.2),
        ),
    )
    path = tmp_path / "mixture.json"
    manifest.save(path)
    loaded = TrainingMixtureManifest.load(path)
    assert loaded == manifest

    with MixtureBatchSource(catalog_path, store.root, loaded) as source:
        batch = source.next_batch(2, np.random.default_rng(3))
        assert batch.board.shape == (2, 25, 19, 19)
        identity = LearnerDataIdentity(
            snapshot_id=f"mixture:{loaded.content_sha256}",
            sampling_config_sha256=source.sampling_config_sha256,
            mixture_manifest_sha256=loaded.content_sha256,
            component_snapshot_ids=source.component_snapshot_ids,
        )
        learner = Learner(
            load_config("configs/test.toml"),
            CheckpointManager(tmp_path / "run", keep=1),
            data_identity=identity,
        )
        metrics = learner.train_optimizer_step(source, np.random.default_rng(4))
        checkpoint = learner.save_checkpoint(np.random.default_rng(5))
    assert metrics.step == 1
    payload = CheckpointManager.load(checkpoint)
    assert payload["data_identity"]["mixture_manifest_sha256"] == manifest.content_sha256
    assert tuple(payload["data_identity"]["component_snapshot_ids"]) == (
        first_snapshot,
        second_snapshot,
    )


def test_mixture_components_share_trajectory_shard_cache(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    trajectory_factory,
    manifest_factory,
) -> None:
    record = trajectory_factory()
    store = ShardStore(tmp_path / "processed")
    catalog_path = tmp_path / "catalog.sqlite"
    asset = ManifestAsset("source.zip", record.asset_sha256, 1)
    with Catalog(catalog_path, store) as catalog:
        catalog.register_asset(manifest_factory(asset), asset)
        info = store.write_trajectories([record])
        catalog.commit_trajectory_shard(info, [record])
        unfiltered = catalog.create_snapshot(1, validation_fraction=0.0)
        external = catalog.create_snapshot(1, validation_fraction=0.0, source_kind="external")
    assert unfiltered != external
    manifest = TrainingMixtureManifest(
        TRAINING_MIXTURE_SCHEMA.current,
        (MixtureComponent(unfiltered, 1.0), MixtureComponent(external, 1.0)),
    )
    calls = 0
    original = ShardStore.read_verified_trajectories

    def counting_read(self, relative_path: str, expected_sha256: str):
        nonlocal calls
        calls += 1
        return original(self, relative_path, expected_sha256)

    monkeypatch.setattr(ShardStore, "read_verified_trajectories", counting_read)
    with MixtureBatchSource(catalog_path, store.root, manifest) as source:
        source.sources[0].next_batch(2, np.random.default_rng(1))
        source.sources[1].next_batch(2, np.random.default_rng(2))
    assert calls == 1
