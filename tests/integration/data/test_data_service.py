from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

from zero_ttt_contracts import ArtifactKind
from zero_ttt_data.manifest import ManifestAsset
from zero_ttt_data.selfplay_admission import admit_bundle
from zero_ttt_data.snapshots import SnapshotSpec
from zero_ttt_data.unit_of_work import DataUnitOfWork
from zero_ttt_dataset import LocalArtifactStore, SelfPlayBundle, SelfPlayShard, ShardStore


def test_private_uow_commits_and_reopens_snapshot(
    tmp_path: Path, trajectory_factory, manifest_factory
) -> None:
    database = tmp_path / "state" / "data.sqlite"
    shards = tmp_path / "artifacts" / "data" / "shards"
    record = trajectory_factory()
    asset = ManifestAsset("source.zip", record.asset_sha256, 1)
    manifest = manifest_factory(asset)
    with DataUnitOfWork(database, shards) as unit:
        unit.repository.register_asset(manifest, asset)
        info = unit.store.write_trajectories([record])
        unit.repository.commit_trajectory_shard(info, [record], [])
        snapshot_id = unit.snapshots.create(
            SnapshotSpec(seed=9, split="train", validation_fraction=0.0)
        )
        unit.lifecycle.verify()

    with DataUnitOfWork(database, shards) as reopened:
        assert reopened.repository.has_trajectory(record.game_id)
        locators = reopened.snapshots.trajectories(snapshot_id)
        assert len(locators) == 1
        assert locators[0].content_sha256 == record.content_sha256
        assert (
            reopened.snapshots.create(SnapshotSpec(seed=9, split="train", validation_fraction=0.0))
            == snapshot_id
        )


def test_recovery_removes_only_unregistered_content_addressed_shards(
    tmp_path: Path, trajectory_factory, manifest_factory
) -> None:
    database = tmp_path / "state" / "data.sqlite"
    shards = tmp_path / "artifacts" / "data" / "shards"
    record = trajectory_factory()
    with DataUnitOfWork(database, shards) as unit:
        orphan = unit.store.write_trajectories([record])
        assert unit.store.resolve(orphan.relative_path).is_file()
        recovered = unit.lifecycle.recover()
        assert orphan.relative_path in recovered
        assert not unit.store.resolve(orphan.relative_path).exists()

        asset = ManifestAsset("source.zip", record.asset_sha256, 1)
        unit.repository.register_asset(manifest_factory(asset), asset)
        committed = unit.store.write_trajectories([record])
        unit.repository.commit_trajectory_shard(committed, [record], [])
        assert unit.lifecycle.recover() == ()
        assert unit.store.resolve(committed.relative_path).is_file()


def test_selfplay_bundle_admission_is_hash_checked_and_idempotent(
    tmp_path: Path, trajectory_factory
) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_store = LocalArtifactStore(artifact_root)
    source_manifest = artifact_root / "selfplay" / "tasks" / "wf" / "source.json"
    source_manifest.parent.mkdir(parents=True)
    source_manifest.write_bytes(b'{"sealed":true}\n')
    source_sha = hashlib.sha256(source_manifest.read_bytes()).hexdigest()
    base = trajectory_factory(source_sha)
    positions = base.trainable_position_count
    record = dataclasses.replace(
        base,
        content_sha256="",
        source_kind="selfplay/mcts",
        task_id="collection",
        termination="passes",
        black_agent_id="evaluator",
        white_agent_id="evaluator",
        publication_sha256="a" * 64,
        feature_schema_id="go19-features-v1",
        search_config_sha256="b" * 64,
        search_budgets=(2,) * positions,
        root_values=(0.0,) * positions,
        root_score_margins=(0.0,) * positions,
        temperatures=(0.0,) * positions,
        search_seeds=tuple(range(positions)),
        root_noise_mask=(False,) * positions,
        search_metadata_mask=(True,) * positions,
        root_score_mask=(True,) * positions,
    )
    producer_store = ShardStore(artifact_root / "selfplay" / "tasks" / "wf" / "shards")
    info = producer_store.write_trajectories([record])
    shard_path = producer_store.resolve(info.relative_path)
    shard_uri = f"artifact://{shard_path.relative_to(artifact_root).as_posix()}"
    bundle = SelfPlayBundle(
        task_id="wf",
        source_manifest_uri="artifact://selfplay/tasks/wf/source.json",
        source_manifest_sha256=source_sha,
        source_manifest_size_bytes=source_manifest.stat().st_size,
        publication_sha256="a" * 64,
        evaluator_id="evaluator",
        search_config_sha256="b" * 64,
        requested_games=1,
        collected_games=1,
        shards=(
            SelfPlayShard(
                uri=shard_uri,
                sha256=info.sha256,
                size_bytes=info.size_bytes,
                games=1,
                positions=positions,
            ),
        ),
    )
    reference = artifact_store.commit_json(
        uri="artifact://selfplay/tasks/wf/bundle.json",
        artifact_id="selfplay.wf",
        kind=ArtifactKind.SELFPLAY_BUNDLE,
        value=bundle.model_dump(mode="json"),
        format_version=1,
    )
    arguments = {
        "artifact_store": artifact_store,
        "database_path": tmp_path / "state" / "data.sqlite",
        "shard_root": artifact_root / "data" / "shards",
        "task_id": "wf",
    }
    assert admit_bundle(reference, **arguments)["games"] == 1
    assert admit_bundle(reference, **arguments)["games"] == 1
    with DataUnitOfWork(arguments["database_path"], arguments["shard_root"]) as unit:
        assert unit.repository.selfplay_statistics().games == 1
