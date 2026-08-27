from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from zero_ttt.config import load_config
from zero_ttt.console.config import ConsoleConfig
from zero_ttt.console.engine import TrainingConsole
from zero_ttt.console.state import Operation, TrainingPhase
from zero_ttt.data.catalog import Catalog
from zero_ttt.data.manifest import ManifestAsset, SourceManifest
from zero_ttt.data.records import TrajectoryRecord
from zero_ttt.data.shards import ShardStore
from zero_ttt.data.synthetic import SyntheticBatchSource
from zero_ttt.game.rules import BOARD_AREA
from zero_ttt.learner import Learner
from zero_ttt.training.checkpoint import checkpoint_metadata
from zero_ttt.versioning import RECORD_SCHEMA, SOURCE_MANIFEST_SCHEMA


class OneStepClock:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        return 0.0 if self.calls <= 2 else 10_000.0


def _record(asset_sha256: str, game_id: str) -> TrajectoryRecord:
    moves = (0, 1, 19, 20)
    return TrajectoryRecord(
        schema_version=RECORD_SCHEMA.current,
        game_id=game_id,
        content_sha256="",
        dataset_id="console-test",
        asset_sha256=asset_sha256,
        member_path=f"games/{game_id}.sgf",
        ordinal=0,
        rules="koPOSITIONALscoreAREAtaxNONEsui1",
        komi_half_points=0,
        max_moves=722,
        moves=moves,
        trainable_start_ply=0,
        policy_row_offsets=(0, 1, 2, 3, 4),
        policy_actions=moves,
        policy_values=(1.0, 1.0, 1.0, 1.0),
        value_black=1.0,
        value_available=True,
        score_margin_black=2.0,
        score_available=True,
        ownership_black=(0.0,) * BOARD_AREA,
        ownership_available=False,
    )


def _experiment_config(tmp_path: Path, *, fast_selfplay: bool = False) -> Path:
    destination = tmp_path / "experiment.toml"
    payload = Path("configs/test.toml").read_text(encoding="utf-8")
    run_dir = (tmp_path / "run").as_posix()
    payload = payload.replace('run_dir = "runs/test"', f'run_dir = "{run_dir}"')
    if fast_selfplay:
        payload = payload.replace("max_moves = 12", "max_moves = 2")
        payload = payload.replace("max_simulations = 64", "max_simulations = 2")
    destination.write_text(payload, encoding="utf-8")
    load_config(destination)
    return destination


def _cold_data(tmp_path: Path) -> tuple[Path, Path, str]:
    store = ShardStore(tmp_path / "processed")
    catalog_path = tmp_path / "catalog.sqlite"
    record = _record("a" * 64, "b" * 64)
    asset = ManifestAsset("cold.zip", record.asset_sha256, 1)
    manifest = SourceManifest(
        SOURCE_MANIFEST_SCHEMA.current,
        "console-cold",
        "katago-g170-sgfs-zip",
        "CC0-1.0",
        "https://example.invalid/license",
        (asset,),
    )
    with Catalog(catalog_path, store) as catalog:
        catalog.register_asset(manifest, asset)
        info = store.write_trajectories([record])
        catalog.commit_trajectory_shard(info, [record])
        snapshot = catalog.create_snapshot(
            seed=7,
            validation_fraction=0.0,
            source_kind="external",
        )
    return catalog_path, store.root, snapshot


def _add_selfplay(
    catalog_path: Path,
    store_root: Path,
    *,
    task_id: str = "c" * 64,
    manifest_sha: str = "d" * 64,
    game_id: str = "e" * 64,
    status: str = "sealed",
) -> None:
    store = ShardStore(store_root)
    base = _record(manifest_sha, game_id)
    record = dataclasses.replace(
        base,
        content_sha256="",
        source_kind="selfplay/mcts",
        task_id=task_id,
        termination="two_passes",
        game_seed=5,
        black_agent_id="black",
        white_agent_id="white",
        publication_sha256="f" * 64,
        feature_schema_id="features",
        search_config_sha256="1" * 64,
        search_budgets=(2, 2, 2, 2),
        root_values=(0.0, 0.0, 0.0, 0.0),
        root_score_margins=(0.0, 0.0, 0.0, 0.0),
        temperatures=(1.0, 1.0, 0.0, 0.0),
        search_seeds=(1, 2, 3, 4),
        root_noise_mask=(True, True, True, True),
        search_metadata_mask=(True, True, True, True),
        root_score_mask=(False, False, False, False),
    )
    with Catalog(catalog_path, store) as catalog:
        catalog.register_selfplay_task(
            task_id=task_id,
            manifest_relative_path=f"metadata/selfplay/{task_id}.json",
            manifest_sha256=manifest_sha,
            manifest_size_bytes=1,
            publication_sha256="f" * 64,
            evaluator_id="2" * 64,
            search_config_sha256="1" * 64,
            requested_games=1,
        )
        info = store.write_trajectories([record])
        catalog.commit_trajectory_shard(info, [record])
        catalog.set_selfplay_task_status(task_id, status)


def _console(tmp_path: Path) -> TrainingConsole:
    catalog_path, store_root, cold_snapshot = _cold_data(tmp_path)
    settings = ConsoleConfig(
        schema_version=1,
        experiment_config=_experiment_config(tmp_path),
        catalog_path=catalog_path,
        store_root=store_root,
        cold_start_snapshot_id=cold_snapshot,
        max_runtime_hours=1.0,
    )
    return TrainingConsole(settings, clock=OneStepClock(), output=lambda _line: None)


def _cold_identity(console: TrainingConsole):
    plan = console.data_planner.build(use_mixture=False)
    try:
        return plan.identity
    finally:
        plan.close()


def test_cold_train_warm_start_and_restart_status(tmp_path: Path) -> None:
    catalog_path, store_root, cold_snapshot = _cold_data(tmp_path)
    settings = ConsoleConfig(
        schema_version=1,
        experiment_config=_experiment_config(tmp_path),
        catalog_path=catalog_path,
        store_root=store_root,
        cold_start_snapshot_id=cold_snapshot,
        max_runtime_hours=1.0,
    )
    console = TrainingConsole(settings, clock=OneStepClock(), output=lambda _line: None)
    console.reconcile()
    console.train()
    cold_status = console.status()
    assert cold_status.optimizer_step == 1
    assert cold_status.publication_step == 1
    assert console.state.phase is TrainingPhase.COLD_START

    _add_selfplay(catalog_path, store_root)
    console.clock = OneStepClock()
    console.train(warm_start=True)
    mixture_status = console.status()
    assert mixture_status.optimizer_step == 2
    assert mixture_status.publication_step == 2
    assert mixture_status.pending_games == 0
    assert mixture_status.mixture_manifest_sha256
    assert console.state.phase is TrainingPhase.MIXTURE
    assert console.state.migrations[-1].reason == "warm_start"

    restarted = TrainingConsole(settings, clock=OneStepClock(), output=lambda _line: None)
    restarted.reconcile()
    assert restarted.state.phase is TrainingPhase.MIXTURE
    assert restarted.status().optimizer_step == 2


def test_collection_soft_stops_after_one_complete_actor_round(tmp_path: Path) -> None:
    catalog_path, store_root, cold_snapshot = _cold_data(tmp_path)
    settings = ConsoleConfig(
        schema_version=1,
        experiment_config=_experiment_config(tmp_path, fast_selfplay=True),
        catalog_path=catalog_path,
        store_root=store_root,
        cold_start_snapshot_id=cold_snapshot,
        max_runtime_hours=1.0,
    )
    console = TrainingConsole(settings, clock=OneStepClock(), output=lambda _line: None)
    console.reconcile()
    console.train()
    console.clock = OneStepClock()
    console.collect()

    status = console.status()
    assert status.selfplay.sealed_tasks == 1
    assert status.selfplay.collecting_tasks == 0
    assert status.selfplay.failed_tasks == 0
    assert status.selfplay.games == console.config.selfplay.actor_count
    assert status.pending_games == console.config.selfplay.actor_count
    assert console.state.next_collection_round == 1


def test_selfplay_training_visibility_requires_sealed_tasks(tmp_path: Path) -> None:
    catalog_path, store_root, _cold_snapshot = _cold_data(tmp_path)
    _add_selfplay(catalog_path, store_root)
    _add_selfplay(
        catalog_path,
        store_root,
        task_id="1" * 64,
        manifest_sha="2" * 64,
        game_id="3" * 64,
        status="collecting",
    )
    _add_selfplay(
        catalog_path,
        store_root,
        task_id="4" * 64,
        manifest_sha="5" * 64,
        game_id="6" * 64,
        status="failed",
    )

    with Catalog(catalog_path, ShardStore(store_root)) as catalog:
        statistics = catalog.selfplay_statistics()
        assert statistics.sealed_tasks == 1
        assert statistics.collecting_tasks == 1
        assert statistics.failed_tasks == 1
        assert statistics.games == 1
        assert statistics.positions == 4
        assert catalog.selfplay_outside_snapshot(None) == (1, 4)
        snapshot = catalog.create_snapshot(
            seed=7,
            validation_fraction=0.0,
            source_kind="selfplay",
        )
        snapshot_statistics = catalog.snapshot_statistics(snapshot)
        assert snapshot_statistics.games == 1
        assert snapshot_statistics.positions == 4
        assert catalog.selfplay_outside_snapshot(snapshot) == (0, 0)


def test_reconcile_replaces_same_step_publication_from_another_run(
    tmp_path: Path,
) -> None:
    console = _console(tmp_path)
    identity = _cold_identity(console)
    first = Learner(
        console.config,
        console.manager,
        data_identity=identity,
        run_id="run-a",
    )
    first.publish()
    second = Learner(
        console.config,
        console.manager,
        data_identity=identity,
        run_id="run-b",
    )
    second.save_checkpoint()

    console.reconcile()

    publication = console.manager.current_publication()
    assert publication is not None
    payload = console.manager.load_publication(publication)
    assert payload["run_id"] == "run-b"
    assert console.status().artifact_consistency == ("checkpoint and publication aligned")


def test_publication_only_reconcile_accepts_matching_config(tmp_path: Path) -> None:
    console = _console(tmp_path)
    identity = _cold_identity(console)
    learner = Learner(
        console.config,
        console.manager,
        data_identity=identity,
        run_id="publication-only",
    )
    publication = learner.publish()

    console.reconcile()

    assert console.state.operation is Operation.READY
    status = console.status()
    assert status.checkpoint_path is None
    assert status.publication_path == publication
    assert status.artifact_consistency == ("publication exists without resumable checkpoint")


@pytest.mark.parametrize(
    ("publication_step", "publication_samples", "message"),
    (
        (1, 4, "ahead"),
        (0, 4, "conflicts"),
    ),
)
def test_reconcile_rejects_same_run_publication_conflicts(
    tmp_path: Path,
    publication_step: int,
    publication_samples: int,
    message: str,
) -> None:
    console = _console(tmp_path)
    identity = _cold_identity(console)
    learner = Learner(
        console.config,
        console.manager,
        data_identity=identity,
        run_id="shared-run",
    )
    learner.save_checkpoint()
    console.manager.save_publication(
        "shared-run",
        publication_step,
        publication_samples,
        learner.slow.state_dict(),
        checkpoint_metadata(
            console.config.canonical_json(),
            console.config.sha256,
        ),
    )

    with pytest.raises(ValueError, match=message):
        console.reconcile()
    assert console.state.operation is Operation.FAILED


def test_publication_only_reconcile_requires_matching_config(tmp_path: Path) -> None:
    console = _console(tmp_path)
    identity = _cold_identity(console)
    learner = Learner(
        console.config,
        console.manager,
        data_identity=identity,
        run_id="publication-only",
    )
    alternative = dataclasses.replace(console.config, run_name="other-experiment")
    console.manager.save_publication(
        "publication-only",
        0,
        0,
        learner.slow.state_dict(),
        checkpoint_metadata(
            alternative.canonical_json(),
            alternative.sha256,
        ),
    )

    with pytest.raises(ValueError, match="configured experiment"):
        console.reconcile()
    assert console.state.operation is Operation.FAILED


def test_reconcile_repairs_publication_catalog_and_checkpoint_state(
    tmp_path: Path,
) -> None:
    console = _console(tmp_path)
    identity = _cold_identity(console)
    learner = Learner(
        console.config,
        console.manager,
        data_identity=identity,
        run_id="repair-run",
    )
    rng = np.random.default_rng(31)
    learner.train_optimizer_step(SyntheticBatchSource(), rng)
    learner.save_checkpoint(rng)
    publication = learner.publish()

    console.reconcile()

    with Catalog(
        console.settings.catalog_path,
        ShardStore(console.settings.store_root),
    ) as catalog:
        row = catalog.connection.execute(
            "SELECT run_id,optimizer_step,samples_seen,relative_path "
            "FROM publications WHERE run_id='repair-run'"
        ).fetchone()
    assert row is not None
    assert dict(row) == {
        "run_id": "repair-run",
        "optimizer_step": 1,
        "samples_seen": 4,
        "relative_path": publication.relative_to(console.run_dir).as_posix(),
    }
    checkpoint = console.manager.latest_checkpoint()
    assert checkpoint is not None
    state = console.manager.load(checkpoint)["learner_state"]
    assert state["last_published_step"] == 1
    assert state["last_published_samples"] == 4
    assert state["next_publish_sample"] == 8


def test_training_does_not_republish_a_final_due_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    console = _console(tmp_path)

    class TwoStepClock:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self) -> float:
            self.calls += 1
            return 0.0 if self.calls <= 3 else 10_000.0

    console.clock = TwoStepClock()
    calls = 0
    publish = console.artifacts.publish_learner

    def counted_publish(learner, rng):
        nonlocal calls
        calls += 1
        return publish(learner, rng)

    monkeypatch.setattr(console.artifacts, "publish_learner", counted_publish)
    console.reconcile()
    console.train()

    assert console.status().optimizer_step == 2
    assert calls == 1
