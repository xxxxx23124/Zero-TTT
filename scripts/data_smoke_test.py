"""Real-data vertical smoke test; intentionally capped at 64 accepted games."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

from zero_ttt.config import load_config
from zero_ttt.data.catalog import Catalog
from zero_ttt.data.catalog_source import CatalogBatchSource
from zero_ttt.data.manifest import ManifestAsset, SourceManifest, sha256_file
from zero_ttt.data.pipeline import import_manifest
from zero_ttt.data.shards import ShardStore
from zero_ttt.learner import Learner, LearnerDataIdentity
from zero_ttt.training.checkpoint import CheckpointManager


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="/datasets/zero-ttt")
    parser.add_argument("--config", default="configs/test.toml")
    arguments = parser.parse_args()
    dataset_root = Path(arguments.dataset_root).resolve()
    archives = sorted((dataset_root / "raw" / "katago" / "g170" / "selfplay").glob("*.zip"))
    if not archives:
        raise FileNotFoundError("no KataGo g170 zip archive was found")
    archive = archives[0]
    asset = ManifestAsset(
        relative_path=archive.relative_to(dataset_root).as_posix(),
        sha256=sha256_file(archive),
        size_bytes=archive.stat().st_size,
    )
    manifest = SourceManifest(
        schema_version=1,
        dataset_id="katago-g170-smoke",
        source_type="katago-g170-sgfs-zip",
        license_id="CC0-1.0",
        license_url="https://katagoarchive.org/g170/LICENSE.txt",
        assets=(asset,),
    )
    config = load_config(arguments.config)
    rng = np.random.default_rng(config.seed)
    with tempfile.TemporaryDirectory(prefix="zero-ttt-data-smoke-") as temporary:
        work = Path(temporary)
        manifest_path = work / "manifest.json"
        store_root = work / "processed"
        catalog_path = work / "catalog.sqlite"
        manifest.save(manifest_path)
        summary = import_manifest(
            manifest_path,
            dataset_root,
            store_root,
            catalog_path,
            max_accepted=64,
            target_shard_bytes=4 * 1024 * 1024,
        )
        if summary.accepted != 64:
            raise AssertionError(f"expected 64 accepted games, got {summary.accepted}")
        store = ShardStore(store_root)
        with Catalog(catalog_path, store) as catalog:
            catalog.verify()
            train_snapshot = catalog.create_snapshot(
                seed=config.seed,
                split="train",
                validation_fraction=0.1,
            )
            validation_snapshot = catalog.create_snapshot(
                seed=config.seed,
                split="validation",
                validation_fraction=0.1,
            )
            validation_games = len(catalog.snapshot_trajectories(validation_snapshot))
        with CatalogBatchSource(catalog_path, store_root, train_snapshot) as source:
            identity = LearnerDataIdentity(train_snapshot, source.sampling_config_sha256)
            manager = CheckpointManager(work / "run", keep=1)
            learner = Learner(config, manager, data_identity=identity, run_id="g170-smoke")
            metrics = learner.train_optimizer_step(source, rng)
            publication = learner.publish()
            checkpoint = learner.save_checkpoint(rng)
            learner.restore(checkpoint, rng)
        print(
            json.dumps(
                {
                    "archive": asset.relative_path,
                    "import": asdict(summary),
                    "train_snapshot": train_snapshot,
                    "validation_games": validation_games,
                    "optimizer_step": metrics.step,
                    "checkpoint_round_trip": True,
                    "publication": publication.name,
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
