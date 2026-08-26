"""File-configured command line entry points."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from zero_ttt.config import load_config
from zero_ttt.data.catalog import Catalog
from zero_ttt.data.catalog_source import CatalogBatchSource
from zero_ttt.data.manifest import SourceManifest
from zero_ttt.data.pipeline import import_manifest
from zero_ttt.data.shards import ShardStore
from zero_ttt.data.synthetic import SyntheticBatchSource
from zero_ttt.game.rules import ACTION_SIZE, BOARD_SIZE
from zero_ttt.learner import Learner, LearnerDataIdentity
from zero_ttt.model import PolicyValueTransformer
from zero_ttt.training.checkpoint import CheckpointManager


def _add_config(child: argparse.ArgumentParser) -> None:
    child.add_argument("--config", required=True, help="versioned TOML experiment file")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="zero-ttt")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("config-check", "model-smoke", "train-smoke"):
        _add_config(subparsers.add_parser(command))

    manifest_create = subparsers.add_parser("manifest-create")
    manifest_create.add_argument("--dataset-id", required=True)
    manifest_create.add_argument("--source-root", required=True)
    manifest_create.add_argument("--glob", default="**/*.zip")
    manifest_create.add_argument("--source-type", default="katago-g170-sgfs-zip")
    manifest_create.add_argument("--license-id", required=True)
    manifest_create.add_argument("--license-url", required=True)
    manifest_create.add_argument("--output", required=True)

    manifest_check = subparsers.add_parser("manifest-check")
    manifest_check.add_argument("--manifest", required=True)
    manifest_check.add_argument("--source-root", required=True)

    data_import = subparsers.add_parser("data-import")
    data_import.add_argument("--manifest", required=True)
    data_import.add_argument("--source-root", required=True)
    data_import.add_argument("--store-root", required=True)
    data_import.add_argument("--catalog", required=True)
    data_import.add_argument(
        "--max-games",
        type=int,
        help="maximum number of new, catalog-deduplicated games to add in this run",
    )
    data_import.add_argument("--target-shard-bytes", type=int, default=128 * 1024 * 1024)

    data_verify = subparsers.add_parser("data-verify")
    data_verify.add_argument("--store-root", required=True)
    data_verify.add_argument("--catalog", required=True)

    snapshot = subparsers.add_parser("snapshot-create")
    snapshot.add_argument("--store-root", required=True)
    snapshot.add_argument("--catalog", required=True)
    snapshot.add_argument("--seed", required=True, type=int)
    snapshot.add_argument("--split", choices=("train", "validation"), default="train")
    snapshot.add_argument("--validation-fraction", type=float, default=0.1)

    offline = subparsers.add_parser("offline-imitation")
    _add_config(offline)
    offline.add_argument("--store-root", required=True)
    offline.add_argument("--catalog", required=True)
    offline.add_argument("--snapshot", required=True)
    offline.add_argument("--steps", required=True, type=int)
    offline.add_argument(
        "--annotation-mode",
        choices=("none", "prefer_exact", "require_exact"),
        default="none",
    )
    offline.add_argument("--teacher-fingerprint")
    offline.add_argument("--resume", nargs="?", const="latest")
    offline.add_argument("--run-id")
    return parser


def _model_smoke(config_path: str) -> None:
    config = load_config(config_path)
    device = torch.device(config.runtime.device)
    model = PolicyValueTransformer(config.model, config.execution).to(device).train()
    batch = config.training.batch_size
    board = torch.zeros(batch, config.model.input_planes, BOARD_SIZE, BOARD_SIZE, device=device)
    globals_ = torch.zeros(batch, config.model.global_features, device=device)
    legal = torch.ones(batch, ACTION_SIZE, dtype=torch.bool, device=device)
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else torch.autocast(device_type="cpu", enabled=False)
    )
    with autocast:
        output = model(board, globals_, legal)
        loss = output.policy_logits.float().mean() + output.value.float().mean()
    loss.backward()
    print(json.dumps({"parameters": sum(p.numel() for p in model.parameters()), "loss": loss.item()}))


def _config_check(config_path: str) -> None:
    config = load_config(config_path)
    print(
        json.dumps(
            {
                "schema_version": config.schema_version,
                "run_name": config.run_name,
                "config_sha256": config.sha256,
                "effective_batch_size": config.training.effective_batch_size,
            }
        )
    )


def _train_smoke(config_path: str) -> None:
    config = load_config(config_path)
    rng = np.random.default_rng(config.seed)
    with tempfile.TemporaryDirectory(prefix="zero-ttt-train-smoke-") as run_dir:
        manager = CheckpointManager(run_dir, keep=1)
        learner = Learner(config, manager)
        metrics = learner.train_optimizer_step(SyntheticBatchSource(), rng)
        checkpoint = learner.save_checkpoint(rng)
        publication = learner.publish()
        learner.restore(checkpoint, rng)
        print(
            json.dumps(
                {
                    "optimizer_step": metrics.step,
                    "total_loss": metrics.total_loss,
                    "checkpoint_round_trip": True,
                    "publication": publication.as_posix(),
                }
            )
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _offline_imitation(arguments: argparse.Namespace) -> None:
    if arguments.steps < 0:
        raise ValueError("steps cannot be negative")
    config = load_config(arguments.config)
    rng = np.random.default_rng(config.seed)
    run_dir = config.run_dir
    manager = CheckpointManager(run_dir, keep=config.training.checkpoint_keep)
    with CatalogBatchSource(
        arguments.catalog,
        arguments.store_root,
        arguments.snapshot,
        annotation_mode=arguments.annotation_mode,
        teacher_fingerprint=arguments.teacher_fingerprint,
    ) as source:
        identity = LearnerDataIdentity(
            snapshot_id=arguments.snapshot,
            sampling_config_sha256=source.sampling_config_sha256,
        )
        learner = Learner(
            config,
            manager,
            data_identity=identity,
            run_id=arguments.run_id,
        )
        if arguments.resume:
            checkpoint = (
                manager.latest_checkpoint()
                if arguments.resume == "latest"
                else Path(arguments.resume)
            )
            if checkpoint is None:
                raise FileNotFoundError("no checkpoint is available to resume")
            learner.restore(checkpoint, rng)
        metrics = []
        with Catalog(arguments.catalog, ShardStore(arguments.store_root)) as catalog:
            for _ in range(arguments.steps):
                metric = learner.train_optimizer_step(source, rng)
                metrics.append(asdict(metric))
                publication = learner.publish_if_due()
                if publication is not None:
                    relative = publication.relative_to(run_dir).as_posix()
                    catalog.record_publication(
                        learner.state.run_id,
                        learner.state.optimizer_step,
                        learner.state.samples_seen,
                        relative,
                        _sha256(publication),
                    )
                    learner.save_checkpoint(rng)
        checkpoint = learner.save_checkpoint(rng)
        print(
            json.dumps(
                {
                    "run_id": learner.state.run_id,
                    "optimizer_step": learner.state.optimizer_step,
                    "samples_seen": learner.state.samples_seen,
                    "checkpoint": checkpoint.as_posix(),
                    "last_metrics": metrics[-1] if metrics else None,
                }
            )
        )


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "config-check":
        _config_check(arguments.config)
    elif arguments.command == "model-smoke":
        _model_smoke(arguments.config)
    elif arguments.command == "train-smoke":
        _train_smoke(arguments.config)
    elif arguments.command == "manifest-create":
        manifest = SourceManifest.create(
            arguments.dataset_id,
            arguments.source_type,
            arguments.license_id,
            arguments.license_url,
            arguments.source_root,
            arguments.glob,
        )
        manifest.save(arguments.output)
        print(json.dumps({"manifest": arguments.output, "assets": len(manifest.assets)}))
    elif arguments.command == "manifest-check":
        manifest = SourceManifest.load(arguments.manifest)
        manifest.verify(arguments.source_root)
        print(json.dumps({"dataset_id": manifest.dataset_id, "assets": len(manifest.assets)}))
    elif arguments.command == "data-import":
        summary = import_manifest(
            arguments.manifest,
            arguments.source_root,
            arguments.store_root,
            arguments.catalog,
            arguments.max_games,
            arguments.target_shard_bytes,
        )
        print(json.dumps(asdict(summary)))
    elif arguments.command == "data-verify":
        with Catalog(arguments.catalog, ShardStore(arguments.store_root)) as catalog:
            orphans = catalog.recover()
        print(json.dumps({"verified": True, "orphans": orphans}))
    elif arguments.command == "snapshot-create":
        with Catalog(arguments.catalog, ShardStore(arguments.store_root)) as catalog:
            snapshot_id = catalog.create_snapshot(
                arguments.seed,
                arguments.split,
                arguments.validation_fraction,
            )
        print(json.dumps({"snapshot_id": snapshot_id, "split": arguments.split}))
    else:
        _offline_imitation(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
