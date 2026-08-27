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
    snapshot.add_argument("--source-kind", choices=("external", "selfplay"))
    snapshot.add_argument("--task-id")

    mixture = subparsers.add_parser("mixture-create")
    mixture.add_argument(
        "--component",
        action="append",
        required=True,
        help="snapshot SHA-256 and weight as SNAPSHOT=WEIGHT",
    )
    mixture.add_argument("--output", required=True)

    selfplay = subparsers.add_parser("selfplay-collect")
    _add_config(selfplay)
    selfplay.add_argument("--publication", required=True)
    selfplay.add_argument("--store-root", required=True)
    selfplay.add_argument("--catalog", required=True)
    selfplay.add_argument("--games", required=True, type=int)
    selfplay.add_argument("--seed", type=int)
    selfplay.add_argument("--target-shard-bytes", type=int, default=128 * 1024 * 1024)

    offline = subparsers.add_parser("offline-imitation")
    _add_config(offline)
    offline.add_argument("--store-root", required=True)
    offline.add_argument("--catalog", required=True)
    offline_data = offline.add_mutually_exclusive_group(required=True)
    offline_data.add_argument("--snapshot")
    offline_data.add_argument("--mixture")
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
    if arguments.mixture:
        if arguments.annotation_mode != "none" or arguments.teacher_fingerprint:
            raise ValueError("mixture training does not yet support teacher annotation modes")
        from zero_ttt.data.mixture import MixtureBatchSource, TrainingMixtureManifest

        mixture = TrainingMixtureManifest.load(arguments.mixture)
        source_context = MixtureBatchSource(
            arguments.catalog,
            arguments.store_root,
            mixture,
        )
        identity = LearnerDataIdentity(
            snapshot_id=f"mixture:{mixture.content_sha256}",
            sampling_config_sha256=source_context.sampling_config_sha256,
            mixture_manifest_sha256=mixture.content_sha256,
            component_snapshot_ids=source_context.component_snapshot_ids,
        )
    else:
        source_context = CatalogBatchSource(
            arguments.catalog,
            arguments.store_root,
            arguments.snapshot,
            annotation_mode=arguments.annotation_mode,
            teacher_fingerprint=arguments.teacher_fingerprint,
        )
        identity = LearnerDataIdentity(
            snapshot_id=arguments.snapshot,
            sampling_config_sha256=source_context.sampling_config_sha256,
        )
    with source_context as source:
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


def _mixture_create(arguments: argparse.Namespace) -> None:
    from zero_ttt.data.mixture import MixtureComponent, TrainingMixtureManifest

    components = []
    for value in arguments.component:
        snapshot_id, separator, raw_weight = value.rpartition("=")
        if not separator or not snapshot_id or not raw_weight:
            raise ValueError("mixture components must use SNAPSHOT=WEIGHT")
        try:
            weight = float(raw_weight)
        except ValueError as error:
            raise ValueError("mixture component weight must be numeric") from error
        components.append(MixtureComponent(snapshot_id, weight))
    manifest = TrainingMixtureManifest(1, tuple(components))
    manifest.save(arguments.output)
    print(
        json.dumps(
            {
                "mixture": arguments.output,
                "content_sha256": manifest.content_sha256,
                "components": len(components),
            }
        )
    )


def _selfplay_collect(arguments: argparse.Namespace) -> None:
    from zero_ttt.game.features import FEATURE_SCHEMA_ID
    from zero_ttt.game.rules import RULES_ID
    from zero_ttt.inference import BatchedInferenceBroker, PublicationPositionEvaluator
    from zero_ttt.selfplay.collector import SelfPlayCollector, search_config_sha256

    config = load_config(arguments.config)
    evaluator = PublicationPositionEvaluator(
        arguments.publication,
        device=config.runtime.device,
        inference_batch_size=config.selfplay.inference_batch_size,
        compile_model=config.selfplay.compile_inference,
        compile_mode=config.execution.compile_mode,
    )
    if evaluator.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(evaluator.device)
    search_hash = search_config_sha256(config)
    evaluator_id = hashlib.sha256(
        json.dumps(
            [evaluator.model_version, FEATURE_SCHEMA_ID, RULES_ID, search_hash],
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    with BatchedInferenceBroker(
        evaluator,
        batch_size=config.selfplay.inference_batch_size,
        batch_wait_ms=config.selfplay.batch_wait_ms,
        cache_size=config.selfplay.inference_cache_size,
    ) as broker:
        summary = SelfPlayCollector(
            config,
            broker,
            publication_sha256=evaluator.publication_sha256,
            evaluator_id=evaluator_id,
            store_root=arguments.store_root,
            catalog_path=arguments.catalog,
            games=arguments.games,
            seed=config.seed if arguments.seed is None else arguments.seed,
            target_shard_bytes=arguments.target_shard_bytes,
        ).collect()
    payload = asdict(summary)
    if evaluator.device.type == "cuda":
        torch.cuda.synchronize(evaluator.device)
        payload["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(
            evaluator.device
        )
    else:
        payload["gpu_peak_allocated_bytes"] = 0
    print(json.dumps(payload))


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
                arguments.source_kind,
                arguments.task_id,
            )
        print(json.dumps({"snapshot_id": snapshot_id, "split": arguments.split}))
    elif arguments.command == "mixture-create":
        _mixture_create(arguments)
    elif arguments.command == "selfplay-collect":
        _selfplay_collect(arguments)
    else:
        _offline_imitation(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
