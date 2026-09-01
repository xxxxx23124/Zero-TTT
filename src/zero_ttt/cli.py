"""File-configured command line entry points."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict

import numpy as np
import torch

from zero_ttt.config import load_config
from zero_ttt.data.catalog import Catalog
from zero_ttt.data.catalog_source import CatalogBatchSource
from zero_ttt.data.manifest import SourceManifest
from zero_ttt.data.mixture import MixtureBatchSource, TrainingMixtureManifest
from zero_ttt.data.pipeline import import_manifest
from zero_ttt.data.shards import ShardStore
from zero_ttt.data.synthetic import SyntheticBatchSource
from zero_ttt.game.rules import ACTION_SIZE, BOARD_SIZE
from zero_ttt.learner import Learner, LearnerDataIdentity
from zero_ttt.model import PolicyValueTransformer
from zero_ttt.precision import configure_strict_fp32
from zero_ttt.training.artifacts import ArtifactCoordinator
from zero_ttt.training.checkpoint import CheckpointManager
from zero_ttt.training.session import TrainingSession


def _add_config(child: argparse.ArgumentParser) -> None:
    child.add_argument("--config", required=True, help="versioned TOML experiment file")


def _add_manifest_commands(subparsers) -> None:
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


def _add_data_commands(subparsers) -> None:
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


def _add_mixture_command(subparsers) -> None:
    mixture = subparsers.add_parser("mixture-create")
    mixture.add_argument(
        "--component",
        action="append",
        required=True,
        help="snapshot SHA-256 and weight as SNAPSHOT=WEIGHT",
    )
    mixture.add_argument("--output", required=True)


def _add_selfplay_command(subparsers) -> None:
    selfplay = subparsers.add_parser("selfplay-collect")
    _add_config(selfplay)
    selfplay.add_argument("--publication", required=True)
    selfplay.add_argument("--store-root", required=True)
    selfplay.add_argument("--catalog", required=True)
    selfplay.add_argument("--games", required=True, type=int)
    selfplay.add_argument("--seed", type=int)
    selfplay.add_argument("--target-shard-bytes", type=int, default=128 * 1024 * 1024)


def _add_offline_command(subparsers) -> None:
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


def _add_console_command(subparsers) -> None:
    console = subparsers.add_parser("console", help="Docker training console")
    console.add_argument("--config", default="configs/console.toml", help="console TOML file")
    actions = console.add_subparsers(dest="console_action")
    status = actions.add_parser("status", help="inspect current console status")
    status.add_argument("--json", action="store_true")
    for name in ("reconcile", "train", "collect", "warm-start"):
        child = actions.add_parser(name)
        child.add_argument("--events", choices=("none", "jsonl"), default="none")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="zero-ttt")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_console_command(subparsers)
    for command in ("config-check", "model-smoke", "train-smoke"):
        _add_config(subparsers.add_parser(command))
    _add_manifest_commands(subparsers)
    _add_data_commands(subparsers)
    _add_mixture_command(subparsers)
    _add_selfplay_command(subparsers)
    _add_offline_command(subparsers)
    return parser


def _model_smoke(config_path: str) -> None:
    configure_strict_fp32()
    config = load_config(config_path)
    device = torch.device(config.runtime.device)
    model = PolicyValueTransformer(config.model, config.execution).to(device).train()
    batch = config.training.batch_size
    board = torch.zeros(batch, config.model.input_planes, BOARD_SIZE, BOARD_SIZE, device=device)
    globals_ = torch.zeros(batch, config.model.global_features, device=device)
    legal = torch.ones(batch, ACTION_SIZE, dtype=torch.bool, device=device)
    output = model(board, globals_, legal)
    loss = output.policy_logits.mean() + output.value.mean()
    loss.backward()
    print(
        json.dumps({"parameters": sum(p.numel() for p in model.parameters()), "loss": loss.item()})
    )


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


def _offline_source(
    arguments: argparse.Namespace,
) -> tuple[CatalogBatchSource | MixtureBatchSource, LearnerDataIdentity]:
    if arguments.mixture:
        if arguments.annotation_mode != "none" or arguments.teacher_fingerprint:
            raise ValueError("mixture training does not yet support teacher annotation modes")
        mixture = TrainingMixtureManifest.load(arguments.mixture)
        source = MixtureBatchSource(
            arguments.catalog,
            arguments.store_root,
            mixture,
        )
        return source, LearnerDataIdentity(
            snapshot_id=f"mixture:{mixture.content_sha256}",
            sampling_config_sha256=source.sampling_config_sha256,
            mixture_manifest_sha256=mixture.content_sha256,
            component_snapshot_ids=source.component_snapshot_ids,
        )
    source = CatalogBatchSource(
        arguments.catalog,
        arguments.store_root,
        arguments.snapshot,
        annotation_mode=arguments.annotation_mode,
        teacher_fingerprint=arguments.teacher_fingerprint,
    )
    return (
        source,
        LearnerDataIdentity(
            snapshot_id=arguments.snapshot,
            sampling_config_sha256=source.sampling_config_sha256,
        ),
    )


def _offline_imitation(arguments: argparse.Namespace) -> None:
    if arguments.steps < 0:
        raise ValueError("steps cannot be negative")
    config = load_config(arguments.config)
    rng = np.random.default_rng(config.seed)
    run_dir = config.run_dir
    manager = CheckpointManager(run_dir, keep=config.training.checkpoint_keep)
    source_context, identity = _offline_source(arguments)
    with source_context as source:
        artifacts = ArtifactCoordinator(
            config,
            manager,
            run_dir=run_dir,
            catalog_path=arguments.catalog,
            store_root=arguments.store_root,
        )
        session = TrainingSession(
            config,
            manager,
            data_identity=identity,
            run_id=arguments.run_id,
            artifacts=artifacts,
            rng=rng,
        )
        if arguments.resume:
            session.restore_requested(arguments.resume)
        metrics = []
        for _ in range(arguments.steps):
            metrics.append(asdict(session.step(source)))
            session.publish_if_due()
        checkpoint = session.save_checkpoint()
        learner = session.learner
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
    from zero_ttt.versioning import TRAINING_MIXTURE_SCHEMA

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
    manifest = TrainingMixtureManifest(
        TRAINING_MIXTURE_SCHEMA.current,
        tuple(components),
    )
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
    from zero_ttt.selfplay.service import SelfPlayService

    config = load_config(arguments.config)
    with SelfPlayService(
        config,
        arguments.publication,
        store_root=arguments.store_root,
        catalog_path=arguments.catalog,
    ) as service:
        summary = service.collect(
            games=arguments.games,
            seed=config.seed if arguments.seed is None else arguments.seed,
            target_shard_bytes=arguments.target_shard_bytes,
        )
        gpu_peak = service.gpu_peak_allocated_bytes()
    payload = asdict(summary)
    payload["gpu_peak_allocated_bytes"] = gpu_peak
    print(json.dumps(payload))


def _console_sinks(arguments: argparse.Namespace, run_dir):
    from zero_ttt.console.events import CompositeEventSink, ConsoleEventSink, JsonLineEventSink
    from zero_ttt.observability import TensorBoardEventSink

    sinks: list[ConsoleEventSink] = [TensorBoardEventSink(run_dir)]
    if getattr(arguments, "events", "none") == "jsonl":
        sinks.insert(0, JsonLineEventSink(lambda line: print(line, flush=True)))
    return CompositeEventSink(sinks)


def _console_command(arguments: argparse.Namespace) -> int:
    from zero_ttt.console import TrainingConsole, load_console_config
    from zero_ttt.console.status import status_payload

    settings = load_console_config(arguments.config)
    if arguments.console_action is None:
        return TrainingConsole(settings).run_interactive()
    console = TrainingConsole(settings)
    if arguments.console_action == "status":
        status = console.status()
        print(json.dumps(status_payload(status)) if arguments.json else status)
        return 0
    events = _console_sinks(arguments, console.run_dir)
    console.events = events
    try:
        with console.lock:
            console.reconcile()
            if arguments.console_action == "train":
                console.train()
            elif arguments.console_action == "collect":
                console.collect()
            elif arguments.console_action == "warm-start":
                console.train(warm_start=True)
    finally:
        events.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "console":
        return _console_command(arguments)
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
