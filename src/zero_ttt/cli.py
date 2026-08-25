"""File-configured command line entry points."""

from __future__ import annotations

import argparse
import json
import tempfile

import numpy as np
import torch

from zero_ttt.config import load_config
from zero_ttt.data.synthetic import SyntheticBatchSource
from zero_ttt.game.rules import ACTION_SIZE, BOARD_SIZE
from zero_ttt.model import PolicyValueTransformer
from zero_ttt.training.checkpoint import CheckpointManager
from zero_ttt.training.trainer import Trainer


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="zero-ttt")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("config-check", "model-smoke", "train-smoke"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", required=True, help="versioned TOML experiment file")
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
            }
        )
    )


def _train_smoke(config_path: str) -> None:
    config = load_config(config_path)
    rng = np.random.default_rng(config.seed)
    with tempfile.TemporaryDirectory(prefix="zero-ttt-train-smoke-") as run_dir:
        manager = CheckpointManager(run_dir, keep=1)
        trainer = Trainer(config, manager)
        metrics = trainer.train_optimizer_step(SyntheticBatchSource(), rng)
        checkpoint = trainer.save_checkpoint(rng)
        publication = trainer.publish()
        trainer.restore(checkpoint, rng)
        print(
            json.dumps(
                {
                    "optimizer_step": metrics.step,
                    "total_loss": metrics.total_loss,
                    "checkpoint_round_trip": True,
                    "publication": publication.name,
                }
            )
        )


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "config-check":
        _config_check(arguments.config)
    elif arguments.command == "model-smoke":
        _model_smoke(arguments.config)
    else:
        _train_smoke(arguments.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
