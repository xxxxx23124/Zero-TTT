"""File-configured command line entry points."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import torch

from zero_ttt.config import load_config
from zero_ttt.game.rules import ACTION_SIZE, BOARD_SIZE
from zero_ttt.model.transformer import PolicyValueTransformer
from zero_ttt.selfplay.loop import CoreLoop


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="zero-ttt")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("loop", "selfplay", "train", "smoke"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", required=True, help="versioned TOML experiment file")
    return parser


def _smoke(config_path: str) -> None:
    config = load_config(config_path)
    device = torch.device(config.runtime.device)
    model = PolicyValueTransformer(config.model).to(device).train()
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


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "smoke":
        _smoke(arguments.config)
        return 0
    config = load_config(arguments.config)
    with CoreLoop(config) as loop:
        try:
            if arguments.command == "selfplay":
                games, positions = loop.selfplay_phase()
                loop.save()
                print(json.dumps({"games": games, "positions": positions}))
            elif arguments.command == "train":
                metrics = loop.train_replay_once()
                loop.save()
                print(json.dumps({"optimizer_steps": len(metrics)}))
            else:
                while True:
                    print(json.dumps(asdict(loop.run_cycle())))
        except KeyboardInterrupt:
            loop.save()
            return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
