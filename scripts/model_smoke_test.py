"""Run the production model's compiled BF16 batch-16 GPU memory smoke test."""

from __future__ import annotations

import argparse
import copy
import gc
import json
from dataclasses import replace

import torch

from zero_ttt.config import load_config
from zero_ttt.game.rules import ACTION_SIZE, BOARD_SIZE
from zero_ttt.model.transformer import PolicyValueTransformer


MEMORY_LIMIT_BYTES = int(14.5 * 1024**3)


def run_case(config, hypernet_enabled: bool) -> dict[str, float | int | bool]:
    hypernet = replace(config.model.hypernet, enabled=hypernet_enabled)
    model_config = replace(config.model, hypernet=hypernet)
    fast = PolicyValueTransformer(model_config).cuda().train()
    slow = copy.deepcopy(fast).float().eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        fast.parameters(),
        lr=config.training.learning_rate,
        betas=(config.training.beta1, config.training.beta2),
        eps=config.training.eps,
        weight_decay=config.training.weight_decay,
        fused=True,
    )
    if config.runtime.compile_model:
        fast.compile(dynamic=False, mode=config.runtime.compile_mode)
    batch = config.training.batch_size
    board = torch.zeros(
        batch,
        model_config.input_planes,
        BOARD_SIZE,
        BOARD_SIZE,
        device="cuda",
    )
    global_features = torch.zeros(batch, model_config.global_features, device="cuda")
    legal = torch.ones(batch, ACTION_SIZE, dtype=torch.bool, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = fast(board, global_features, legal)
        loss = (
            output.policy_logits.float().mean()
            + output.value.float().mean()
            + output.ownership.float().mean()
            + output.score_margin.float().mean() / 400.0
        )
    if not torch.isfinite(loss):
        raise FloatingPointError("production smoke loss is non-finite")
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_reserved()
    if peak > MEMORY_LIMIT_BYTES:
        raise RuntimeError(
            f"peak reserved memory {peak / 1024**3:.2f} GiB exceeds 14.5 GiB"
        )
    result = {
        "hypernet_enabled": hypernet_enabled,
        "parameters": sum(parameter.numel() for parameter in fast.parameters()),
        "batch_size": batch,
        "peak_reserved_gib": peak / 1024**3,
        "loss": loss.detach().item(),
    }
    del output, loss, optimizer, slow, fast, board, global_features, legal
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rtx4090l.toml")
    arguments = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the production memory smoke test")
    config = load_config(arguments.config)
    if config.runtime.device != "cuda":
        raise ValueError("the smoke-test configuration must select CUDA")
    results = [run_case(config, False), run_case(config, True)]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
