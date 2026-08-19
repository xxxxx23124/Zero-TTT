"""Run compiled BF16 batch-16 GPU memory smokes for production configs."""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import time
from typing import Any

import torch
from torch import nn

from zero_ttt.config import ExperimentConfig, load_config
from zero_ttt.game.rules import ACTION_SIZE, BOARD_AREA, BOARD_SIZE
from zero_ttt.model.transformer import ModelOutput, PolicyValueTransformer
from zero_ttt.training.ema import update_slow_weights
from zero_ttt.training.losses import TrainingTargets, compute_losses
from zero_ttt.training.trainer import parameters_are_finite


MEMORY_LIMIT_BYTES = int(14.5 * 1024**3)
PARAMETER_LIMIT = 630_000_000
EXPECTED_PARAMETERS = {
    "configs/rtx4090l.toml": 625_357_745,
    "configs/rtx4090l_baseline.toml": 620_432_901,
}


def _build_optimizer(
    model: PolicyValueTransformer,
    config: ExperimentConfig,
) -> torch.optim.Optimizer:
    hyper_ids = {id(parameter) for parameter in model.hypernet_parameters()}
    groups: dict[tuple[bool, bool], list[nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        is_hyper = id(parameter) in hyper_ids
        decay = parameter.ndim >= 2 and "norm" not in name and name != "cls_token"
        groups.setdefault((is_hyper, decay), []).append(parameter)
    optimizer_groups: list[dict[str, Any]] = []
    for (is_hyper, decay), parameters in groups.items():
        optimizer_groups.append(
            {
                "params": parameters,
                "weight_decay": config.training.weight_decay if decay else 0.0,
                "lr": 0.0,
                "group_name": "hypernet" if is_hyper else "base",
            }
        )
    return torch.optim.AdamW(
        optimizer_groups,
        lr=0.0,
        betas=(config.training.beta1, config.training.beta2),
        eps=config.training.eps,
        fused=True,
    )


def _set_learning_rate(
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    step: int,
) -> float:
    warmup = config.training.warmup_steps
    if step <= warmup:
        learning_rate = config.training.learning_rate * step / warmup
    else:
        learning_rate = config.training.learning_rate * math.sqrt(warmup / step)
    for group in optimizer.param_groups:
        multiplier = (
            config.model.hypernet.lr_multiplier
            if group["group_name"] == "hypernet"
            else 1.0
        )
        group["lr"] = learning_rate * multiplier
    return learning_rate


def _targets(batch: int, device: torch.device) -> TrainingTargets:
    return TrainingTargets(
        policy=torch.full(
            (batch, ACTION_SIZE),
            1.0 / ACTION_SIZE,
            device=device,
        ),
        value=torch.zeros(batch, device=device),
        ownership=torch.zeros(batch, BOARD_AREA, device=device),
        score_margin=torch.zeros(batch, device=device),
        ownership_mask=torch.ones(batch, device=device),
        score_mask=torch.ones(batch, device=device),
    )


def _loss(
    model: PolicyValueTransformer,
    board: torch.Tensor,
    global_features: torch.Tensor,
    legal: torch.Tensor,
    targets: TrainingTargets,
    config: ExperimentConfig,
) -> tuple[ModelOutput, torch.Tensor]:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(board, global_features, legal)
        loss = compute_losses(output, targets, config.training).total
    return output, loss


def _clip_gradients(
    model: PolicyValueTransformer,
    config: ExperimentConfig,
) -> tuple[float, float | None]:
    hyper_ids = {id(parameter) for parameter in model.hypernet_parameters()}
    hyper_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) in hyper_ids and parameter.grad is not None
    ]
    hyper_norm_value: float | None = None
    if hyper_parameters:
        hyper_norm = torch.nn.utils.clip_grad_norm_(
            hyper_parameters,
            config.model.hypernet.grad_clip,
        )
        if not torch.isfinite(hyper_norm):
            raise FloatingPointError("production smoke hypernetwork gradient is non-finite")
        hyper_norm_value = float(hyper_norm)
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        config.training.gradient_clip,
    )
    if not torch.isfinite(gradient_norm):
        raise FloatingPointError("production smoke gradient is non-finite")
    return float(gradient_norm), hyper_norm_value


def _branch_gradients_are_finite(model: PolicyValueTransformer) -> tuple[bool, bool]:
    hyper_finite = model.hypernet is None or all(
        parameter.grad is not None
        and parameters_are_finite((parameter.grad,))
        for parameter in model.hypernet.parameters()
    )
    dwa_finite = model.depth_mixing is None or all(
        parameter.grad is not None
        and parameters_are_finite((parameter.grad,))
        for parameter in model.depth_mixing.parameters()
    )
    return bool(hyper_finite), bool(dwa_finite)


def run_case(
    config_path: str,
    measured_optimizer_steps: int,
) -> dict[str, float | int | bool | str | None]:
    config = load_config(config_path)
    if config.runtime.device != "cuda" or config.runtime.ema_device != "cpu":
        raise ValueError("the production smoke test requires CUDA fast weights and CPU EMA")
    if measured_optimizer_steps <= 0:
        raise ValueError("measured_optimizer_steps must be positive")

    fast = PolicyValueTransformer(config.model).cuda().train()
    slow = PolicyValueTransformer(config.model).cpu().float().eval().requires_grad_(False)
    slow.load_state_dict(fast.state_dict())
    inference = PolicyValueTransformer(config.model)
    inference.load_state_dict(slow.state_dict())
    inference = inference.cuda().to(dtype=torch.bfloat16).eval().requires_grad_(False)
    optimizer = _build_optimizer(fast, config)
    if config.runtime.compile_model:
        fast.compile_training_blocks(dynamic=False, mode=config.runtime.compile_mode)
        inference.compile(dynamic=False, mode=config.runtime.compile_mode)

    batch = config.training.batch_size
    accumulation = config.training.accumulation_steps
    device = torch.device("cuda")
    board = torch.zeros(
        batch,
        config.model.input_planes,
        BOARD_SIZE,
        BOARD_SIZE,
        device=device,
    )
    global_features = torch.zeros(batch, config.model.global_features, device=device)
    legal = torch.ones(batch, ACTION_SIZE, dtype=torch.bool, device=device)
    targets = _targets(batch, device)

    compile_started = time.perf_counter()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        publication_output = inference(board, global_features, legal)
    if not all(
        torch.isfinite(tensor).all()
        for tensor in (
            publication_output.policy_logits,
            publication_output.value,
            publication_output.ownership,
            publication_output.score_margin,
        )
    ):
        raise FloatingPointError("production publication output is non-finite")
    _set_learning_rate(optimizer, config, step=1)
    optimizer.zero_grad(set_to_none=False)
    _, compile_loss = _loss(fast, board, global_features, legal, targets, config)
    (compile_loss / accumulation).backward()
    _clip_gradients(fast, config)
    optimizer.step()
    if not parameters_are_finite(fast.parameters()):
        raise FloatingPointError("production smoke parameter is non-finite")
    torch.cuda.synchronize()
    compile_seconds = time.perf_counter() - compile_started
    compile_peak_reserved_gib = torch.cuda.max_memory_reserved() / 1024**3
    post_compile_reserved_gib = torch.cuda.memory_reserved() / 1024**3

    effective_batch = batch * accumulation
    ema_pending_samples = effective_batch
    microbatch_seconds: list[float] = []
    optimizer_step_seconds: list[float] = []
    gradient_norms: list[float] = []
    hyper_gradient_norms: list[float] = []
    ema_update_seconds: float | None = None
    ema_natural = False
    output: ModelOutput | None = None
    loss: torch.Tensor | None = None
    memory_peaks: dict[str, float] = {}
    torch.cuda.reset_peak_memory_stats()

    for optimizer_step in range(2, measured_optimizer_steps + 2):
        _set_learning_rate(optimizer, config, optimizer_step)
        optimizer.zero_grad(set_to_none=False)
        step_started = time.perf_counter()
        for _ in range(accumulation):
            torch.cuda.synchronize()
            microbatch_started = time.perf_counter()
            output, loss = _loss(fast, board, global_features, legal, targets, config)
            if not torch.isfinite(loss):
                raise FloatingPointError("production smoke loss is non-finite")
            (loss / accumulation).backward()
            torch.cuda.synchronize()
            microbatch_seconds.append(time.perf_counter() - microbatch_started)
        memory_peaks["accumulation"] = torch.cuda.max_memory_reserved() / 1024**3
        gradient_norm, hyper_gradient_norm = _clip_gradients(fast, config)
        torch.cuda.synchronize()
        memory_peaks["gradient_clip"] = torch.cuda.max_memory_reserved() / 1024**3
        gradient_norms.append(gradient_norm)
        if hyper_gradient_norm is not None:
            hyper_gradient_norms.append(hyper_gradient_norm)
        optimizer.step()
        if not parameters_are_finite(fast.parameters()):
            raise FloatingPointError("production smoke parameter is non-finite")
        torch.cuda.synchronize()
        memory_peaks["optimizer_step"] = torch.cuda.max_memory_reserved() / 1024**3
        optimizer_step_seconds.append(time.perf_counter() - step_started)
        ema_pending_samples += effective_batch
        if optimizer_step % config.training.ema_update_interval == 0:
            ema_started = time.perf_counter()
            update_slow_weights(
                slow,
                fast,
                samples=ema_pending_samples,
                half_life_samples=config.training.ema_half_life_samples,
            )
            ema_update_seconds = time.perf_counter() - ema_started
            ema_pending_samples = 0
            ema_natural = True

    if ema_update_seconds is None:
        ema_started = time.perf_counter()
        update_slow_weights(
            slow,
            fast,
            samples=ema_pending_samples,
            half_life_samples=config.training.ema_half_life_samples,
        )
        ema_update_seconds = time.perf_counter() - ema_started
    memory_peaks["ema"] = torch.cuda.max_memory_reserved() / 1024**3

    if output is None or loss is None:
        raise RuntimeError("production smoke executed no measured optimizer steps")
    hyper_gradients_finite, dwa_gradients_finite = _branch_gradients_are_finite(fast)
    if not hyper_gradients_finite or not dwa_gradients_finite:
        raise FloatingPointError("production smoke branch gradient is missing or non-finite")
    if not parameters_are_finite(fast.parameters()):
        raise FloatingPointError("production smoke parameter is non-finite")

    torch.cuda.synchronize()
    memory_peaks["validation"] = torch.cuda.max_memory_reserved() / 1024**3
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    if peak_reserved > MEMORY_LIMIT_BYTES:
        raise RuntimeError(
            f"peak allocated/reserved memory {peak_allocated / 1024**3:.2f}/"
            f"{peak_reserved / 1024**3:.2f} GiB exceeds 14.5 GiB reserved; "
            f"compile peak/current: {compile_peak_reserved_gib:.3f}/"
            f"{post_compile_reserved_gib:.3f} GiB; phase peaks: {memory_peaks}"
        )
    parameters = sum(parameter.numel() for parameter in fast.parameters())
    expected_parameters = EXPECTED_PARAMETERS.get(config_path)
    if expected_parameters is not None and parameters != expected_parameters:
        raise RuntimeError(
            f"{config_path} has {parameters} parameters, expected {expected_parameters}"
        )
    if config.model.hypernet.enabled and parameters >= PARAMETER_LIMIT:
        raise RuntimeError(f"default model has {parameters} parameters, expected <630M")

    result = {
        "config": config_path,
        "hypernet_enabled": config.model.hypernet.enabled,
        "depth_mixing_enabled": config.model.depth_mixing.enabled,
        "parameters": parameters,
        "batch_size": batch,
        "accumulation_steps": accumulation,
        "measured_optimizer_steps": measured_optimizer_steps,
        "compile_seconds": compile_seconds,
        "compile_peak_reserved_gib": compile_peak_reserved_gib,
        "median_microbatch_seconds": statistics.median(microbatch_seconds),
        "median_optimizer_step_seconds": statistics.median(optimizer_step_seconds),
        "ema_update_seconds": ema_update_seconds,
        "ema_natural": ema_natural,
        "peak_allocated_gib": peak_allocated / 1024**3,
        "peak_reserved_gib": peak_reserved / 1024**3,
        "loss": loss.detach().item(),
        "gradient_norm": statistics.median(gradient_norms),
        "hyper_gradient_norm": (
            statistics.median(hyper_gradient_norms) if hyper_gradient_norms else None
        ),
        "hyper_gradients_finite": hyper_gradients_finite,
        "dwa_gradients_finite": dwa_gradients_finite,
        "hyper_a_saturation": output.hyper_a_saturation.item(),
        "hyper_b_saturation": output.hyper_b_saturation.item(),
        "hyper_dynamic_rms": output.hyper_dynamic_rms.item(),
        "hyper_static_rms": output.hyper_static_rms.item(),
    }
    del output, loss, publication_output, compile_loss, optimizer, inference, slow, fast
    del board, global_features, legal, targets
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["configs/rtx4090l_baseline.toml", "configs/rtx4090l.toml"],
    )
    parser.add_argument("--default-optimizer-steps", type=int, default=16)
    parser.add_argument("--baseline-optimizer-steps", type=int, default=1)
    arguments = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the production memory smoke test")
    results = []
    for path in arguments.configs:
        config = load_config(path)
        measured_steps = (
            arguments.default_optimizer_steps
            if config.model.hypernet.enabled
            else arguments.baseline_optimizer_steps
        )
        results.append(run_case(path, measured_steps))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
