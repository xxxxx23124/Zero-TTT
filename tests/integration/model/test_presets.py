from __future__ import annotations

import torch
from zero_ttt.config import load_config
from zero_ttt.model import PolicyValueTransformer


def test_production_and_baseline_experiment_switches() -> None:
    production = load_config("configs/profiles/rtx4090l.toml")
    baseline = load_config("configs/profiles/rtx4090l_baseline.toml")
    future = load_config("configs/profiles/rtx4090l_625m_future.toml")
    future_baseline = load_config("configs/profiles/rtx4090l_625m_future_baseline.toml")

    assert production.model.d_model == 512
    assert production.model.n_heads == 8
    assert production.model.d_model // production.model.n_heads == 64
    assert production.model.n_layers == 12
    assert production.model.d_ff == 1536
    assert production.execution.activation_checkpoint_stride == 1
    assert not production.execution.activation_checkpoint
    assert production.execution.compile_model
    assert production.training.effective_batch_size == 4096
    assert production.training.batch_size == 64
    assert production.training.accumulation_steps == 64
    assert production.training.learning_rate == 1e-4
    assert production.training.beta2 == 0.98
    assert production.training.weight_decay == 0.03
    assert production.training.warmup_samples == 512_000
    assert production.training.ema_update_interval_samples == 4096
    assert production.training.publish_interval_samples == 65_536
    assert production.model.hypernet.enabled
    assert production.model.hypernet.num_layers == 6
    assert production.model.depth_mixing.enabled
    assert production.selfplay.actor_count == 64
    assert production.selfplay.inference_batch_size == 64
    assert production.selfplay.compile_inference
    assert production.runtime.ema_device == "cpu"
    assert baseline.model.d_model == production.model.d_model
    assert baseline.model.n_heads == production.model.n_heads
    assert baseline.model.n_layers == production.model.n_layers
    assert baseline.model.d_ff == production.model.d_ff
    assert baseline.training == production.training
    assert baseline.execution == production.execution
    assert baseline.selfplay == production.selfplay
    assert production.training.mixture.selfplay_weight == 0.8
    assert production.training.mixture.cold_start_weight == 0.2
    assert not baseline.model.hypernet.enabled
    assert not baseline.model.depth_mixing.enabled

    assert future.model.d_model == 1280
    assert future.model.n_heads == 20
    assert future.model.d_model // future.model.n_heads == 64
    assert future.model.n_layers == 32
    assert future.model.d_ff == 3328
    assert future.model.hypernet.enabled
    assert future.model.hypernet.num_layers == 16
    assert future.model.depth_mixing.enabled
    assert future.training.batch_size == 16
    assert future.training.accumulation_steps == 256
    assert future.training.effective_batch_size == 4096
    assert future.execution.activation_checkpoint
    assert future.execution.compile_model
    assert future.selfplay.actor_count == 64
    assert future.selfplay.inference_batch_size == 64
    assert future_baseline.model.d_model == future.model.d_model
    assert future_baseline.model.n_heads == future.model.n_heads
    assert future_baseline.model.n_layers == future.model.n_layers
    assert future_baseline.model.d_ff == future.model.d_ff
    assert future_baseline.training == future.training
    assert future_baseline.execution == future.execution
    assert future_baseline.selfplay == future.selfplay
    assert not future_baseline.model.hypernet.enabled
    assert not future_baseline.model.depth_mixing.enabled

    with torch.device("meta"):
        production_model = PolicyValueTransformer(production.model, production.execution)
        baseline_model = PolicyValueTransformer(baseline.model, baseline.execution)
        future_model = PolicyValueTransformer(future.model, future.execution)
        future_baseline_model = PolicyValueTransformer(
            future_baseline.model,
            future_baseline.execution,
        )
    assert sum(parameter.numel() for parameter in production_model.parameters()) == 43_371_150
    assert sum(parameter.numel() for parameter in baseline_model.parameters()) == 41_189_893
    assert sum(parameter.numel() for parameter in future_model.parameters()) == 625_357_745
    assert sum(parameter.numel() for parameter in future_baseline_model.parameters()) == 620_432_901
