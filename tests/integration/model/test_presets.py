from __future__ import annotations

import torch

from zero_ttt.config import load_config
from zero_ttt.model import PolicyValueTransformer


def test_production_and_baseline_experiment_switches() -> None:
    production = load_config("configs/rtx4090l.toml")
    baseline = load_config("configs/rtx4090l_baseline.toml")
    assert production.run_name == "rtx4090l-625m"
    assert production.runtime.run_dir == "runs/rtx4090l-625m"
    assert production.model.d_model == 1280
    assert production.model.n_heads == 20
    assert production.model.d_model // production.model.n_heads == 64
    assert production.model.n_layers == 32
    assert production.model.d_ff == 3328
    assert production.execution.activation_checkpoint_stride == 1
    assert production.execution.activation_checkpoint
    assert production.training.effective_batch_size == 4096
    assert production.training.accumulation_steps == 256
    assert production.training.learning_rate == 1e-4
    assert production.training.beta2 == 0.98
    assert production.training.weight_decay == 0.03
    assert production.training.warmup_samples == 512_000
    assert production.training.ema_update_interval_samples == 4096
    assert production.training.publish_interval_samples == 65_536
    assert production.model.hypernet.enabled
    assert production.model.hypernet.num_layers == 16
    assert production.model.depth_mixing.enabled
    assert baseline.model.d_model == production.model.d_model
    assert baseline.model.n_heads == production.model.n_heads
    assert baseline.model.n_layers == production.model.n_layers
    assert baseline.model.d_ff == production.model.d_ff
    assert baseline.runtime.run_dir == "runs/rtx4090l-625m-baseline"
    assert not baseline.model.hypernet.enabled
    assert not baseline.model.depth_mixing.enabled

    with torch.device("meta"):
        production_model = PolicyValueTransformer(production.model, production.execution)
        baseline_model = PolicyValueTransformer(baseline.model, baseline.execution)
    assert sum(parameter.numel() for parameter in production_model.parameters()) == 625_357_745
    assert sum(parameter.numel() for parameter in baseline_model.parameters()) == 620_432_901
