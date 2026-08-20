from __future__ import annotations

import numpy as np
import torch
from torch import nn

from zero_ttt.config import load_config
from zero_ttt.data.synthetic import SyntheticBatchSource
from zero_ttt.model.transformer import PolicyValueTransformer
from zero_ttt.training.checkpoint import CheckpointManager
from zero_ttt.training.ema import ema_decay, update_slow_weights
from zero_ttt.training.trainer import Trainer, parameters_are_finite


def test_sample_based_ema_uses_equivalent_batched_decay() -> None:
    fast = nn.Linear(2, 2, bias=False)
    slow = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        fast.weight.fill_(1.0)
        slow.weight.zero_()
    decay = update_slow_weights(slow, fast, samples=32, half_life_samples=32)
    assert decay == ema_decay(32, 32) == 0.5
    assert torch.allclose(slow.weight, torch.full_like(slow.weight, 0.5))


def test_parameter_finiteness_check_detects_nan_and_infinity() -> None:
    parameter = nn.Parameter(torch.tensor([1.0, -2.0]))
    assert parameters_are_finite((parameter,))
    with torch.no_grad():
        parameter[0] = torch.nan
    assert not parameters_are_finite((parameter,))
    with torch.no_grad():
        parameter[0] = torch.inf
    assert not parameters_are_finite((parameter,))


def test_ema_synchronizes_named_buffers() -> None:
    fast = nn.BatchNorm1d(2)
    slow = nn.BatchNorm1d(2)
    with torch.no_grad():
        fast.running_mean.copy_(torch.tensor([2.0, 3.0]))
        fast.running_var.copy_(torch.tensor([4.0, 5.0]))
        fast.num_batches_tracked.fill_(7)
        slow.running_mean.zero_()
        slow.running_var.fill_(1.0)
        slow.num_batches_tracked.zero_()
    update_slow_weights(slow, fast, samples=1, half_life_samples=1)
    assert torch.equal(slow.running_mean, fast.running_mean)
    assert torch.equal(slow.running_var, fast.running_var)
    assert torch.equal(slow.num_batches_tracked, fast.num_batches_tracked)


def test_hypernetwork_trains_from_first_step_at_reduced_learning_rate(tmp_path) -> None:
    config = load_config("configs/test.toml")
    trainer = Trainer(config, CheckpointManager(tmp_path, keep=2))
    learning_rate = trainer._set_schedule(1)
    hyper_lrs = [
        group["lr"] for group in trainer.optimizer.param_groups if group["group_name"] == "hypernet"
    ]
    assert hyper_lrs and set(hyper_lrs) == {
        learning_rate * config.model.hypernet.lr_multiplier
    }
    assert trainer.slow.cls_token.device.type == "cpu"
    assert trainer.slow.cls_token.dtype == torch.float32


def test_one_optimizer_step_ema_publish_and_restore(tmp_path) -> None:
    torch.manual_seed(5)
    config = load_config("configs/test.toml")
    run_dir = tmp_path / "run"
    manager = CheckpointManager(run_dir, keep=config.training.checkpoint_keep)
    source = SyntheticBatchSource()
    rng = np.random.default_rng(8)
    trainer = Trainer(config, manager, PolicyValueTransformer(config.model))
    metrics = trainer.train_optimizer_step(source, rng)
    assert metrics.step == 1
    assert np.isfinite(metrics.total_loss)
    assert metrics.hyper_gradient_norm is not None
    assert metrics.ema_update_seconds is not None
    assert trainer.state.samples_seen == (
        config.training.batch_size * config.training.accumulation_steps
    )
    checkpoint = trainer.save_checkpoint(rng)
    publication = trainer.publish()
    assert trainer.slow.cls_token.device.type == "cpu"
    saved_parameter = next(trainer.fast.parameters()).detach().clone()
    with torch.no_grad():
        next(trainer.fast.parameters()).add_(2.0)
    trainer.restore(checkpoint, rng)
    assert torch.equal(next(trainer.fast.parameters()), saved_parameter)
    assert publication.exists()
    published = torch.load(publication, map_location="cpu", weights_only=False)
    floating = next(
        tensor for tensor in published["slow_state"].values() if tensor.is_floating_point()
    )
    assert floating.dtype == torch.bfloat16
