from __future__ import annotations

import numpy as np
import torch

from zero_ttt.config import load_config
from zero_ttt.data.synthetic import SyntheticBatchSource
from zero_ttt.model import PolicyValueTransformer
from zero_ttt.training.checkpoint import CheckpointManager
from zero_ttt.training.trainer import Trainer


def test_hypernetwork_trains_from_first_step_at_reduced_learning_rate(tmp_path) -> None:
    config = load_config("configs/test.toml")
    trainer = Trainer(config, CheckpointManager(tmp_path, keep=2))
    learning_rate = trainer._set_schedule(config.training.effective_batch_size)
    hyper_lrs = [
        group["lr"]
        for group in trainer.optimizer.param_groups
        if group["group_name"] == "hypernet"
    ]
    assert hyper_lrs and set(hyper_lrs) == {
        learning_rate * config.training.hypernet.learning_rate_multiplier
    }
    assert trainer.fast.encoder.summary_token.dtype == torch.float32
    assert all(parameter.dtype == torch.float32 for parameter in trainer.fast.parameters())
    assert trainer.slow.encoder.summary_token.device.type == "cpu"
    assert trainer.slow.encoder.summary_token.dtype == torch.float32


def test_one_optimizer_step_ema_publish_and_restore(tmp_path) -> None:
    torch.manual_seed(5)
    config = load_config("configs/test.toml")
    run_dir = tmp_path / "run"
    manager = CheckpointManager(run_dir, keep=config.training.checkpoint_keep)
    source = SyntheticBatchSource()
    rng = np.random.default_rng(8)
    trainer = Trainer(
        config,
        manager,
        PolicyValueTransformer(config.model, config.execution),
    )
    metrics = trainer.train_optimizer_step(source, rng)
    assert metrics.step == 1
    assert np.isfinite(metrics.total_loss)
    assert metrics.hypernet_gradient_norm is not None
    assert metrics.ema_update_seconds is not None
    assert trainer.state.samples_seen == (
        config.training.batch_size * config.training.accumulation_steps
    )
    checkpoint = trainer.save_checkpoint(rng)
    publication = trainer.publish()
    assert trainer.slow.encoder.summary_token.device.type == "cpu"
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
