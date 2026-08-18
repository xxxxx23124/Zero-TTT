from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch
from torch import nn

from zero_ttt.config import load_config
from zero_ttt.model.transformer import PolicyValueTransformer
from zero_ttt.replay.sampler import ReplaySampler
from zero_ttt.replay.sqlite_store import ReplayStore
from zero_ttt.training.checkpoint import CheckpointManager
from zero_ttt.training.ema import ema_decay, update_slow_weights
from zero_ttt.training.trainer import Trainer
from test_replay import make_record


def test_sample_based_ema_uses_equivalent_batched_decay() -> None:
    fast = nn.Linear(2, 2, bias=False)
    slow = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        fast.weight.fill_(1.0)
        slow.weight.zero_()
    decay = update_slow_weights(slow, fast, samples=32, half_life_samples=32)
    assert decay == ema_decay(32, 32) == 0.5
    assert torch.allclose(slow.weight, torch.full_like(slow.weight, 0.5))


def test_hypernetwork_freeze_and_device_scalar_ramp(tmp_path) -> None:
    config = load_config("configs/test.toml")
    hyper = replace(config.model.hypernet, enabled=True)
    config = replace(config, model=replace(config.model, hypernet=hyper))
    trainer = Trainer(config, CheckpointManager(tmp_path, keep=2))
    _, frozen = trainer._set_schedule(hyper.freeze_steps)
    hyper_lrs = [
        group["lr"] for group in trainer.optimizer.param_groups if group["group_name"] == "hypernet"
    ]
    assert frozen == 0.0
    assert hyper_lrs and set(hyper_lrs) == {0.0}
    _, ramped = trainer._set_schedule(hyper.freeze_steps + 1)
    assert ramped == 1.0 / hyper.ramp_steps
    assert trainer.fast.hypernet_scale.item() == ramped


def test_one_optimizer_step_ema_publish_and_restore(tmp_path) -> None:
    torch.manual_seed(5)
    config = load_config("configs/test.toml")
    run_dir = tmp_path / "run"
    manager = CheckpointManager(run_dir, keep=config.training.checkpoint_keep)
    with ReplayStore(
        tmp_path / "replay.sqlite3",
        capacity_positions=20,
        decoded_cache_games=2,
    ) as store:
        store.add_game(make_record(length=4))
        sampler = ReplaySampler(store, decoded_cache_games=2)
        rng = np.random.default_rng(8)
        trainer = Trainer(config, manager, PolicyValueTransformer(config.model))
        metrics = trainer.train_optimizer_step(sampler, rng)
        assert metrics.step == 1
        assert np.isfinite(metrics.total_loss)
        assert trainer.state.samples_seen == (
            config.training.batch_size * config.training.accumulation_steps
        )
        checkpoint = trainer.save_checkpoint(rng)
        publication = trainer.publish()
        saved_parameter = next(trainer.fast.parameters()).detach().clone()
        with torch.no_grad():
            next(trainer.fast.parameters()).add_(2.0)
        trainer.restore(checkpoint, rng)
        assert torch.equal(next(trainer.fast.parameters()), saved_parameter)
        assert publication.exists()
        published = torch.load(publication, map_location="cpu", weights_only=False)
        floating = next(tensor for tensor in published["slow_state"].values() if tensor.is_floating_point())
        assert floating.dtype == torch.bfloat16
