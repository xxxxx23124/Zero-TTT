from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from zero_ttt.config import load_config
from zero_ttt.data.synthetic import SyntheticBatchSource
from zero_ttt.model import PolicyValueTransformer
from zero_ttt.training.checkpoint import CheckpointManager
from zero_ttt.training.ema import ema_decay, update_slow_weights
from zero_ttt.training.gradients import clip_model_gradients
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


def test_fp32_ema_retains_updates_smaller_than_bfloat16_resolution() -> None:
    fast = nn.Linear(1, 1, bias=False).float()
    slow = nn.Linear(1, 1, bias=False).float()
    with torch.no_grad():
        slow.weight.fill_(1.0)
        fast.weight.fill_(1.001)
    update_slow_weights(slow, fast, samples=1, half_life_samples=1)
    assert slow.weight.dtype == torch.float32
    assert 1.0 < slow.weight.item() < fast.weight.item()
    assert torch.tensor(slow.weight.item(), dtype=torch.bfloat16).item() == 1.0


def test_parameter_groups_are_complete_disjoint_and_clipped_once() -> None:
    config = load_config("configs/test.toml")
    model = PolicyValueTransformer(config.model, config.execution)
    groups = model.parameter_groups()
    assert {group.name for group in groups} == {"base", "hypernet"}
    grouped_ids = [id(parameter) for group in groups for parameter in group.parameters]
    trainable_ids = [id(parameter) for parameter in model.parameters() if parameter.requires_grad]
    assert len(grouped_ids) == len(set(grouped_ids))
    assert set(grouped_ids) == set(trainable_ids)
    base = next(group for group in groups if group.name == "base")
    assert any(
        parameter is model.encoder.summary_token for parameter in base.no_decay
    )

    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    norms = clip_model_gradients(
        groups,
        base_max_norm=0.5,
        hypernet_max_norm=0.25,
    )
    assert norms.base > 0.5
    assert norms.hypernet is not None and norms.hypernet > 0.25
    for group, limit in ((base, 0.5), (next(g for g in groups if g.name == "hypernet"), 0.25)):
        post_clip = torch.linalg.vector_norm(
            torch.stack(
                [
                    torch.linalg.vector_norm(parameter.grad)
                    for parameter in group.parameters
                    if parameter.grad is not None
                ]
            )
        )
        assert post_clip <= limit + 1e-5


def test_hypernetwork_trains_from_first_step_at_reduced_learning_rate(tmp_path) -> None:
    config = load_config("configs/test.toml")
    trainer = Trainer(config, CheckpointManager(tmp_path, keep=2))
    learning_rate = trainer._set_schedule(config.training.effective_batch_size)
    hyper_lrs = [
        group["lr"] for group in trainer.optimizer.param_groups if group["group_name"] == "hypernet"
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


def test_legacy_checkpoint_schema_is_rejected(tmp_path) -> None:
    path = tmp_path / "legacy.pt"
    torch.save({"checkpoint_schema_version": 3}, path)
    with pytest.raises(ValueError, match="unsupported checkpoint schema"):
        CheckpointManager.load(path)


def test_publications_are_immutable_run_scoped_and_retained(tmp_path) -> None:
    manager = CheckpointManager(tmp_path, keep=1, publication_keep=2)
    metadata = {
        "checkpoint_schema_version": 4,
        "config_json": "{}",
        "config_sha256": "0" * 64,
    }
    state = {"weight": torch.ones(1)}
    first = manager.save_publication("run-a", 1, 4, state, metadata)
    manager.save_publication("run-a", 2, 8, state, metadata)
    latest = manager.save_publication("run-a", 3, 12, state, metadata)
    assert not first.exists()
    assert latest.exists()
    assert manager.current_publication() == latest
    assert manager.save_publication("run-a", 3, 12, state, metadata) == latest
    with pytest.raises(FileExistsError, match="conflicting"):
        manager.save_publication("run-a", 3, 16, state, metadata)
