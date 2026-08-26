from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from zero_ttt.config import config_from_mapping, load_config
from zero_ttt.data.synthetic import SyntheticBatchSource
from zero_ttt.learner import Learner, LearnerDataIdentity
from zero_ttt.model import PolicyValueTransformer
from zero_ttt.training.checkpoint import CheckpointManager


def test_legacy_step_schedule_is_normalized_once(tmp_path: Path) -> None:
    import tomllib

    raw = tomllib.loads(Path("configs/test.toml").read_text(encoding="utf-8"))
    training = raw["training"]
    training["warmup_steps"] = training.pop("warmup_samples") // 4
    training["ema_update_interval"] = training.pop("ema_update_interval_samples") // 4
    training["publish_interval"] = training.pop("publish_interval_samples") // 4
    config = config_from_mapping(raw)
    assert config.training.warmup_samples == 8
    assert config.training.ema_update_interval_samples == 4
    assert config.training.publish_interval_samples == 8

    raw["training"]["warmup_samples"] = 8
    with pytest.raises(ValueError, match="cannot both"):
        config_from_mapping(raw)


def test_learning_rate_depends_on_samples_not_accumulation(tmp_path: Path) -> None:
    config = load_config("configs/test.toml")
    alternative_training = dataclasses.replace(
        config.training,
        batch_size=1,
        accumulation_steps=4,
    )
    alternative = dataclasses.replace(config, training=alternative_training)
    first = Learner(config, CheckpointManager(tmp_path / "a", keep=1))
    second = Learner(alternative, CheckpointManager(tmp_path / "b", keep=1))
    for samples in (4, 8, 32):
        assert first._base_lr(samples) == second._base_lr(samples)


def test_accumulated_and_single_batch_updates_match(tmp_path: Path) -> None:
    torch.manual_seed(11)
    config = load_config("configs/test.toml")
    large_config = dataclasses.replace(
        config,
        training=dataclasses.replace(config.training, batch_size=4, accumulation_steps=1),
    )
    accumulated_config = dataclasses.replace(
        config,
        training=dataclasses.replace(config.training, batch_size=2, accumulation_steps=2),
    )
    large_model = PolicyValueTransformer(large_config.model, large_config.execution)
    accumulated_model = PolicyValueTransformer(
        accumulated_config.model, accumulated_config.execution
    )
    accumulated_model.load_state_dict(large_model.state_dict())
    large = Learner(
        large_config,
        CheckpointManager(tmp_path / "large", keep=1),
        fast_model=large_model,
    )
    accumulated = Learner(
        accumulated_config,
        CheckpointManager(tmp_path / "accumulated", keep=1),
        fast_model=accumulated_model,
    )
    large.train_optimizer_step(SyntheticBatchSource(), np.random.default_rng(5))
    accumulated.train_optimizer_step(SyntheticBatchSource(), np.random.default_rng(5))
    for left, right in zip(large.fast.parameters(), accumulated.fast.parameters()):
        assert torch.allclose(left, right, atol=2e-6, rtol=2e-5)


def test_checkpoint_rejects_different_data_identity(tmp_path: Path) -> None:
    config = load_config("configs/test.toml")
    manager = CheckpointManager(tmp_path, keep=2)
    first = Learner(
        config,
        manager,
        data_identity=LearnerDataIdentity("snapshot-a", "1" * 64),
    )
    path = first.save_checkpoint(np.random.default_rng(1))
    second = Learner(
        config,
        manager,
        data_identity=LearnerDataIdentity("snapshot-b", "1" * 64),
    )
    with pytest.raises(ValueError, match="snapshot"):
        second.restore(path)


def test_schema_v4_checkpoint_with_legacy_step_keys_restores(tmp_path: Path) -> None:
    config = load_config("configs/test.toml")
    manager = CheckpointManager(tmp_path, keep=2)
    learner = Learner(config, manager)
    payload = learner.checkpoint_payload(np.random.default_rng(2))
    legacy = json.loads(config.canonical_json())
    training = legacy["training"]
    effective = config.training.effective_batch_size
    training["warmup_steps"] = training.pop("warmup_samples") // effective
    training["ema_update_interval"] = training.pop("ema_update_interval_samples") // effective
    training["publish_interval"] = training.pop("publish_interval_samples") // effective
    config_json = json.dumps(legacy, sort_keys=True, separators=(",", ":"))
    payload["config_json"] = config_json
    payload["config_sha256"] = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
    for key in (
        "next_ema_sample",
        "next_publish_sample",
        "last_published_samples",
        "run_id",
    ):
        payload["trainer_state"].pop(key)
    payload.pop("data_identity")
    path = tmp_path / "legacy-v4.pt"
    torch.save(payload, path)
    learner.restore(path, np.random.default_rng(2))
    assert learner.state.next_ema_sample == config.training.ema_update_interval_samples
    assert learner.state.next_publish_sample == config.training.publish_interval_samples


def test_publication_failure_does_not_advance_state(tmp_path: Path, monkeypatch) -> None:
    config = load_config("configs/test.toml")
    manager = CheckpointManager(tmp_path, keep=1)
    learner = Learner(config, manager)

    def fail(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(manager, "save_publication", fail)
    before = dataclasses.asdict(learner.state)
    with pytest.raises(OSError, match="disk full"):
        learner.publish()
    assert dataclasses.asdict(learner.state) == before
