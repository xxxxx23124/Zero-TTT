from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch

from zero_ttt.config import config_from_mapping, load_config
from zero_ttt.data.synthetic import SyntheticBatchSource
from zero_ttt.learner import Learner, LearnerDataIdentity
from zero_ttt.model import PolicyValueTransformer
from zero_ttt.training.checkpoint import CheckpointManager


def test_legacy_step_schedule_keys_are_rejected() -> None:
    import tomllib

    raw = tomllib.loads(Path("configs/test.toml").read_text(encoding="utf-8"))
    training = raw["training"]
    training["warmup_steps"] = training.pop("warmup_samples") // 4
    with pytest.raises(ValueError, match=r"unknown fields: warmup_steps"):
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
    for left, right in zip(large.fast.parameters(), accumulated.fast.parameters(), strict=False):
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


def test_explicit_data_transition_preserves_full_training_state(tmp_path: Path) -> None:
    torch.manual_seed(13)
    config = load_config("configs/test.toml")
    manager = CheckpointManager(tmp_path, keep=2)
    source = SyntheticBatchSource()
    first_rng = np.random.default_rng(21)
    first_identity = LearnerDataIdentity("snapshot-a", "1" * 64)
    first = Learner(config, manager, data_identity=first_identity)
    first.train_optimizer_step(source, first_rng)
    path = first.save_checkpoint(first_rng)
    expected_state = dataclasses.asdict(first.state)
    expected_parameters = [parameter.detach().clone() for parameter in first.fast.parameters()]
    expected_optimizer = first.optimizer.state_dict()

    destination_identity = LearnerDataIdentity(
        "mixture:" + "2" * 64,
        "3" * 64,
        "2" * 64,
        ("4" * 64, "5" * 64),
    )
    second_rng = np.random.default_rng(99)
    second = Learner(config, manager, data_identity=destination_identity)
    previous = second.restore_for_data_transition(path, second_rng)

    assert previous == first_identity
    assert dataclasses.asdict(second.state) == expected_state
    for actual, expected in zip(second.fast.parameters(), expected_parameters, strict=False):
        assert torch.equal(actual, expected)
    assert second.optimizer.state_dict()["param_groups"] == expected_optimizer["param_groups"]
    assert second_rng.bit_generator.state == first_rng.bit_generator.state
    transitioned = second.checkpoint_payload(second_rng)
    assert transitioned["data_identity"] == dataclasses.asdict(destination_identity)


@pytest.mark.parametrize("missing", ("data_identity", "learner_state", "state_field"))
def test_checkpoint_requires_complete_current_state(tmp_path: Path, missing: str) -> None:
    config = load_config("configs/test.toml")
    manager = CheckpointManager(tmp_path, keep=2)
    learner = Learner(config, manager)
    payload = learner.checkpoint_payload(np.random.default_rng(2))
    if missing == "state_field":
        payload["learner_state"].pop("run_id")
    else:
        payload.pop(missing)
    path = tmp_path / f"missing-{missing}.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match=r"checkpoint .* (identity|state) is incomplete"):
        learner.restore(path, np.random.default_rng(2))


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
