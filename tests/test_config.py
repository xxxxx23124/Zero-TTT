from __future__ import annotations

from pathlib import Path

import pytest
import torch

from zero_ttt.config import load_config
from zero_ttt.model import PolicyValueTransformer


def test_load_test_config_is_stable() -> None:
    first = load_config(Path("configs/test.toml"))
    second = load_config(Path("configs/test.toml"))
    assert first == second
    assert first.sha256 == second.sha256
    assert first.model.d_model == 64
    assert first.schema_version == 4
    assert first.model.hypernet.enabled
    assert first.model.depth_mixing.enabled
    assert first.runtime.ema_device == "cpu"
    assert not hasattr(first, "search")
    assert not hasattr(first, "replay")
    assert not hasattr(first, "selfplay")


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


def test_unknown_config_field_is_rejected(tmp_path: Path) -> None:
    source = Path("configs/test.toml").read_text(encoding="utf-8")
    path = tmp_path / "invalid.toml"
    path.write_text(source.replace("seed = 7", "seed = 7\nunknown = 1"), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown fields"):
        load_config(path)
