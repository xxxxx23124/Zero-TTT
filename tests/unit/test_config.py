from __future__ import annotations

from pathlib import Path

import pytest

from zero_ttt.config import load_config
from zero_ttt.versioning import EXPERIMENT_CONFIG_SCHEMA


def test_load_test_config_is_stable() -> None:
    first = load_config(Path("configs/test.toml"))
    second = load_config(Path("configs/test.toml"))
    assert first == second
    assert first.sha256 == second.sha256
    assert first.model.d_model == 64
    assert first.schema_version == EXPERIMENT_CONFIG_SCHEMA.current
    assert first.model.hypernet.enabled
    assert first.model.depth_mixing.enabled
    assert first.runtime.ema_device == "cpu"
    assert first.search.max_simulations == 64
    assert not hasattr(first, "replay")
    assert first.selfplay.inference_batch_size == 16
    assert first.training.mixture.selfplay_weight == 0.8
    assert not hasattr(first, "run_name")
    assert not hasattr(first.runtime, "run_dir")


def test_unknown_config_field_is_rejected(tmp_path: Path) -> None:
    source = Path("configs/test.toml").read_text(encoding="utf-8")
    path = tmp_path / "invalid.toml"
    path.write_text(source.replace("seed = 7", "seed = 7\nunknown = 1"), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown fields"):
        load_config(path)


def test_previous_config_schema_is_rejected(tmp_path: Path) -> None:
    source = Path("configs/test.toml").read_text(encoding="utf-8")
    path = tmp_path / "v6.toml"
    path.write_text(source.replace("schema_version = 8", "schema_version = 7"), encoding="utf-8")
    with pytest.raises(ValueError, match=r"experiment config.*expected v8.*current template"):
        load_config(path)
