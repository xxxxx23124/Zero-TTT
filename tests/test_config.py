from __future__ import annotations

from pathlib import Path

import pytest

from zero_ttt.config import load_config


def test_load_test_config_is_stable() -> None:
    first = load_config(Path("configs/test.toml"))
    second = load_config(Path("configs/test.toml"))
    assert first == second
    assert first.sha256 == second.sha256
    assert first.model.d_model == 64
    assert first.search.budgets == (1, 2, 3, 4)


def test_unknown_config_field_is_rejected(tmp_path: Path) -> None:
    source = Path("configs/test.toml").read_text(encoding="utf-8")
    path = tmp_path / "invalid.toml"
    path.write_text(source.replace("seed = 7", "seed = 7\nunknown = 1"), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown fields"):
        load_config(path)
