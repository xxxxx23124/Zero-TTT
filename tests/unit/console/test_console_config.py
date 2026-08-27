from __future__ import annotations

from pathlib import Path

import pytest

from zero_ttt.console.config import load_console_config


def _write(path: Path, snapshot: str = "a" * 64, hours: str = "8.0") -> None:
    path.write_text(
        "\n".join(
            (
                "schema_version = 1",
                'experiment_config = "experiment.toml"',
                'catalog_path = "catalog.sqlite"',
                'store_root = "processed"',
                f'cold_start_snapshot_id = "{snapshot}"',
                f"max_runtime_hours = {hours}",
                "",
            )
        ),
        encoding="utf-8",
    )


def test_console_config_is_strict_and_resolves_relative_paths(tmp_path: Path) -> None:
    path = tmp_path / "console.toml"
    _write(path)
    config = load_console_config(path)
    assert config.experiment_config == tmp_path / "experiment.toml"
    assert config.catalog_path == tmp_path / "catalog.sqlite"
    assert config.store_root == tmp_path / "processed"
    assert config.max_runtime_seconds == 8 * 60 * 60


@pytest.mark.parametrize(
    ("snapshot", "hours", "message"),
    (
        ("0" * 64, "8.0", "snapshot ID"),
        ("not-a-hash", "8.0", "snapshot ID"),
        ("a" * 64, "0.0", "positive"),
    ),
)
def test_console_config_rejects_placeholders_and_invalid_budget(
    tmp_path: Path,
    snapshot: str,
    hours: str,
    message: str,
) -> None:
    path = tmp_path / "console.toml"
    _write(path, snapshot, hours)
    with pytest.raises((TypeError, ValueError), match=message):
        load_console_config(path)
