from pathlib import Path

import pytest

from zero_ttt.console.config import RunContext


def test_run_context_keeps_operational_inputs_out_of_experiment_config(tmp_path: Path) -> None:
    context = RunContext(
        run_id="a" * 32,
        name="first run",
        experiment_config=tmp_path / "experiment.toml",
        run_dir=tmp_path / "runs" / ("a" * 32),
        catalog_path=tmp_path / "catalog.sqlite",
        store_root=tmp_path / "processed",
        cold_start_snapshot_id="b" * 64,
        max_runtime_hours=8.0,
    )
    assert context.max_runtime_seconds == 8 * 60 * 60


@pytest.mark.parametrize(
    ("run_id", "snapshot", "hours", "message"),
    (
        ("bad", "b" * 64, 8.0, "run_id"),
        ("a" * 32, "bad", 8.0, "snapshot"),
        ("a" * 32, "b" * 64, 0.0, "positive"),
    ),
)
def test_run_context_rejects_untrusted_identity_or_budget(
    tmp_path: Path, run_id: str, snapshot: str, hours: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        RunContext(
            run_id=run_id,
            name="run",
            experiment_config=tmp_path / "experiment.toml",
            run_dir=tmp_path / "run",
            catalog_path=tmp_path / "catalog.sqlite",
            store_root=tmp_path / "processed",
            cold_start_snapshot_id=snapshot,
            max_runtime_hours=hours,
        )
