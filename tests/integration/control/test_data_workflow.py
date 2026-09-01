from __future__ import annotations

import zipfile
from pathlib import Path

from zero_ttt.console.engine import TrainingConsole
from zero_ttt.control.data import DataService
from zero_ttt.control.runs import RunRepository, RuntimeLayout
from zero_ttt.data.catalog import Catalog
from zero_ttt.data.shards import ShardStore


class OneStepClock:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        return 0.0 if self.calls <= 2 else 10_000.0


def test_web_data_workflow_scans_trials_resumes_verifies_and_snapshots(
    tmp_path: Path,
) -> None:
    valid_sgf = (
        b"(;FF[4]GM[1]SZ[19]HA[0]KM[0]"
        b"RU[koPOSITIONALscoreAREAtaxNONEsui1]RE[0]"
        b"C[startTurnIdx=1,mode=normal];B[aa];W[bb];B[];W[])"
    )
    data_root = tmp_path / "data"
    raw = data_root / "raw" / "katago" / "g170" / "selfplay"
    raw.mkdir(parents=True)
    for path in (
        data_root / "staging",
        data_root / "manifests",
        data_root / "processed",
        data_root / "catalog",
    ):
        path.mkdir(parents=True)
    with zipfile.ZipFile(raw / "games.zip", "w") as archive:
        archive.writestr("games.sgfs", b"\n".join((valid_sgf,) * 3))
    layout = RuntimeLayout(
        source_root=data_root,
        staging_root=data_root / "staging",
        manifest_root=data_root / "manifests",
        catalog_path=data_root / "catalog" / "catalog.sqlite",
        store_root=data_root / "processed",
        run_root=tmp_path / "runs",
        profile_root=tmp_path / "profiles",
    )
    service = DataService(layout)
    events = []

    def emit(name, payload) -> None:
        events.append((name, payload))

    for operation in ("scan", "trial-import", "full-import", "verify", "snapshot-create"):
        assert service.run(operation, emit, lambda: False)

    status = service.status()
    assert status.manifest_assets == 1
    assert status.imported_assets == 1
    assert status.games == 3
    assert status.full_import_complete
    assert status.verification_current
    assert any(name == "data_progress" for name, _payload in events)
    assert any(name == "snapshot_created" for name, _payload in events)
    with Catalog(layout.catalog_path, ShardStore(layout.store_root)) as catalog:
        snapshots = catalog.list_snapshots()
    assert len(snapshots) == 1
    assert snapshots[0].source_kind == "external"
    assert snapshots[0].split == "train"
    assert snapshots[0].games > 0

    layout.profile_root.mkdir()
    (layout.profile_root / "tiny.toml").write_text(
        Path("configs/test.toml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    runs = RunRepository(layout)
    run = runs.create("Web integration run", "tiny", snapshots[0].snapshot_id)
    context = runs.context(run.run_id, max_runtime_hours=1.0)
    console = TrainingConsole(context, clock=OneStepClock(), output=lambda _line: None)
    console.reconcile()
    console.train()
    assert console.status().optimizer_step == 1

    restarted = TrainingConsole(context, clock=OneStepClock(), output=lambda _line: None)
    restarted.reconcile()
    assert restarted.status().optimizer_step == 1
