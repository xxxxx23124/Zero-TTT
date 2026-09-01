"""Fixed-profile data preparation used by the local training web agent."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from zero_ttt._io import atomic_write_json, sha256_file
from zero_ttt.control.runs import RuntimeLayout
from zero_ttt.data.catalog import Catalog, ImportStatistics
from zero_ttt.data.manifest import SourceManifest
from zero_ttt.data.pipeline import ImportSummary, import_manifest
from zero_ttt.data.shards import ShardStore

DATA_OPERATIONS = frozenset({"scan", "trial-import", "full-import", "verify", "snapshot-create"})
DATASET_ID = "katago-g170"
SOURCE_TYPE = "katago-g170-sgfs-zip"
SOURCE_GLOB = "raw/katago/g170/selfplay/*.zip"
LICENSE_ID = "CC0-1.0"
LICENSE_URL = "https://katagoarchive.org/g170/LICENSE.txt"
TRIAL_GAMES = 1000

Emit = Callable[[str, dict[str, object]], None]
StopRequested = Callable[[], bool]


@dataclass(frozen=True, slots=True)
class DataStatus:
    source_root: str
    raw_directory: str
    raw_directory_exists: bool
    writable_directories_ready: bool
    raw_assets: int
    raw_bytes: int
    manifest_exists: bool
    manifest_assets: int
    verified_assets: int
    partial_assets: int
    imported_assets: int
    games: int
    positions: int
    shards: int
    full_import_complete: bool
    verification_current: bool


class DataService:
    def __init__(self, layout: RuntimeLayout) -> None:
        self.layout = layout

    @property
    def manifest_path(self) -> Path:
        return self.layout.manifest_root / "g170.json"

    @property
    def verification_path(self) -> Path:
        return self.layout.manifest_root / "g170.verify.json"

    def _clear_verification(self) -> None:
        if self.verification_path.exists():
            self.verification_path.unlink()

    def _source_files(self) -> tuple[Path, ...]:
        return tuple(
            path for path in sorted(self.layout.source_root.glob(SOURCE_GLOB)) if path.is_file()
        )

    def _import_statistics(self) -> ImportStatistics:
        if not self.layout.catalog_path.is_file():
            return ImportStatistics(0, 0, 0, 0, 0, 0)
        with Catalog(self.layout.catalog_path, ShardStore(self.layout.store_root)) as catalog:
            return catalog.import_statistics(DATASET_ID)

    def status(self) -> DataStatus:
        files = self._source_files()
        manifest = SourceManifest.load(self.manifest_path) if self.manifest_path.is_file() else None
        statistics = self._import_statistics()
        writable = all(
            path.is_dir()
            for path in (
                self.layout.staging_root,
                self.layout.manifest_root,
                self.layout.store_root,
                self.layout.catalog_path.parent,
            )
        )
        manifest_assets = 0 if manifest is None else len(manifest.assets)
        complete = (
            manifest_assets > 0
            and statistics.imported_assets == manifest_assets
            and statistics.verified_assets == 0
            and statistics.partial_assets == 0
        )
        verification_current = False
        if complete and self.verification_path.is_file() and self.manifest_path.is_file():
            try:
                verification = json.loads(self.verification_path.read_text(encoding="utf-8"))
                verification_current = (
                    verification.get("manifest_sha256") == sha256_file(self.manifest_path)
                    and verification.get("games") == statistics.games
                    and verification.get("positions") == statistics.positions
                    and verification.get("imported_assets") == statistics.imported_assets
                )
            except (OSError, ValueError):
                verification_current = False
        return DataStatus(
            source_root=str(self.layout.source_root),
            raw_directory=str(self.layout.source_root / "raw"),
            raw_directory_exists=(self.layout.source_root / "raw").is_dir(),
            writable_directories_ready=writable,
            raw_assets=len(files),
            raw_bytes=sum(path.stat().st_size for path in files),
            manifest_exists=manifest is not None,
            manifest_assets=manifest_assets,
            **asdict(statistics),
            full_import_complete=complete,
            verification_current=verification_current,
        )

    @staticmethod
    def _progress(emit: Emit) -> Callable[[str, int, int, str], None]:
        def report(phase: str, completed: int, total: int, item: str) -> None:
            emit(
                "data_progress",
                {"phase": phase, "completed": completed, "total": total, "item": item},
            )

        return report

    def _scan(self, emit: Emit, stop_requested: StopRequested) -> None:
        if not (self.layout.source_root / "raw").is_dir():
            raise FileNotFoundError(f"raw data directory does not exist: {self.layout.source_root / 'raw'}")
        self.layout.manifest_root.mkdir(parents=True, exist_ok=True)
        self._clear_verification()
        manifest = SourceManifest.create(
            DATASET_ID,
            SOURCE_TYPE,
            LICENSE_ID,
            LICENSE_URL,
            self.layout.source_root,
            SOURCE_GLOB,
            progress=self._progress(emit),
            stop_requested=stop_requested,
        )
        manifest.save(self.manifest_path)
        manifest.verify(
            self.layout.source_root,
            progress=self._progress(emit),
            stop_requested=stop_requested,
        )
        emit("data_scan_finished", {"manifest": str(self.manifest_path), "assets": len(manifest.assets)})

    def _import(
        self,
        max_games: int | None,
        emit: Emit,
        stop_requested: StopRequested,
    ) -> ImportSummary:
        if not self.manifest_path.is_file():
            raise FileNotFoundError("source manifest does not exist; scan the raw directory first")
        self._clear_verification()
        summary = import_manifest(
            self.manifest_path,
            self.layout.source_root,
            self.layout.store_root,
            self.layout.catalog_path,
            max_games,
            progress=self._progress(emit),
            stop_requested=stop_requested,
        )
        emit("data_import_finished", asdict(summary))
        return summary

    def _verify(self, emit: Emit) -> None:
        if not self.layout.catalog_path.is_file():
            raise FileNotFoundError("catalog does not exist; import data first")
        with Catalog(self.layout.catalog_path, ShardStore(self.layout.store_root)) as catalog:
            orphans = catalog.recover()
            catalog.verify()
        status = self.status()
        atomic_write_json(
            self.verification_path,
            {
                "manifest_sha256": sha256_file(self.manifest_path),
                "games": status.games,
                "positions": status.positions,
                "imported_assets": status.imported_assets,
            },
            indent=2,
        )
        emit("data_verify_finished", {"verified": True, "orphans": list(orphans)})

    def _snapshot(self, emit: Emit) -> None:
        status = self.status()
        if not status.full_import_complete or not status.verification_current:
            raise RuntimeError("full external import and verification must finish before snapshot creation")
        with Catalog(self.layout.catalog_path, ShardStore(self.layout.store_root)) as catalog:
            snapshot_id = catalog.create_snapshot(
                seed=7,
                split="train",
                validation_fraction=0.1,
                source_kind="external",
            )
            statistics = catalog.snapshot_statistics(snapshot_id)
        if statistics.games <= 0 or statistics.positions <= 0:
            raise ValueError("created snapshot contains no trainable data")
        emit(
            "snapshot_created",
            {"snapshot_id": snapshot_id, "games": statistics.games, "positions": statistics.positions},
        )

    def run(self, operation: str, emit: Emit, stop_requested: StopRequested) -> bool:
        if operation not in DATA_OPERATIONS:
            raise ValueError(f"unsupported data operation: {operation}")
        emit("data_operation_started", {"operation": operation})
        try:
            if operation == "scan":
                self._scan(emit, stop_requested)
            elif operation == "trial-import":
                self._import(TRIAL_GAMES, emit, stop_requested)
            elif operation == "full-import":
                self._import(None, emit, stop_requested)
            elif operation == "verify":
                self._verify(emit)
            else:
                self._snapshot(emit)
        except InterruptedError:
            emit("data_operation_interrupted", {"operation": operation})
            return False
        if stop_requested():
            emit("data_operation_interrupted", {"operation": operation})
            return False
        emit("data_operation_finished", {"operation": operation, "status": asdict(self.status())})
        return True
