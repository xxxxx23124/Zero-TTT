"""Immutable web-created run specifications and host-independent runtime paths."""

from __future__ import annotations

import json
import os
import re
import shutil
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from zero_ttt._io import atomic_write_json
from zero_ttt.config import ExperimentConfig, load_config
from zero_ttt.console.config import RunContext
from zero_ttt.data.catalog import Catalog
from zero_ttt.data.shards import ShardStore
from zero_ttt.versioning import RUN_SPEC_SCHEMA

_RUN_ID = re.compile(r"[0-9a-f]{32}")
_PROFILE_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")


@dataclass(frozen=True, slots=True)
class RuntimeLayout:
    source_root: Path
    staging_root: Path
    manifest_root: Path
    catalog_path: Path
    store_root: Path
    run_root: Path
    profile_root: Path

    @classmethod
    def from_environment(cls) -> RuntimeLayout:
        return cls(
            source_root=Path(os.environ.get("ZERO_TTT_SOURCE_ROOT", "/datasets")),
            staging_root=Path(os.environ.get("ZERO_TTT_STAGING_ROOT", "/datasets/staging")),
            manifest_root=Path(os.environ.get("ZERO_TTT_MANIFEST_ROOT", "/datasets/manifests")),
            catalog_path=Path(
                os.environ.get("ZERO_TTT_CATALOG_PATH", "/datasets/catalog/catalog.sqlite")
            ),
            store_root=Path(os.environ.get("ZERO_TTT_STORE_ROOT", "/datasets/processed")),
            run_root=Path(os.environ.get("ZERO_TTT_RUN_ROOT", "/runs")),
            profile_root=Path(os.environ.get("ZERO_TTT_PROFILE_ROOT", "/profiles")),
        )


@dataclass(frozen=True, slots=True)
class ProfileSummary:
    profile_id: str
    config_sha256: str
    seed: int
    model: dict[str, int]
    training: dict[str, int | float]
    selfplay: dict[str, int]


@dataclass(frozen=True, slots=True)
class RunSpec:
    schema_version: int
    run_id: str
    name: str
    profile_id: str
    config_sha256: str
    cold_start_snapshot_id: str
    created_ns: int

    def payload(self) -> dict[str, object]:
        return asdict(self)


class RunRepository:
    def __init__(self, layout: RuntimeLayout) -> None:
        self.layout = layout

    def _profile_path(self, profile_id: str) -> Path:
        if _PROFILE_ID.fullmatch(profile_id) is None:
            raise ValueError("invalid profile ID")
        path = self.layout.profile_root / f"{profile_id}.toml"
        if not path.is_file():
            raise KeyError(f"unknown profile {profile_id}")
        return path

    @staticmethod
    def _profile_summary(profile_id: str, config: ExperimentConfig) -> ProfileSummary:
        return ProfileSummary(
            profile_id=profile_id,
            config_sha256=config.sha256,
            seed=config.seed,
            model={
                "d_model": config.model.d_model,
                "n_layers": config.model.n_layers,
                "n_heads": config.model.n_heads,
            },
            training={
                "batch_size": config.training.batch_size,
                "accumulation_steps": config.training.accumulation_steps,
                "learning_rate": config.training.learning_rate,
            },
            selfplay={
                "actor_count": config.selfplay.actor_count,
                "max_simulations": config.search.max_simulations,
            },
        )

    def list_profiles(self) -> tuple[ProfileSummary, ...]:
        if not self.layout.profile_root.is_dir():
            return ()
        return tuple(
            self._profile_summary(path.stem, load_config(path))
            for path in sorted(self.layout.profile_root.glob("*.toml"))
            if _PROFILE_ID.fullmatch(path.stem)
        )

    def _run_dir(self, run_id: str) -> Path:
        if _RUN_ID.fullmatch(run_id) is None:
            raise ValueError("invalid run ID")
        return self.layout.run_root / run_id

    def load(self, run_id: str) -> RunSpec:
        path = self._run_dir(run_id) / "run.json"
        if not path.is_file():
            raise KeyError(f"unknown run {run_id}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        RUN_SPEC_SCHEMA.require(raw.get("schema_version"))
        spec = RunSpec(**raw)
        if spec.run_id != run_id:
            raise ValueError("run directory and descriptor IDs do not match")
        config = load_config(path.parent / "experiment.toml")
        if config.sha256 != spec.config_sha256:
            raise ValueError("frozen experiment configuration does not match run.json")
        return spec

    def list_runs(self) -> tuple[RunSpec, ...]:
        if not self.layout.run_root.is_dir():
            return ()
        runs = [
            self.load(path.name)
            for path in self.layout.run_root.iterdir()
            if path.is_dir() and _RUN_ID.fullmatch(path.name) and (path / "run.json").is_file()
        ]
        return tuple(sorted(runs, key=lambda item: item.created_ns, reverse=True))

    def _validate_snapshot(self, snapshot_id: str) -> None:
        if not self.layout.catalog_path.is_file():
            raise FileNotFoundError("catalog does not exist; import and verify data first")
        with Catalog(self.layout.catalog_path, ShardStore(self.layout.store_root)) as catalog:
            summary = next(
                (item for item in catalog.list_snapshots() if item.snapshot_id == snapshot_id),
                None,
            )
        if summary is None:
            raise KeyError(f"unknown snapshot {snapshot_id}")
        if summary.source_kind != "external" or summary.split != "train":
            raise ValueError("cold-start snapshot must be a train snapshot from external data")
        if summary.games <= 0 or summary.positions <= 0:
            raise ValueError("cold-start snapshot contains no trainable data")

    def create(self, name: str, profile_id: str, cold_start_snapshot_id: str) -> RunSpec:
        clean_name = name.strip()
        if not clean_name or len(clean_name) > 80:
            raise ValueError("run name must contain 1 to 80 characters")
        if any(item.name.casefold() == clean_name.casefold() for item in self.list_runs()):
            raise ValueError("run name already exists")
        profile_path = self._profile_path(profile_id)
        config = load_config(profile_path)
        self._validate_snapshot(cold_start_snapshot_id)
        run_id = uuid.uuid4().hex
        spec = RunSpec(
            RUN_SPEC_SCHEMA.current,
            run_id,
            clean_name,
            profile_id,
            config.sha256,
            cold_start_snapshot_id,
            time.time_ns(),
        )
        self.layout.run_root.mkdir(parents=True, exist_ok=True)
        temporary = self.layout.run_root / f".run-{run_id}"
        destination = self._run_dir(run_id)
        temporary.mkdir()
        try:
            shutil.copyfile(profile_path, temporary / "experiment.toml")
            atomic_write_json(temporary / "run.json", spec.payload(), indent=2)
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return spec

    def context(self, run_id: str, max_runtime_hours: float) -> RunContext:
        spec = self.load(run_id)
        run_dir = self._run_dir(run_id)
        return RunContext(
            run_id=spec.run_id,
            name=spec.name,
            experiment_config=run_dir / "experiment.toml",
            run_dir=run_dir,
            catalog_path=self.layout.catalog_path,
            store_root=self.layout.store_root,
            cold_start_snapshot_id=spec.cold_start_snapshot_id,
            max_runtime_hours=max_runtime_hours,
        )
