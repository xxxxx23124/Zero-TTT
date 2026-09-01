"""External-source manifests with deterministic integrity checking."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from zero_ttt._io import atomic_write_json, sha256_file
from zero_ttt.versioning import SOURCE_MANIFEST_SCHEMA

ManifestProgress = Callable[[str, int, int, str], None]
StopRequested = Callable[[], bool]


@dataclass(frozen=True, slots=True)
class ManifestAsset:
    relative_path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        path = Path(self.relative_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("manifest asset paths must be relative and contained")
        try:
            bytes.fromhex(self.sha256)
        except ValueError as error:
            raise ValueError("manifest asset integrity metadata is invalid") from error
        if len(self.sha256) != 64 or self.size_bytes < 0:
            raise ValueError("manifest asset integrity metadata is invalid")


@dataclass(frozen=True, slots=True)
class SourceManifest:
    schema_version: int
    dataset_id: str
    source_type: str
    license_id: str
    license_url: str
    assets: tuple[ManifestAsset, ...]

    def __post_init__(self) -> None:
        SOURCE_MANIFEST_SCHEMA.require(self.schema_version)
        if not self.dataset_id or not self.source_type or not self.license_id:
            raise ValueError("manifest source identity cannot be empty")
        if not self.assets:
            raise ValueError("manifest must contain at least one asset")
        paths = [asset.relative_path for asset in self.assets]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError("manifest assets must be unique and lexicographically sorted")

    @classmethod
    def create(
        cls,
        dataset_id: str,
        source_type: str,
        license_id: str,
        license_url: str,
        source_root: str | Path,
        pattern: str,
        *,
        progress: ManifestProgress | None = None,
        stop_requested: StopRequested | None = None,
    ) -> SourceManifest:
        root = Path(source_root).resolve()
        assets = []
        files = tuple(path for path in sorted(root.glob(pattern)) if path.is_file())
        for index, path in enumerate(files, start=1):
            if stop_requested is not None and stop_requested():
                raise InterruptedError("manifest scan was safely stopped between source files")
            if progress is not None:
                progress("hashing", index - 1, len(files), path.relative_to(root).as_posix())
            assets.append(
                ManifestAsset(
                    relative_path=path.relative_to(root).as_posix(),
                    sha256=sha256_file(path),
                    size_bytes=path.stat().st_size,
                )
            )
            if progress is not None:
                progress("hashing", index, len(files), path.relative_to(root).as_posix())
        return cls(
            schema_version=SOURCE_MANIFEST_SCHEMA.current,
            dataset_id=dataset_id,
            source_type=source_type,
            license_id=license_id,
            license_url=license_url,
            assets=tuple(assets),
        )

    def save(self, path: str | Path) -> None:
        atomic_write_json(path, asdict(self), indent=2)

    @classmethod
    def load(cls, path: str | Path) -> SourceManifest:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        SOURCE_MANIFEST_SCHEMA.require(raw.get("schema_version"))
        assets = tuple(ManifestAsset(**item) for item in raw.pop("assets"))
        return cls(assets=assets, **raw)

    def verify(
        self,
        source_root: str | Path,
        *,
        progress: ManifestProgress | None = None,
        stop_requested: StopRequested | None = None,
    ) -> None:
        root = Path(source_root).resolve()
        for index, asset in enumerate(self.assets, start=1):
            if stop_requested is not None and stop_requested():
                raise InterruptedError("manifest verification was safely stopped between source files")
            if progress is not None:
                progress("verifying", index - 1, len(self.assets), asset.relative_path)
            path = (root / asset.relative_path).resolve()
            try:
                path.relative_to(root)
            except ValueError as error:
                raise ValueError(f"asset escapes source root: {asset.relative_path}") from error
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.stat().st_size != asset.size_bytes:
                raise ValueError(f"asset size mismatch: {asset.relative_path}")
            if sha256_file(path) != asset.sha256:
                raise ValueError(f"asset SHA-256 mismatch: {asset.relative_path}")
            if progress is not None:
                progress("verifying", index, len(self.assets), asset.relative_path)
