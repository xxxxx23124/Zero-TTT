"""Source-importer contracts and source-type dispatch."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Protocol

from zero_ttt_dataset.records import ImportEvent

from zero_ttt_data.manifest import ManifestAsset, SourceManifest


class RecordImporter(Protocol):
    source_type: str

    def import_asset(
        self,
        manifest: SourceManifest,
        asset: ManifestAsset,
        source_root: str | Path,
    ) -> Iterator[ImportEvent]: ...


class ImporterRegistry:
    def __init__(self, importers: Iterable[RecordImporter]) -> None:
        by_source: dict[str, RecordImporter] = {}
        for importer in importers:
            if not importer.source_type:
                raise ValueError("importer source_type cannot be empty")
            if importer.source_type in by_source:
                raise ValueError(f"duplicate importer source_type {importer.source_type!r}")
            by_source[importer.source_type] = importer
        self._by_source = by_source

    def resolve(self, source_type: str) -> RecordImporter:
        try:
            return self._by_source[source_type]
        except KeyError as error:
            raise ValueError(f"unsupported source_type {source_type!r}") from error
