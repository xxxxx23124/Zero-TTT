from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from zero_ttt.data.importers.protocols import ImporterRegistry
from zero_ttt.data.manifest import ManifestAsset, SourceManifest
from zero_ttt.data.records import ImportEvent


class FakeImporter:
    source_type = "fake"

    def import_asset(
        self,
        manifest: SourceManifest,
        asset: ManifestAsset,
        source_root: str | Path,
    ) -> Iterator[ImportEvent]:
        del manifest, asset, source_root
        return iter(())


def test_importer_registry_dispatches_and_rejects_unknown_sources() -> None:
    importer = FakeImporter()
    registry = ImporterRegistry((importer,))
    assert registry.resolve("fake") is importer
    with pytest.raises(ValueError, match="unsupported source_type"):
        registry.resolve("missing")


def test_importer_registry_rejects_duplicate_source_types() -> None:
    with pytest.raises(ValueError, match="duplicate importer"):
        ImporterRegistry((FakeImporter(), FakeImporter()))
