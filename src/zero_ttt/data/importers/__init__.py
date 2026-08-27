"""Pure importer implementations with lazy optional-dependency loading."""

from typing import TYPE_CHECKING, Any

from zero_ttt.data.importers.protocols import ImporterRegistry, RecordImporter

if TYPE_CHECKING:
    from zero_ttt.data.importers.katago_sgf import KataGoSgfImporter


class _LazyKataGoSgfImporter:
    source_type = "katago-g170-sgfs-zip"

    def import_asset(self, *args: Any, **kwargs: Any) -> Any:
        from zero_ttt.data.importers.katago_sgf import KataGoSgfImporter

        return KataGoSgfImporter().import_asset(*args, **kwargs)


DEFAULT_IMPORTERS = ImporterRegistry((_LazyKataGoSgfImporter(),))


def __getattr__(name: str) -> object:
    if name == "KataGoSgfImporter":
        from zero_ttt.data.importers.katago_sgf import KataGoSgfImporter

        return KataGoSgfImporter
    raise AttributeError(name)


__all__ = [
    "DEFAULT_IMPORTERS",
    "ImporterRegistry",
    "KataGoSgfImporter",
    "RecordImporter",
]
