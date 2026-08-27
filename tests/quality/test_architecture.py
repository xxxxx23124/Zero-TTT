from __future__ import annotations

import ast
from pathlib import Path

FORBIDDEN_IMPORTS = {
    package: {"cli", "console", "workflow"} for package in ("game", "model", "data", "training")
}


def _zero_ttt_root(module: str) -> str | None:
    parts = module.split(".")
    return parts[1] if len(parts) > 1 and parts[0] == "zero_ttt" else None


def test_package_dependencies_follow_the_architecture_layers() -> None:
    violations: list[str] = []
    root = Path("src/zero_ttt")
    for package, forbidden in FORBIDDEN_IMPORTS.items():
        for path in sorted((root / package).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                modules: list[str] = []
                if isinstance(node, ast.ImportFrom) and node.module:
                    modules.append(node.module)
                elif isinstance(node, ast.Import):
                    modules.extend(alias.name for alias in node.names)
                for module in modules:
                    imported_root = _zero_ttt_root(module)
                    if imported_root in forbidden:
                        violations.append(f"{path}:{node.lineno} imports {module}")
    assert violations == []


def test_stable_package_exports_are_unchanged() -> None:
    expected = {
        "data": {
            "AnnotationRecord",
            "BatchSource",
            "Catalog",
            "CatalogBatchSource",
            "ImportEvent",
            "ManifestAsset",
            "MixtureBatchSource",
            "MixtureComponent",
            "ShardStore",
            "SelfPlayStatistics",
            "SnapshotStatistics",
            "SourceManifest",
            "SyntheticBatchSource",
            "TrainBatch",
            "TrainingMixtureManifest",
            "TrajectoryRecord",
        },
        "model": {
            "BasePolicyValueModel",
            "ModelDiagnostics",
            "ModelOutput",
            "ModelParameterGroup",
            "PolicyValueTransformer",
            "TokenLayout",
        },
        "inference": {
            "BatchedInferenceBroker",
            "BatchingStats",
            "InferenceBatch",
            "InferenceOutput",
            "PositionEvaluator",
            "PublicationPositionEvaluator",
            "StateEvaluation",
        },
        "training": {
            "CheckpointSummary",
            "LearnerDataIdentity",
            "ModelArtifactIdentity",
            "PublicationSummary",
        },
        "search": {
            "MCTSSearchResult",
            "OpenSpielEvaluator",
            "OpenSpielGoGame",
            "OpenSpielGoState",
            "search_position",
        },
        "selfplay": {"CollectionSummary", "SelfPlayCollector"},
    }
    for package, names in expected.items():
        path = Path("src/zero_ttt") / package / "__init__.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        assignment = next(
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets
            )
        )
        assert isinstance(assignment.value, ast.List | ast.Tuple)
        actual = {item.value for item in assignment.value.elts if isinstance(item, ast.Constant)}
        assert actual == names


def test_core_module_and_function_size_limits() -> None:
    root = Path("src/zero_ttt")
    oversized_modules = []
    oversized_functions = []
    schema_modules = {"catalog_session.py", "shard_codecs.py"}
    for path in sorted(root.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        line_count = len(source.splitlines())
        if line_count > 500:
            oversized_modules.append(f"{path}:{line_count}")
        if path.name in schema_modules:
            continue
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef) or node.end_lineno is None:
                continue
            length = node.end_lineno - node.lineno + 1
            if length > 60:
                oversized_functions.append(f"{path}:{node.lineno} {node.name} ({length})")
    assert oversized_modules == []
    assert oversized_functions == []
