from __future__ import annotations

import ast
from pathlib import Path


def test_runtime_schema_versions_are_not_hard_coded_outside_registry() -> None:
    violations: list[str] = []
    for path in sorted(Path("src/zero_ttt").rglob("*.py")):
        if path.name == "versioning.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.keyword)
                and node.arg == "schema_version"
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, int)
            ):
                violations.append(f"{path}:{node.lineno}")
            if isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values, strict=False):
                    if (
                        isinstance(key, ast.Constant)
                        and key.value in {"schema_version", "checkpoint_schema_version"}
                        and isinstance(value, ast.Constant)
                        and isinstance(value.value, int)
                    ):
                        violations.append(f"{path}:{node.lineno}")
    assert violations == []
