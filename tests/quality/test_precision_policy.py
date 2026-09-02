from __future__ import annotations

import ast
from pathlib import Path


def test_runtime_has_no_reduced_precision_execution_paths() -> None:
    violations: list[str] = []
    roots = (Path("packages/model"), Path("services/trainer"), Path("services/selfplay"))
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and node.attr in {"bfloat16", "float16"}:
                    violations.append(f"{path}:{node.lineno} uses {node.attr}")
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "torch"
                    and node.func.attr == "autocast"
                ):
                    violations.append(f"{path}:{node.lineno} uses torch.autocast")
    assert violations == []
