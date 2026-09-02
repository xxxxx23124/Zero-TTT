from __future__ import annotations

import ast
import tomllib
from pathlib import Path

SERVICE_MODULES = {
    "control": "zero_ttt_control",
    "data": "zero_ttt_data",
    "trainer": "zero_ttt_trainer",
    "selfplay": "zero_ttt_selfplay_worker",
    "ui": "zero_ttt_ui",
}


def _imports(path: Path) -> tuple[tuple[int, str], ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            result.append((node.lineno, node.module))
        elif isinstance(node, ast.Import):
            result.extend((node.lineno, alias.name) for alias in node.names)
    return tuple(result)


def test_services_never_import_another_service() -> None:
    violations: list[str] = []
    for service, own_module in SERVICE_MODULES.items():
        root = Path("services") / service / "src"
        forbidden = set(SERVICE_MODULES.values()) - {own_module}
        for path in sorted(root.rglob("*.py")):
            for line, module in _imports(path):
                if any(module == name or module.startswith(f"{name}.") for name in forbidden):
                    violations.append(f"{path}:{line} imports {module}")
    assert violations == []


def test_shared_packages_do_not_import_services() -> None:
    violations: list[str] = []
    service_modules = set(SERVICE_MODULES.values())
    for path in sorted(Path("packages").rglob("*.py")):
        for line, module in _imports(path):
            if any(module == name or module.startswith(f"{name}.") for name in service_modules):
                violations.append(f"{path}:{line} imports {module}")
    assert violations == []


def test_cpu_service_manifests_do_not_depend_on_torch_or_model_package() -> None:
    for service in ("control", "data", "ui"):
        document = tomllib.loads(
            (Path("services") / service / "pyproject.toml").read_text(encoding="utf-8")
        )
        dependencies = "\n".join(document["project"].get("dependencies", ())).lower()
        assert "torch" not in dependencies
        assert "zero-ttt-model" not in dependencies


def test_legacy_orchestration_and_compatibility_facades_are_gone() -> None:
    forbidden_paths = (
        Path("src/zero_ttt/console"),
        Path("src/zero_ttt/control"),
        Path("src/zero_ttt/dashboard"),
        Path("src/zero_ttt/cli.py"),
        Path("src/zero_ttt/data/catalog/__init__.py"),
        Path("src/zero_ttt/training/session.py"),
        Path("src/zero_ttt/training/artifacts.py"),
    )
    assert not any(path.exists() for path in forbidden_paths)


def test_contracts_have_no_torch_or_service_dependencies() -> None:
    violations = []
    for path in sorted(Path("packages/contracts").rglob("*.py")):
        for line, module in _imports(path):
            if (
                module == "torch"
                or module.startswith("torch.")
                or module in SERVICE_MODULES.values()
            ):
                violations.append(f"{path}:{line} imports {module}")
    assert violations == []
