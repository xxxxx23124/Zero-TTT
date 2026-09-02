"""Validate current documentation without imposing arbitrary file-size limits."""

from __future__ import annotations

import re
from pathlib import Path

LINK_PATTERN = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
DOCUMENTATION_ROOTS = (Path("README.md"), Path("docs"))
STALE_CAPABILITIES = {
    "/api/v2": "the public API is /api/v1",
    "training-console": "the legacy console was removed",
    "console.toml": "service config uses environment variables",
    "reconcile": "lease recovery replaced reconcile",
    "无限循环": "workflows are finite",
}


def _markdown_files(root: Path) -> tuple[Path, ...]:
    return (root,) if root.is_file() else tuple(sorted(root.rglob("*.md")))


def validate_docs(root: Path) -> list[str]:
    errors: list[str] = []
    for path in _markdown_files(root):
        text = path.read_text(encoding="utf-8")
        for raw_target in LINK_PATTERN.findall(text):
            target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            relative = target.split("#", 1)[0]
            destination = (path.parent / relative).resolve()
            if not destination.exists():
                errors.append(f"{path}: broken link {target}")
        lowered = text.lower()
        for phrase, reason in STALE_CAPABILITIES.items():
            if phrase.lower() in lowered:
                errors.append(f"{path}: stale capability {phrase!r}: {reason}")
    return errors


def main() -> None:
    errors = [error for root in DOCUMENTATION_ROOTS for error in validate_docs(root)]
    if errors:
        raise SystemExit("\n".join(errors))
    print("Documentation links and implemented-capability claims are current.")


if __name__ == "__main__":
    main()
