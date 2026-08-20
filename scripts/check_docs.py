"""Enforce small Markdown files and valid local links under docs/."""

from __future__ import annotations

import re
from pathlib import Path


MAX_LINES = 150
MAX_BYTES = 12 * 1024
LINK_PATTERN = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def validate_docs(root: Path) -> list[str]:
    errors: list[str] = []
    for path in sorted(root.rglob("*.md")):
        payload = path.read_bytes()
        text = payload.decode("utf-8")
        lines = len(text.splitlines())
        if lines > MAX_LINES:
            errors.append(f"{path}: {lines} lines exceeds {MAX_LINES}")
        if len(payload) > MAX_BYTES:
            errors.append(f"{path}: {len(payload)} bytes exceeds {MAX_BYTES}")
        for raw_target in LINK_PATTERN.findall(text):
            target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            relative = target.split("#", 1)[0]
            destination = (path.parent / relative).resolve()
            if not destination.exists():
                errors.append(f"{path}: broken link {target}")
    return errors


def main() -> None:
    errors = validate_docs(Path("docs"))
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Documentation checks passed ({MAX_LINES} lines, {MAX_BYTES} bytes per file).")


if __name__ == "__main__":
    main()
