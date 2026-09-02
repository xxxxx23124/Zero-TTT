"""Read-only experiment profiles frozen into RunSpec at creation time."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

from zero_ttt_contracts.hashing import payload_sha256

_PROFILE_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")


class ProfileRepository:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def load(self, profile_id: str) -> dict[str, Any]:
        if _PROFILE_ID.fullmatch(profile_id) is None:
            raise ValueError("invalid profile ID")
        path = self.root / f"{profile_id}.toml"
        if not path.is_file():
            raise KeyError(f"unknown profile {profile_id}")
        with path.open("rb") as handle:
            return tomllib.load(handle)

    def list(self) -> tuple[dict[str, Any], ...]:
        if not self.root.is_dir():
            return ()
        profiles = []
        for path in sorted(self.root.glob("*.toml")):
            if _PROFILE_ID.fullmatch(path.stem) is None:
                continue
            raw = self.load(path.stem)
            profiles.append(
                {
                    "profile_id": path.stem,
                    "sha256": payload_sha256(raw),
                    "model": raw.get("model", {}),
                    "training": raw.get("training", {}),
                    "selfplay": raw.get("selfplay", {}),
                }
            )
        return tuple(profiles)
