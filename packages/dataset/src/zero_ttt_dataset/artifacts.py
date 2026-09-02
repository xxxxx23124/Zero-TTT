"""Local content-addressed implementation of the artifact:// contract."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from zero_ttt_contracts import ArtifactKind, ArtifactRef
from zero_ttt_contracts.hashing import canonical_json_bytes, sha256_bytes, sha256_file


class LocalArtifactStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, uri: str) -> Path:
        prefix = "artifact://"
        if not uri.startswith(prefix):
            raise ValueError("artifact URI must use artifact://")
        relative = PurePosixPath(uri[len(prefix) :])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("artifact URI escapes the store root")
        path = (self.root / Path(*relative.parts)).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as error:
            raise ValueError("artifact URI escapes the store root") from error
        return path

    def verify(self, reference: ArtifactRef) -> Path:
        path = self.resolve(reference.uri)
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != reference.size_bytes:
            raise ValueError(f"artifact size mismatch: {reference.artifact_id}")
        if sha256_file(path) != reference.sha256:
            raise ValueError(f"artifact SHA-256 mismatch: {reference.artifact_id}")
        return path

    def commit_json(
        self,
        *,
        uri: str,
        artifact_id: str,
        kind: ArtifactKind,
        value: Any,
        format_version: int,
        labels: dict[str, str] | None = None,
    ) -> ArtifactRef:
        payload = canonical_json_bytes(value)
        digest = sha256_bytes(payload)
        destination = self.resolve(uri)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.read_bytes() != payload:
                raise ValueError(f"artifact destination already contains different data: {uri}")
        else:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
            )
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, destination)
            finally:
                if temporary.exists():
                    temporary.unlink()
        return ArtifactRef(
            kind=kind,
            artifact_id=artifact_id,
            format_version=format_version,
            sha256=digest,
            uri=uri,
            size_bytes=len(payload),
            labels=labels or {},
        )
