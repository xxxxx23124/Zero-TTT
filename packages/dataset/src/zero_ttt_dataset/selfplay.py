"""Portable sealed output from the Self-play Service."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SelfPlayShard(_StrictModel):
    uri: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(gt=0)
    games: int = Field(gt=0)
    positions: int = Field(gt=0)


class SelfPlayBundle(_StrictModel):
    format_version: int = 1
    task_id: str
    source_manifest_uri: str
    source_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_manifest_size_bytes: int = Field(gt=0)
    publication_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluator_id: str
    search_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    requested_games: int = Field(gt=0)
    collected_games: int = Field(gt=0)
    shards: tuple[SelfPlayShard, ...]

    @field_validator("source_manifest_uri")
    @classmethod
    def _artifact_uri(cls, value: str) -> str:
        if not value.startswith("artifact://"):
            raise ValueError("source manifest must use an artifact URI")
        return value

    @model_validator(mode="after")
    def _validate_bundle(self) -> SelfPlayBundle:
        if self.format_version != 1:
            raise ValueError("expected self-play bundle format v1")
        if not self.shards or self.collected_games != self.requested_games:
            raise ValueError("sealed bundle must contain every requested game")
        if sum(item.games for item in self.shards) != self.collected_games:
            raise ValueError("bundle game count does not match its shards")
        return self
