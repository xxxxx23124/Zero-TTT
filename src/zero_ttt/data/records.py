"""Versioned, storage-independent records emitted by data importers."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from zero_ttt.game.rules import ACTION_SIZE, BOARD_AREA


RECORD_SCHEMA_VERSION = 3
SUPPORTED_RECORD_SCHEMA_VERSIONS = frozenset({2, 3})


def stable_game_id(
    dataset_id: str,
    asset_sha256: str,
    member_path: str,
    ordinal: int,
) -> str:
    payload = json.dumps(
        [dataset_id, asset_sha256, member_path, ordinal],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_json(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class TrajectoryRecord:
    schema_version: int
    game_id: str
    content_sha256: str
    dataset_id: str
    asset_sha256: str
    member_path: str
    ordinal: int
    rules: str
    komi_half_points: int
    max_moves: int
    moves: tuple[int, ...]
    trainable_start_ply: int
    policy_row_offsets: tuple[int, ...]
    policy_actions: tuple[int, ...]
    policy_values: tuple[float, ...]
    value_black: float
    value_available: bool
    score_margin_black: float
    score_available: bool
    ownership_black: tuple[float, ...]
    ownership_available: bool
    source_kind: str = "external/played_move"
    task_id: str = ""
    termination: str = "external"
    game_seed: int = 0
    black_agent_id: str = ""
    white_agent_id: str = ""
    publication_sha256: str = ""
    feature_schema_id: str = ""
    search_config_sha256: str = ""
    search_budgets: tuple[int, ...] = ()
    root_values: tuple[float, ...] = ()
    root_score_margins: tuple[float, ...] = ()
    temperatures: tuple[float, ...] = ()
    search_seeds: tuple[int, ...] = ()
    root_noise_mask: tuple[bool, ...] = ()
    search_metadata_mask: tuple[bool, ...] = ()
    root_score_mask: tuple[bool, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version not in SUPPORTED_RECORD_SCHEMA_VERSIONS:
            raise ValueError("unsupported trajectory record schema")
        try:
            bytes.fromhex(self.game_id)
            bytes.fromhex(self.asset_sha256)
        except ValueError as error:
            raise ValueError("game and asset identities must be SHA-256 hex strings") from error
        if len(self.game_id) != 64 or len(self.asset_sha256) != 64:
            raise ValueError("game and asset identities must be SHA-256 hex strings")
        if self.max_moves < 2:
            raise ValueError("trajectory max_moves must be at least 2")
        if not 0 <= self.trainable_start_ply <= len(self.moves):
            raise ValueError("trainable_start_ply is outside the move sequence")
        if any(not 0 <= action < ACTION_SIZE for action in self.moves):
            raise ValueError("trajectory contains an invalid action")
        if len(self.moves) > self.max_moves:
            raise ValueError("trajectory exceeds its max_moves boundary")
        positions = self.trainable_position_count
        defaults: tuple[tuple[str, object], ...] = (
            ("search_budgets", 0),
            ("root_values", 0.0),
            ("root_score_margins", 0.0),
            ("temperatures", 0.0),
            ("search_seeds", 0),
            ("root_noise_mask", False),
            ("search_metadata_mask", False),
            ("root_score_mask", False),
        )
        for name, default in defaults:
            values = getattr(self, name)
            if not values:
                object.__setattr__(self, name, (default,) * positions)
            elif len(values) != positions:
                raise ValueError(f"{name} must describe every trainable position")
        if len(self.policy_row_offsets) != positions + 1:
            raise ValueError("policy_row_offsets must describe every trainable position")
        if not self.policy_row_offsets or self.policy_row_offsets[0] != 0:
            raise ValueError("policy_row_offsets must start at zero")
        if any(a > b for a, b in zip(self.policy_row_offsets, self.policy_row_offsets[1:])):
            raise ValueError("policy_row_offsets must be monotonic")
        if self.policy_row_offsets[-1] != len(self.policy_actions):
            raise ValueError("policy offsets and actions disagree")
        if len(self.policy_actions) != len(self.policy_values):
            raise ValueError("policy actions and values disagree")
        normalized_policy = tuple(float(value) for value in np.asarray(self.policy_values, dtype=np.float32))
        normalized_ownership = tuple(
            float(value) for value in np.asarray(self.ownership_black, dtype=np.float32)
        )
        object.__setattr__(self, "policy_values", normalized_policy)
        object.__setattr__(self, "value_black", float(np.float32(self.value_black)))
        object.__setattr__(self, "score_margin_black", float(np.float32(self.score_margin_black)))
        object.__setattr__(self, "ownership_black", normalized_ownership)
        normalized_roots = tuple(
            float(value) for value in np.asarray(self.root_values, dtype=np.float32)
        )
        normalized_scores = tuple(
            float(value)
            for value in np.asarray(self.root_score_margins, dtype=np.float32)
        )
        normalized_temperatures = tuple(
            float(value) for value in np.asarray(self.temperatures, dtype=np.float32)
        )
        object.__setattr__(self, "root_values", normalized_roots)
        object.__setattr__(self, "root_score_margins", normalized_scores)
        object.__setattr__(self, "temperatures", normalized_temperatures)
        for start, end in zip(self.policy_row_offsets, self.policy_row_offsets[1:]):
            values = normalized_policy[start:end]
            if not values or any(value < 0 or not math.isfinite(value) for value in values):
                raise ValueError("each policy row must be a non-negative distribution")
            if abs(sum(values) - 1.0) > 1e-5:
                raise ValueError("each policy row must sum to one")
        if any(not 0 <= action < ACTION_SIZE for action in self.policy_actions):
            raise ValueError("policy contains an invalid action")
        if len(self.ownership_black) != BOARD_AREA:
            raise ValueError("ownership_black must contain exactly 361 values")
        if not all(
            math.isfinite(value)
            for value in (
                self.value_black,
                self.score_margin_black,
                *self.ownership_black,
                *normalized_roots,
                *normalized_scores,
                *normalized_temperatures,
            )
        ):
            raise ValueError("trajectory targets must be finite")
        if self.schema_version == 2:
            if self.source_kind != "external/played_move" or any(self.search_metadata_mask):
                raise ValueError("v2 trajectories cannot contain v3 search metadata")
        else:
            if not self.source_kind or not self.termination:
                raise ValueError("v3 trajectories require source and termination identities")
            if not 0 <= self.game_seed < 1 << 64:
                raise ValueError("game_seed must be an unsigned 64-bit integer")
            if any(not 0 <= seed < 1 << 64 for seed in self.search_seeds):
                raise ValueError("search seeds must be unsigned 64-bit integers")
            if any(value < 0 for value in self.search_budgets):
                raise ValueError("search budgets cannot be negative")
            if any(value < 0 for value in normalized_temperatures):
                raise ValueError("temperatures cannot be negative")
            for index, available in enumerate(self.search_metadata_mask):
                if available and self.search_budgets[index] <= 0:
                    raise ValueError("available search metadata requires a positive budget")
            if self.source_kind == "selfplay/mcts":
                identities = (
                    self.task_id,
                    self.black_agent_id,
                    self.white_agent_id,
                    self.publication_sha256,
                    self.feature_schema_id,
                    self.search_config_sha256,
                )
                if any(not value for value in identities) or not all(
                    self.search_metadata_mask
                ):
                    raise ValueError("MCTS self-play trajectories require complete identities")
        expected = self.compute_content_sha256()
        if self.content_sha256 and self.content_sha256 != expected:
            raise ValueError("trajectory content_sha256 does not match its contents")
        if not self.content_sha256:
            object.__setattr__(self, "content_sha256", expected)

    @property
    def trainable_position_count(self) -> int:
        return len(self.moves) - self.trainable_start_ply

    def policy_at(self, local_position: int) -> tuple[tuple[int, ...], tuple[float, ...]]:
        if not 0 <= local_position < self.trainable_position_count:
            raise IndexError("trainable position is outside this trajectory")
        start = self.policy_row_offsets[local_position]
        end = self.policy_row_offsets[local_position + 1]
        return self.policy_actions[start:end], self.policy_values[start:end]

    def compute_content_sha256(self) -> str:
        payload = {
                "schema_version": self.schema_version,
                "game_id": self.game_id,
                "dataset_id": self.dataset_id,
                "asset_sha256": self.asset_sha256,
                "member_path": self.member_path,
                "ordinal": self.ordinal,
                "rules": self.rules,
                "komi_half_points": self.komi_half_points,
                "max_moves": self.max_moves,
                "moves": self.moves,
                "trainable_start_ply": self.trainable_start_ply,
                "policy_row_offsets": self.policy_row_offsets,
                "policy_actions": self.policy_actions,
                "policy_values": self.policy_values,
                "value_black": self.value_black,
                "value_available": self.value_available,
                "score_margin_black": self.score_margin_black,
                "score_available": self.score_available,
                "ownership_black": self.ownership_black,
                "ownership_available": self.ownership_available,
            }
        if self.schema_version >= 3:
            payload.update(
                {
                    "source_kind": self.source_kind,
                    "task_id": self.task_id,
                    "termination": self.termination,
                    "game_seed": self.game_seed,
                    "black_agent_id": self.black_agent_id,
                    "white_agent_id": self.white_agent_id,
                    "publication_sha256": self.publication_sha256,
                    "feature_schema_id": self.feature_schema_id,
                    "search_config_sha256": self.search_config_sha256,
                    "search_budgets": self.search_budgets,
                    "root_values": self.root_values,
                    "root_score_margins": self.root_score_margins,
                    "temperatures": self.temperatures,
                    "search_seeds": self.search_seeds,
                    "root_noise_mask": self.root_noise_mask,
                    "search_metadata_mask": self.search_metadata_mask,
                    "root_score_mask": self.root_score_mask,
                }
            )
        return _sha256_json(payload)


@dataclass(frozen=True, slots=True)
class AnnotationRecord:
    schema_version: int
    game_id: str
    ply: int
    teacher_fingerprint: str
    policy_actions: tuple[int, ...]
    policy_values: tuple[float, ...]
    value: float
    value_available: bool
    score_margin: float
    score_available: bool
    ownership: tuple[float, ...]
    ownership_available: bool
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version not in SUPPORTED_RECORD_SCHEMA_VERSIONS:
            raise ValueError("unsupported annotation record schema")
        try:
            bytes.fromhex(self.game_id)
        except ValueError as error:
            raise ValueError("annotation identity is invalid") from error
        if len(self.game_id) != 64 or self.ply < 0 or not self.teacher_fingerprint:
            raise ValueError("annotation identity is invalid")
        if len(self.policy_actions) != len(self.policy_values):
            raise ValueError("annotation policy actions and values disagree")
        normalized_policy = tuple(float(value) for value in np.asarray(self.policy_values, dtype=np.float32))
        normalized_ownership = tuple(
            float(value) for value in np.asarray(self.ownership, dtype=np.float32)
        )
        object.__setattr__(self, "policy_values", normalized_policy)
        object.__setattr__(self, "value", float(np.float32(self.value)))
        object.__setattr__(self, "score_margin", float(np.float32(self.score_margin)))
        object.__setattr__(self, "ownership", normalized_ownership)
        if normalized_policy:
            if any(value < 0 or not math.isfinite(value) for value in normalized_policy):
                raise ValueError("annotation policy cannot contain negative mass")
            if abs(sum(normalized_policy) - 1.0) > 1e-5:
                raise ValueError("annotation policy must sum to one")
        if any(not 0 <= action < ACTION_SIZE for action in self.policy_actions):
            raise ValueError("annotation policy contains an invalid action")
        if len(self.ownership) != BOARD_AREA:
            raise ValueError("annotation ownership must contain exactly 361 values")
        if not all(
            math.isfinite(value)
            for value in (self.value, self.score_margin, *self.ownership)
        ):
            raise ValueError("annotation targets must be finite")
        object.__setattr__(self, "content_sha256", self.compute_content_sha256())

    def compute_content_sha256(self) -> str:
        return _sha256_json(
            {
                "schema_version": self.schema_version,
                "game_id": self.game_id,
                "ply": self.ply,
                "teacher_fingerprint": self.teacher_fingerprint,
                "policy_actions": self.policy_actions,
                "policy_values": self.policy_values,
                "value": self.value,
                "value_available": self.value_available,
                "score_margin": self.score_margin,
                "score_available": self.score_available,
                "ownership": self.ownership,
                "ownership_available": self.ownership_available,
            }
        )


@dataclass(frozen=True, slots=True)
class ImportEvent:
    kind: Literal["trajectory", "annotation", "reject"]
    game_id: str
    record: TrajectoryRecord | AnnotationRecord | None = None
    reason_code: str | None = None
    message: str | None = None
    asset_sha256: str | None = None
    member_path: str | None = None
    ordinal: int | None = None

    def __post_init__(self) -> None:
        try:
            bytes.fromhex(self.game_id)
        except ValueError as error:
            raise ValueError("import event game_id must be a SHA-256 hex string") from error
        if len(self.game_id) != 64:
            raise ValueError("import event game_id must be a SHA-256 hex string")
        if self.kind == "reject":
            if self.record is not None or not self.reason_code:
                raise ValueError("reject events require a reason and no record")
        elif self.kind == "trajectory":
            if not isinstance(self.record, TrajectoryRecord) or self.reason_code is not None:
                raise ValueError("trajectory events require a trajectory record")
        elif self.kind == "annotation":
            if not isinstance(self.record, AnnotationRecord) or self.reason_code is not None:
                raise ValueError("annotation events require an annotation record")
