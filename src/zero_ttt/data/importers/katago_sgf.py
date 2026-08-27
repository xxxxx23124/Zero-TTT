"""Streaming importer for KataGo archives containing line-delimited SGFs."""

# The dependency is supplied by the Docker runtime.
# pyright: reportMissingImports=false

from __future__ import annotations

import re
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from sgfmill import sgf

from zero_ttt.config import GameConfig
from zero_ttt.data.manifest import ManifestAsset, SourceManifest, sha256_file
from zero_ttt.data.records import (
    ImportEvent,
    TrajectoryRecord,
    stable_game_id,
)
from zero_ttt.game.rules import BOARD_AREA, BOARD_SIZE, PASS_ACTION
from zero_ttt.game.state import GameState
from zero_ttt.versioning import RECORD_SCHEMA

_START_TURN = re.compile(r"(?:^|,)startTurnIdx=(\d+)(?:,|$)")
_MODE = re.compile(r"(?:^|,)mode=([^,]+)(?:,|$)")
_INITIAL_POSITION = re.compile(r"(?:^|,)usedInitialPosition=([^,]+)(?:,|$)")
TRAJECTORY_MAX_MOVES = 722


class ImportRejected(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class ParsedKataGoGame:
    board_size: int
    has_setup: bool
    handicap: int
    game_type: str
    comment: str
    komi: float
    rules: str
    result: str
    moves: tuple[tuple[str, int], ...]


def _has_variation_or_collection(data: bytes) -> bool:
    depth = 0
    in_value = False
    escaped = False
    roots = 0
    for byte in data:
        character = chr(byte)
        if in_value:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == "]":
                in_value = False
            continue
        if character == "[":
            in_value = True
        elif character == "(":
            if depth == 0:
                roots += 1
            depth += 1
            if depth > 1:
                return True
        elif character == ")":
            depth -= 1
            if depth < 0:
                return True
    return depth != 0 or roots != 1 or in_value


def _root_text(root, property_name: str, default: str = "") -> str:
    if not root.has_property(property_name):
        return default
    value = root.get(property_name)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="strict")
    return str(value)


def _parse_root_comment(comment: str) -> tuple[int, str, bool]:
    start_match = _START_TURN.search(comment)
    mode_match = _MODE.search(comment)
    initial_match = _INITIAL_POSITION.search(comment)
    start = int(start_match.group(1)) if start_match else 0
    mode = mode_match.group(1) if mode_match else "normal"
    used_initial = initial_match is not None and initial_match.group(1) != "0"
    return start, mode, used_initial


def _parse_result(result: str) -> tuple[float, int | None, bool]:
    normalized = result.strip().upper()
    if normalized in {"0", "DRAW", "JIGO"}:
        return 0.0, 0, False
    if not normalized.startswith(("B+", "W+")):
        raise ImportRejected("unsupported_result", f"unsupported result {result!r}")
    winner_black = normalized.startswith("B+")
    detail = normalized[2:]
    value = 1.0 if winner_black else -1.0
    if detail in {"R", "RESIGN", "T", "TIME", "F", "FORFEIT"}:
        return value, None, True
    try:
        points = float(detail)
    except ValueError as error:
        raise ImportRejected("unsupported_result", f"unsupported result {result!r}") from error
    half_points = round(points * 2)
    if abs(points * 2 - half_points) > 1e-6:
        raise ImportRejected("unsupported_result", "result is not an integer number of half-points")
    return value, half_points if winner_black else -half_points, False


def _rules_are_exact(rules: str) -> bool:
    compact = rules.replace(" ", "").upper()
    return (
        "KOPOSITIONAL" in compact
        and "SCOREAREA" in compact
        and "TAXNONE" in compact
        and "SUI1" in compact
        and "BUTTON1" not in compact
    )


def _move_to_action(move: tuple[int, int] | None) -> int:
    if move is None:
        return PASS_ACTION
    row, col = move
    if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
        raise ImportRejected("invalid_move", "SGF move lies outside a 19x19 board")
    return row * BOARD_SIZE + col


def _parse_external_sgf(raw: bytes) -> ParsedKataGoGame:
    """Contain sgfmill's typed-property exceptions at the format boundary."""

    if _has_variation_or_collection(raw):
        raise ImportRejected("variation", "only one unbranched SGF game is supported")
    try:
        game = sgf.Sgf_game.from_bytes(raw)
        root = game.get_root()
        board_size = game.get_size()
        has_setup = any(root.has_property(name) for name in ("AB", "AW", "AE"))
        handicap = int(root.get("HA")) if root.has_property("HA") else 0
        game_type = _root_text(root, "GM", "1")
        comment = _root_text(root, "C")
        komi = float(root.get("KM")) if root.has_property("KM") else 0.0
        rules = _root_text(root, "RU")
        result = _root_text(root, "RE")
        moves = []
        for node in game.get_main_sequence()[1:]:
            color, move = node.get_move()
            if color is None:
                raise ImportRejected("non_move_node", "non-root nodes must contain a move")
            moves.append((color, _move_to_action(move)))
    except ImportRejected:
        raise
    except (TypeError, ValueError, UnicodeError) as error:
        raise ImportRejected("invalid_sgf", str(error)) from error
    return ParsedKataGoGame(
        board_size=board_size,
        has_setup=has_setup,
        handicap=handicap,
        game_type=game_type,
        comment=comment,
        komi=komi,
        rules=rules,
        result=result,
        moves=tuple(moves),
    )


def _validate_game_header(parsed: ParsedKataGoGame) -> None:
    if parsed.board_size != BOARD_SIZE:
        raise ImportRejected("board_size", "only 19x19 games are supported")
    if parsed.has_setup:
        raise ImportRejected("setup_stones", "setup stones are not supported")
    if parsed.handicap:
        raise ImportRejected("handicap", "handicap games are not supported")
    if parsed.game_type != "1":
        raise ImportRejected("game_type", "only Go SGFs are supported")


def _training_range(parsed: ParsedKataGoGame) -> tuple[int, int, list[int]]:
    trainable_start, mode, used_initial = _parse_root_comment(parsed.comment)
    if mode != "normal":
        raise ImportRejected("mode", f"unsupported KataGo mode {mode!r}")
    if used_initial:
        raise ImportRejected("initial_position", "non-empty initial positions are not supported")
    komi_half_points = round(parsed.komi * 2)
    if abs(parsed.komi * 2 - komi_half_points) > 1e-6:
        raise ImportRejected("komi", "komi must be an integer number of half-points")
    moves = _alternating_moves(parsed.moves)
    if not moves or trainable_start >= len(moves):
        raise ImportRejected("no_trainable_positions", "game has no trainable positions")
    return trainable_start, komi_half_points, moves


def _alternating_moves(parsed_moves: tuple[tuple[str, int], ...]) -> list[int]:
    moves = []
    expected_color = "b"
    for color, action in parsed_moves:
        if color != expected_color:
            raise ImportRejected("turn_order", "moves must alternate starting with black")
        moves.append(action)
        expected_color = "w" if expected_color == "b" else "b"
    return moves


def _replay_moves(moves: list[int], komi_half_points: int) -> GameState:
    state = GameState.new(
        GameConfig(
            board_size=BOARD_SIZE,
            komi_half_points=komi_half_points,
            max_moves=TRAJECTORY_MAX_MOVES,
            history_length=8,
        )
    )
    for ply, action in enumerate(moves):
        if state.is_terminal():
            raise ImportRejected(
                "cleanup_phase",
                f"move {ply} occurs after the local game termination boundary",
            )
        try:
            state = state.play(action)
        except ValueError as error:
            raise ImportRejected("illegal_move", f"illegal move at ply {ply}: {error}") from error
    return state


@dataclass(frozen=True, slots=True)
class _ExternalTargets:
    value_black: float
    value_available: bool
    score_margin_black: float
    score_available: bool
    ownership_black: tuple[float, ...]
    ownership_available: bool
    resigned: bool


def _external_targets(parsed: ParsedKataGoGame, state: GameState) -> _ExternalTargets:
    value_black, declared_margin_half, resigned = _parse_result(parsed.result)
    exact_rules = _rules_are_exact(parsed.rules)
    if exact_rules and resigned:
        return _ExternalTargets(value_black, True, 0.0, False, (0.0,) * BOARD_AREA, False, resigned)
    if exact_rules and declared_margin_half is not None and state.is_terminal():
        local = state.score().score
        if local.margin_half_points == declared_margin_half:
            ownership = tuple(-1.0 if value == 255 else float(value) for value in local.ownership)
            return _ExternalTargets(
                value_black,
                True,
                local.margin_half_points / 2.0,
                True,
                ownership,
                True,
                resigned,
            )
    return _ExternalTargets(value_black, False, 0.0, False, (0.0,) * BOARD_AREA, False, resigned)


def _build_trajectory(
    parsed: ParsedKataGoGame,
    manifest: SourceManifest,
    asset: ManifestAsset,
    member_path: str,
    ordinal: int,
) -> TrajectoryRecord:
    _validate_game_header(parsed)
    trainable_start, komi_half_points, moves = _training_range(parsed)
    state = _replay_moves(moves, komi_half_points)
    targets = _external_targets(parsed, state)
    trainable_count = len(moves) - trainable_start
    policy_actions = tuple(moves[trainable_start:])
    policy_values = (1.0,) * trainable_count
    policy_offsets = tuple(range(trainable_count + 1))
    game_id = stable_game_id(
        manifest.dataset_id,
        asset.sha256,
        member_path,
        ordinal,
    )
    return TrajectoryRecord(
        schema_version=RECORD_SCHEMA.current,
        game_id=game_id,
        content_sha256="",
        dataset_id=manifest.dataset_id,
        asset_sha256=asset.sha256,
        member_path=member_path,
        ordinal=ordinal,
        rules=parsed.rules,
        komi_half_points=komi_half_points,
        max_moves=TRAJECTORY_MAX_MOVES,
        moves=tuple(moves),
        trainable_start_ply=trainable_start,
        policy_row_offsets=policy_offsets,
        policy_actions=policy_actions,
        policy_values=policy_values,
        value_black=targets.value_black,
        value_available=targets.value_available,
        score_margin_black=targets.score_margin_black,
        score_available=targets.score_available,
        ownership_black=targets.ownership_black,
        ownership_available=targets.ownership_available,
        termination=(
            state.termination_reason() or ("resignation" if targets.resigned else "external")
        ),
    )


def _parse_game(
    raw: bytes,
    manifest: SourceManifest,
    asset: ManifestAsset,
    member_path: str,
    ordinal: int,
) -> TrajectoryRecord:
    return _build_trajectory(
        _parse_external_sgf(raw),
        manifest,
        asset,
        member_path,
        ordinal,
    )


class KataGoSgfImporter:
    """Yield records/rejections without knowing anything about persistence."""

    source_type = "katago-g170-sgfs-zip"

    def import_asset(
        self,
        manifest: SourceManifest,
        asset: ManifestAsset,
        source_root: str | Path,
    ) -> Iterator[ImportEvent]:
        if manifest.source_type != self.source_type:
            raise ValueError(f"unsupported source_type {manifest.source_type!r}")
        path = Path(source_root) / asset.relative_path
        if path.stat().st_size != asset.size_bytes or sha256_file(path) != asset.sha256:
            raise ValueError(f"asset integrity check failed: {asset.relative_path}")
        with zipfile.ZipFile(path) as archive:
            members = sorted(name for name in archive.namelist() if name.lower().endswith(".sgfs"))
            for member_path in members:
                with archive.open(member_path) as stream:
                    for ordinal, raw in enumerate(stream):
                        raw = raw.strip()
                        if not raw:
                            continue
                        game_id = stable_game_id(
                            manifest.dataset_id,
                            asset.sha256,
                            member_path,
                            ordinal,
                        )
                        try:
                            record = _parse_game(
                                raw,
                                manifest,
                                asset,
                                member_path,
                                ordinal,
                            )
                        except ImportRejected as error:
                            yield ImportEvent(
                                kind="reject",
                                game_id=game_id,
                                reason_code=error.code,
                                message=str(error),
                                asset_sha256=asset.sha256,
                                member_path=member_path,
                                ordinal=ordinal,
                            )
                            continue
                        yield ImportEvent(
                            kind="trajectory",
                            game_id=record.game_id,
                            record=record,
                            asset_sha256=asset.sha256,
                            member_path=member_path,
                            ordinal=ordinal,
                        )
