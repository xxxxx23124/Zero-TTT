"""No-suicide Tromp-Taylor rule primitives for a fixed 19x19 board."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache
from typing import Collection, Sequence


BOARD_SIZE = 19
BOARD_AREA = BOARD_SIZE * BOARD_SIZE
PASS_ACTION = BOARD_AREA
ACTION_SIZE = BOARD_AREA + 1
EMPTY = 0


class Color(IntEnum):
    BLACK = 1
    WHITE = 2

    @property
    def opponent(self) -> "Color":
        return Color.WHITE if self is Color.BLACK else Color.BLACK


def _build_neighbors() -> tuple[tuple[int, ...], ...]:
    result: list[tuple[int, ...]] = []
    for point in range(BOARD_AREA):
        row, col = divmod(point, BOARD_SIZE)
        adjacent: list[int] = []
        if row:
            adjacent.append(point - BOARD_SIZE)
        if row + 1 < BOARD_SIZE:
            adjacent.append(point + BOARD_SIZE)
        if col:
            adjacent.append(point - 1)
        if col + 1 < BOARD_SIZE:
            adjacent.append(point + 1)
        result.append(tuple(adjacent))
    return tuple(result)


NEIGHBORS = _build_neighbors()


@dataclass(frozen=True, slots=True)
class BoardAnalysis:
    group_at: tuple[int, ...]
    group_colors: tuple[int, ...]
    group_stones: tuple[tuple[int, ...], ...]
    group_liberties: tuple[tuple[int, ...], ...]


def _collect_group(board: Sequence[int], start: int) -> tuple[set[int], set[int]]:
    color = board[start]
    stones = {start}
    liberties: set[int] = set()
    stack = [start]
    while stack:
        point = stack.pop()
        for adjacent in NEIGHBORS[point]:
            value = board[adjacent]
            if value == EMPTY:
                liberties.add(adjacent)
            elif value == color and adjacent not in stones:
                stones.add(adjacent)
                stack.append(adjacent)
    return stones, liberties


@lru_cache(maxsize=65_536)
def analyze_board(board: bytes) -> BoardAnalysis:
    if len(board) != BOARD_AREA:
        raise ValueError(f"board must contain {BOARD_AREA} points")
    group_at = [-1] * BOARD_AREA
    colors: list[int] = []
    groups: list[tuple[int, ...]] = []
    liberties: list[tuple[int, ...]] = []
    for point, value in enumerate(board):
        if value == EMPTY or group_at[point] >= 0:
            continue
        stones, group_liberties = _collect_group(board, point)
        group_id = len(groups)
        for stone in stones:
            group_at[stone] = group_id
        colors.append(value)
        groups.append(tuple(sorted(stones)))
        liberties.append(tuple(sorted(group_liberties)))
    return BoardAnalysis(
        group_at=tuple(group_at),
        group_colors=tuple(colors),
        group_stones=tuple(groups),
        group_liberties=tuple(liberties),
    )


def play_point(
    board: bytes,
    color: Color,
    action: int,
    position_history: Collection[bytes],
) -> bytes | None:
    """Return the resulting board, or None when the point move is illegal."""

    if not 0 <= action < BOARD_AREA or board[action] != EMPTY:
        return None
    analysis = analyze_board(board)
    mutable = bytearray(board)
    mutable[action] = int(color)
    captured_groups: set[int] = set()
    for adjacent in NEIGHBORS[action]:
        if board[adjacent] != int(color.opponent):
            continue
        group_id = analysis.group_at[adjacent]
        if group_id >= 0 and len(analysis.group_liberties[group_id]) == 1:
            captured_groups.add(group_id)
    for group_id in captured_groups:
        for stone in analysis.group_stones[group_id]:
            mutable[stone] = EMPTY
    stones, liberties = _collect_group(mutable, action)
    del stones
    if not liberties:
        return None
    result = bytes(mutable)
    if result in position_history:
        return None
    return result


def legal_actions(
    board: bytes,
    color: Color,
    position_history: Collection[bytes],
) -> tuple[bool, ...]:
    legal = [False] * ACTION_SIZE
    for action, value in enumerate(board):
        if value == EMPTY:
            legal[action] = play_point(board, color, action, position_history) is not None
    legal[PASS_ACTION] = True
    return tuple(legal)


@dataclass(frozen=True, slots=True)
class AreaScore:
    black_area: int
    white_area: int
    black_score_half_points: int
    white_score_half_points: int
    margin_half_points: int
    ownership: bytes

    @property
    def winner(self) -> Color | None:
        if self.margin_half_points > 0:
            return Color.BLACK
        if self.margin_half_points < 0:
            return Color.WHITE
        return None


def area_score(board: bytes, komi_half_points: int) -> AreaScore:
    """Score stones and single-color reachable empty regions."""

    if len(board) != BOARD_AREA:
        raise ValueError(f"board must contain {BOARD_AREA} points")
    black_area = sum(value == int(Color.BLACK) for value in board)
    white_area = sum(value == int(Color.WHITE) for value in board)
    ownership = bytearray(BOARD_AREA)
    for point, value in enumerate(board):
        if value == int(Color.BLACK):
            ownership[point] = 1
        elif value == int(Color.WHITE):
            ownership[point] = 255  # signed -1 represented in bytes
    visited: set[int] = set()
    for start, value in enumerate(board):
        if value != EMPTY or start in visited:
            continue
        region = {start}
        borders: set[int] = set()
        stack = [start]
        visited.add(start)
        while stack:
            point = stack.pop()
            for adjacent in NEIGHBORS[point]:
                neighbor = board[adjacent]
                if neighbor == EMPTY and adjacent not in visited:
                    visited.add(adjacent)
                    region.add(adjacent)
                    stack.append(adjacent)
                elif neighbor != EMPTY:
                    borders.add(neighbor)
        if borders == {int(Color.BLACK)}:
            black_area += len(region)
            for point in region:
                ownership[point] = 1
        elif borders == {int(Color.WHITE)}:
            white_area += len(region)
            for point in region:
                ownership[point] = 255
    black_half = 2 * black_area
    white_half = 2 * white_area + komi_half_points
    return AreaScore(
        black_area=black_area,
        white_area=white_area,
        black_score_half_points=black_half,
        white_score_half_points=white_half,
        margin_half_points=black_half - white_half,
        ownership=bytes(ownership),
    )
