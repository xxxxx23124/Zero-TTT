"""Immutable Go game state."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from zero_ttt.config import GameConfig
from zero_ttt.game.rules import (
    ACTION_SIZE,
    BOARD_AREA,
    PASS_ACTION,
    AreaScore,
    Color,
    area_score,
    legal_actions,
    play_point,
)


@dataclass(frozen=True, slots=True)
class GameResult:
    score: AreaScore
    termination: str

    @property
    def winner(self) -> Color | None:
        return self.score.winner


@dataclass(frozen=True, slots=True)
class GameState:
    board: bytes
    to_play: Color
    move_number: int
    consecutive_passes: int
    komi_half_points: int
    max_moves: int
    history_length: int
    recent_boards: tuple[bytes, ...]
    recent_moves: tuple[int, ...]
    position_history: frozenset[bytes]

    @classmethod
    def new(cls, config: GameConfig) -> GameState:
        empty = bytes(BOARD_AREA)
        return cls(
            board=empty,
            to_play=Color.BLACK,
            move_number=0,
            consecutive_passes=0,
            komi_half_points=config.komi_half_points,
            max_moves=config.max_moves,
            history_length=config.history_length,
            recent_boards=(empty,),
            recent_moves=(),
            position_history=frozenset({empty}),
        )

    def legal_actions(self) -> tuple[bool, ...]:
        if self.is_terminal():
            return (False,) * ACTION_SIZE
        return legal_actions(self.board, self.to_play, self.position_history)

    def play(self, action: int) -> GameState:
        if self.is_terminal():
            raise ValueError("cannot play after the game has ended")
        if not 0 <= action < ACTION_SIZE:
            raise ValueError(f"action must be in [0, {ACTION_SIZE})")
        if action == PASS_ACTION:
            new_board = self.board
            new_history = self.position_history
            passes = self.consecutive_passes + 1
        else:
            new_board = play_point(
                self.board,
                self.to_play,
                action,
                self.position_history,
            )
            if new_board is None:
                raise ValueError(f"illegal action {action}")
            new_history = self.position_history | {new_board}
            passes = 0
        return GameState(
            board=new_board,
            to_play=self.to_play.opponent,
            move_number=self.move_number + 1,
            consecutive_passes=passes,
            komi_half_points=self.komi_half_points,
            max_moves=self.max_moves,
            history_length=self.history_length,
            recent_boards=(new_board, *self.recent_boards[: self.history_length - 1]),
            recent_moves=(action, *self.recent_moves[:1]),
            position_history=frozenset(new_history),
        )

    def play_many(self, actions: Iterable[int]) -> GameState:
        state = self
        for action in actions:
            state = state.play(action)
        return state

    def is_terminal(self) -> bool:
        return self.consecutive_passes >= 2 or self.move_number >= self.max_moves

    def termination_reason(self) -> str | None:
        if self.consecutive_passes >= 2:
            return "two_passes"
        if self.move_number >= self.max_moves:
            return "move_limit"
        return None

    def score(self) -> GameResult:
        return GameResult(
            score=area_score(self.board, self.komi_half_points),
            termination=self.termination_reason() or "adjudicated",
        )

    def identity(self) -> tuple[object, ...]:
        """Exact identity for rule-sensitive caches."""

        return (
            self.board,
            int(self.to_play),
            self.move_number,
            self.consecutive_passes,
            self.komi_half_points,
            self.max_moves,
            self.history_length,
            self.recent_boards,
            self.recent_moves,
            self.position_history,
        )
