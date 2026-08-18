from __future__ import annotations

import numpy as np
import pytest

from zero_ttt.config import load_config
from zero_ttt.game.features import encode_position
from zero_ttt.game.rules import (
    BOARD_AREA,
    BOARD_SIZE,
    PASS_ACTION,
    Color,
    area_score,
    legal_actions,
    play_point,
)
from zero_ttt.game.state import GameState
from zero_ttt.game.symmetry import (
    augment_sample,
    transform_action_vector,
    transform_point,
)


def point(row: int, col: int) -> int:
    return row * BOARD_SIZE + col


def board_with(stones: dict[int, Color]) -> bytes:
    board = bytearray(BOARD_AREA)
    for location, color in stones.items():
        board[location] = int(color)
    return bytes(board)


def test_capture_and_suicide() -> None:
    center = point(1, 1)
    capture_at = point(2, 1)
    board = board_with(
        {
            center: Color.WHITE,
            point(1, 0): Color.BLACK,
            point(0, 1): Color.BLACK,
            point(1, 2): Color.BLACK,
        }
    )
    result = play_point(board, Color.BLACK, capture_at, {board})
    assert result is not None
    assert result[center] == 0
    assert result[capture_at] == int(Color.BLACK)

    suicide_board = board_with(
        {
            point(1, 0): Color.BLACK,
            point(0, 1): Color.BLACK,
            point(1, 2): Color.BLACK,
            point(2, 1): Color.BLACK,
        }
    )
    assert play_point(suicide_board, Color.WHITE, center, {suicide_board}) is None


def test_one_move_can_capture_multiple_groups() -> None:
    action = point(1, 1)
    left = point(1, 0)
    above = point(0, 1)
    board = board_with(
        {
            left: Color.WHITE,
            above: Color.WHITE,
            point(0, 0): Color.BLACK,
            point(2, 0): Color.BLACK,
            point(0, 2): Color.BLACK,
        }
    )
    result = play_point(board, Color.BLACK, action, {board})
    assert result is not None
    assert result[left] == 0
    assert result[above] == 0


def test_positional_superko_forbids_immediate_ko_repetition() -> None:
    before = board_with(
        {
            point(0, 1): Color.WHITE,
            point(1, 0): Color.WHITE,
            point(0, 2): Color.BLACK,
            point(1, 1): Color.BLACK,
        }
    )
    after = play_point(before, Color.BLACK, point(0, 0), {before})
    assert after is not None
    assert play_point(after, Color.WHITE, point(0, 1), {after}) == before
    assert play_point(after, Color.WHITE, point(0, 1), {before, after}) is None


def test_positional_superko_uses_exact_board_and_pass_is_always_legal() -> None:
    empty = bytes(BOARD_AREA)
    once = play_point(empty, Color.BLACK, point(3, 3), {empty})
    assert once is not None
    assert play_point(empty, Color.BLACK, point(3, 3), {empty, once}) is None
    mask = legal_actions(empty, Color.BLACK, {empty, once})
    assert not mask[point(3, 3)]
    assert mask[PASS_ACTION]


def test_two_passes_end_game_and_komi_scores() -> None:
    config = load_config("configs/test.toml")
    state = GameState.new(config.game).play(PASS_ACTION).play(PASS_ACTION)
    assert state.is_terminal()
    assert state.termination_reason() == "two_passes"
    result = state.score()
    assert result.score.margin_half_points == -15
    assert result.winner is Color.WHITE


def test_move_limit_forces_terminal_state() -> None:
    config = load_config("configs/test.toml")
    state = GameState.new(config.game)
    for _ in range(config.game.max_moves):
        if state.is_terminal():
            break
        legal = state.legal_actions()
        action = next(index for index, allowed in enumerate(legal[:-1]) if allowed)
        state = state.play(action)
    assert state.is_terminal()
    assert state.termination_reason() == "move_limit"


def test_area_score_ownership_is_signed() -> None:
    board = board_with({point(0, 0): Color.BLACK, point(18, 18): Color.WHITE})
    score = area_score(board, 15)
    ownership = np.frombuffer(score.ownership, dtype=np.int8)
    assert ownership[point(0, 0)] == 1
    assert ownership[point(18, 18)] == -1
    assert ownership[point(9, 9)] == 0


def test_feature_schema_and_current_player_view() -> None:
    config = load_config("configs/test.toml")
    state = GameState.new(config.game).play(point(3, 3))
    features = encode_position(state)
    assert features.board.shape == (25, 19, 19)
    assert features.global_features.shape == (5,)
    assert features.legal.shape == (362,)
    assert features.board[1, 3, 3] == 1.0  # black is the opponent from White's view
    assert features.global_features[4] == 1.0


def test_d4_action_and_sample_transforms_are_consistent() -> None:
    config = load_config("configs/test.toml")
    features = encode_position(GameState.new(config.game))
    policy = np.arange(362, dtype=np.float32)
    ownership = np.arange(361, dtype=np.float32)
    for symmetry in range(8):
        transformed = augment_sample(features, policy, ownership, symmetry)
        expected_action = transform_point(point(2, 5), symmetry)
        assert transformed.policy[expected_action] == policy[point(2, 5)]
        assert transformed.ownership[expected_action] == ownership[point(2, 5)]
        assert transformed.policy[PASS_ACTION] == policy[PASS_ACTION]
        assert np.array_equal(
            transformed.features.legal,
            transform_action_vector(features.legal, symmetry),
        )


def test_illegal_action_raises() -> None:
    config = load_config("configs/test.toml")
    state = GameState.new(config.game).play(point(0, 0))
    with pytest.raises(ValueError, match="illegal action"):
        state.play(point(0, 0))
