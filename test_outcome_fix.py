#!/usr/bin/env python3
"""Test that outcome calculation is fixed."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import chess
from src.chess_env.board import ChessBoard


def test_white_wins():
    """Test white winning by checkmate."""
    # Scholar's mate - white wins quickly
    board = ChessBoard()
    moves = ['e2e4', 'e7e5', 'f1c4', 'b8c6', 'd1h5', 'g8f6', 'h5f7']  # Scholar's mate

    for move_uci in moves:
        move = chess.Move.from_uci(move_uci)
        board.make_move(move)

    assert board.is_game_over(), "Game should be over"
    result = board.get_result()
    print(f"White wins: outcome = {result} (should be 1.0)")
    assert result == 1.0, f"Expected 1.0 for white win, got {result}"


def test_black_wins():
    """Test black winning by checkmate."""
    # Fool's mate - black wins quickly
    board = ChessBoard()
    moves = ['f2f3', 'e7e5', 'g2g4', 'd8h4']  # Fool's mate

    for move_uci in moves:
        move = chess.Move.from_uci(move_uci)
        board.make_move(move)

    assert board.is_game_over(), "Game should be over"
    result = board.get_result()
    print(f"Black wins: outcome = {result} (should be -1.0)")
    assert result == -1.0, f"Expected -1.0 for black win, got {result}"


def test_draw():
    """Test draw by insufficient material."""
    # King vs King - automatic draw
    board = ChessBoard("k7/8/8/8/8/8/8/K7 w - - 0 1")

    assert board.is_game_over(), "Game should be over (insufficient material)"
    result = board.get_result()
    print(f"Draw: outcome = {result} (should be 0.0)")
    assert result == 0.0, f"Expected 0.0 for draw, got {result}"


if __name__ == '__main__':
    print("Testing outcome calculation fix...\n")

    test_white_wins()
    test_black_wins()
    test_draw()

    print("\n✓ All tests passed! Outcomes are now calculated correctly.")
