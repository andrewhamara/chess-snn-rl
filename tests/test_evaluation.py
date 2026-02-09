"""Tests for evaluation system."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.elo_calculator import EloCalculator
from src.evaluation.opponents import RandomPlayer, GreedyMaterialPlayer, MinimaxPlayer
import chess


def test_elo_expected_score():
    """Test Elo expected score calculation."""
    calc = EloCalculator()

    # Equal ratings
    expected = calc.expected_score(1000, 1000)
    assert abs(expected - 0.5) < 0.01

    # Higher rating
    expected = calc.expected_score(1200, 1000)
    assert expected > 0.5

    # Lower rating
    expected = calc.expected_score(1000, 1200)
    assert expected < 0.5

    print("✓ Elo expected score test passed")


def test_elo_update():
    """Test Elo rating update."""
    calc = EloCalculator(k_factor=32)

    # Win when expected
    new_rating = calc.update_rating(1200, 0.76, 1.0)
    assert new_rating > 1200  # Should increase

    # Loss when expected to win
    new_rating = calc.update_rating(1200, 0.76, 0.0)
    assert new_rating < 1200  # Should decrease significantly

    print("✓ Elo update test passed")


def test_random_player():
    """Test random player."""
    player = RandomPlayer()
    board = chess.Board()

    move = player.select_move(board)
    assert move in board.legal_moves
    print("✓ Random player test passed")


def test_greedy_player():
    """Test greedy material player."""
    player = GreedyMaterialPlayer()
    board = chess.Board()

    # Setup position with clear capture
    board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")

    move = player.select_move(board)
    assert move in board.legal_moves
    print("✓ Greedy player test passed")


def test_minimax_player():
    """Test minimax player."""
    player = MinimaxPlayer(depth=2)
    board = chess.Board()

    move = player.select_move(board)
    assert move in board.legal_moves
    print("✓ Minimax player test passed")


def test_performance_rating():
    """Test performance rating calculation."""
    calc = EloCalculator()

    # Perfect score against 1000-rated opponents
    results = [(1000, 1.0) for _ in range(10)]
    perf_rating = calc.calculate_performance_rating(results)
    assert perf_rating > 1000

    # 50% score
    results = [(1000, 1.0 if i % 2 == 0 else 0.0) for i in range(10)]
    perf_rating = calc.calculate_performance_rating(results)
    assert abs(perf_rating - 1000) < 50

    print("✓ Performance rating test passed")


if __name__ == '__main__':
    test_elo_expected_score()
    test_elo_update()
    test_random_player()
    test_greedy_player()
    test_minimax_player()
    test_performance_rating()
    print("\nAll evaluation tests passed!")
