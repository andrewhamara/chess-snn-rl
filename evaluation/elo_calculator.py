"""Elo rating system implementation."""

import math
from typing import Dict, List, Tuple


class EloCalculator:
    """
    Standard Elo rating system for chess.

    Expected score: E(A) = 1 / (1 + 10^((R_B - R_A)/400))
    Rating update: R'_A = R_A + K * (S - E(A))
    """

    def __init__(self, k_factor: float = 32, initial_rating: float = 1000):
        """
        Initialize Elo calculator.

        Args:
            k_factor: K-factor for rating updates (typically 16-32)
            initial_rating: Default initial rating
        """
        self.k_factor = k_factor
        self.initial_rating = initial_rating

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """
        Calculate expected score for player A against player B.

        Args:
            rating_a: Elo rating of player A
            rating_b: Elo rating of player B

        Returns:
            Expected score for player A (between 0 and 1)
        """
        return 1.0 / (1.0 + math.pow(10, (rating_b - rating_a) / 400.0))

    def update_rating(self, rating: float, expected: float, actual: float) -> float:
        """
        Update rating based on game result.

        Args:
            rating: Current rating
            expected: Expected score
            actual: Actual score (1.0 for win, 0.5 for draw, 0.0 for loss)

        Returns:
            Updated rating
        """
        return rating + self.k_factor * (actual - expected)

    def update_ratings(self, rating_a: float, rating_b: float,
                      result: float) -> Tuple[float, float]:
        """
        Update both players' ratings after a game.

        Args:
            rating_a: Player A's rating
            rating_b: Player B's rating
            result: Result from A's perspective (1.0 win, 0.5 draw, 0.0 loss)

        Returns:
            (new_rating_a, new_rating_b)
        """
        expected_a = self.expected_score(rating_a, rating_b)
        expected_b = 1.0 - expected_a

        new_rating_a = self.update_rating(rating_a, expected_a, result)
        new_rating_b = self.update_rating(rating_b, expected_b, 1.0 - result)

        return new_rating_a, new_rating_b

    def calculate_performance_rating(self, results: List[Tuple[float, float]]) -> float:
        """
        Calculate performance rating from a set of games.

        Args:
            results: List of (opponent_rating, game_result) tuples

        Returns:
            Performance rating
        """
        if not results:
            return self.initial_rating

        total_score = sum(result for _, result in results)
        avg_opponent_rating = sum(rating for rating, _ in results) / len(results)

        # Performance percentage
        performance_pct = total_score / len(results)

        # Handle edge cases
        if performance_pct >= 1.0:
            performance_pct = 0.99
        elif performance_pct <= 0.0:
            performance_pct = 0.01

        # Calculate performance rating
        # R_perf = R_avg + 400 * log10(p / (1-p))
        performance_rating = avg_opponent_rating + 400 * math.log10(
            performance_pct / (1.0 - performance_pct)
        )

        return performance_rating

    def estimate_win_probability(self, rating_a: float, rating_b: float) -> Dict[str, float]:
        """
        Estimate win/draw/loss probabilities.

        This uses a simplified model based on expected score.

        Args:
            rating_a: Player A's rating
            rating_b: Player B's rating

        Returns:
            Dictionary with win/draw/loss probabilities
        """
        expected_a = self.expected_score(rating_a, rating_b)

        # Simplified model: assume ~30% draw rate at equal strength
        draw_rate = 0.30 * (1.0 - abs(expected_a - 0.5) * 2)

        win_prob = expected_a * (1.0 - draw_rate)
        loss_prob = (1.0 - expected_a) * (1.0 - draw_rate)

        return {
            'win': win_prob,
            'draw': draw_rate,
            'loss': loss_prob
        }

    def rating_difference_to_score(self, rating_diff: float) -> float:
        """
        Convert rating difference to expected score percentage.

        Args:
            rating_diff: Rating difference (own - opponent)

        Returns:
            Expected score percentage (0-100)
        """
        expected = self.expected_score(self.initial_rating + rating_diff, self.initial_rating)
        return expected * 100

    def score_to_rating_difference(self, score_pct: float) -> float:
        """
        Convert score percentage to rating difference.

        Args:
            score_pct: Score percentage (0-100)

        Returns:
            Approximate rating difference
        """
        score = score_pct / 100.0

        # Handle edge cases
        if score >= 1.0:
            score = 0.99
        elif score <= 0.0:
            score = 0.01

        # R_A - R_B = 400 * log10(p / (1-p))
        return 400 * math.log10(score / (1.0 - score))
