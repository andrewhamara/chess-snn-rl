"""Evaluation system for benchmarking SNN agent."""

import torch
import chess
from typing import Dict, List, Tuple
from tqdm import tqdm

from src.chess_env.board import ChessBoard
from src.chess_env.encoder import BoardEncoder
from src.chess_env.move_encoding import MoveEncoder
from .elo_calculator import EloCalculator
from .opponents import BaseOpponent, RandomPlayer, GreedyMaterialPlayer, MinimaxPlayer
from config.eval_config import EvalConfig


class Evaluator:
    """
    Evaluates SNN agent against baseline opponents and calculates Elo.
    """

    def __init__(self, model, device: str = "cpu", config: EvalConfig = None):
        """
        Initialize evaluator.

        Args:
            model: ChessSNN model to evaluate
            device: Device to run on
            config: Evaluation configuration
        """
        self.model = model
        self.device = device
        self.config = config if config else EvalConfig()

        self.board_encoder = BoardEncoder()
        self.move_encoder = MoveEncoder()
        self.elo_calculator = EloCalculator(
            k_factor=self.config.K_FACTOR,
            initial_rating=self.config.INITIAL_ELO
        )

        self.model.eval()

        # Initialize opponents
        self.opponents = {
            'random': RandomPlayer(),
            'greedy_material': GreedyMaterialPlayer(),
            'minimax_depth2': MinimaxPlayer(depth=2)
        }

    def play_game(self, opponent: BaseOpponent, agent_plays_white: bool = True,
                  max_moves: int = 200) -> Tuple[float, str, int]:
        """
        Play one game against an opponent.

        Args:
            opponent: Opponent to play against
            agent_plays_white: Whether agent plays white
            max_moves: Maximum moves before draw

        Returns:
            (result, termination, move_count)
            result: 1.0 for agent win, 0.5 for draw, 0.0 for loss
        """
        board = ChessBoard()
        move_count = 0

        with torch.no_grad():
            while not board.is_game_over() and move_count < max_moves:
                agent_turn = (board.turn == chess.WHITE) == agent_plays_white

                if agent_turn:
                    # Agent's turn
                    legal_moves = board.get_legal_moves()
                    state = self.board_encoder.encode(board.board, legal_moves)
                    legal_mask = torch.from_numpy(
                        self.move_encoder.get_legal_action_mask(board.board)
                    )

                    state = state.to(self.device)
                    legal_mask = legal_mask.to(self.device)

                    # Select action deterministically for evaluation
                    action, _, _ = self.model.select_action(
                        state, legal_mask, temperature=0.1, deterministic=True
                    )

                    move = self.move_encoder.action_to_move(action, board.board)

                    if move is None or move not in legal_moves:
                        # Illegal move - agent loses
                        return (0.0, "illegal_move", move_count)

                    board.make_move(move)
                else:
                    # Opponent's turn
                    move = opponent.select_move(board.board)
                    board.make_move(move)

                move_count += 1

        # Game ended
        if move_count >= max_moves:
            return (0.5, "max_moves", move_count)

        termination, outcome = board.get_outcome()

        # Convert outcome to agent's perspective
        if not agent_plays_white:
            outcome = -outcome

        # Convert to win/draw/loss
        if outcome > 0.5:
            result = 1.0  # Win
        elif outcome < -0.5:
            result = 0.0  # Loss
        else:
            result = 0.5  # Draw

        return (result, termination, move_count)

    def evaluate_against_opponent(self, opponent_name: str,
                                  num_games: int = 100) -> Dict:
        """
        Evaluate against one opponent.

        Args:
            opponent_name: Name of opponent ('random', 'greedy_material', 'minimax_depth2')
            num_games: Number of games to play (half as white, half as black)

        Returns:
            Evaluation results dictionary
        """
        if opponent_name not in self.opponents:
            raise ValueError(f"Unknown opponent: {opponent_name}")

        opponent = self.opponents[opponent_name]
        print(f"\nEvaluating against {opponent.name} ({num_games} games)...")

        results = []
        terminations = []
        move_counts = []

        # Play games (alternating colors)
        for i in tqdm(range(num_games)):
            agent_plays_white = (i % 2 == 0)
            result, termination, move_count = self.play_game(
                opponent, agent_plays_white, self.config.MAX_GAME_LENGTH
            )

            results.append(result)
            terminations.append(termination)
            move_counts.append(move_count)

            opponent.reset()

        # Calculate statistics
        wins = sum(1 for r in results if r == 1.0)
        draws = sum(1 for r in results if r == 0.5)
        losses = sum(1 for r in results if r == 0.0)

        win_rate = wins / num_games
        draw_rate = draws / num_games
        loss_rate = losses / num_games

        # Calculate performance rating
        # Assume opponent ratings
        opponent_ratings = {
            'random': 400,
            'greedy_material': 800,
            'minimax_depth2': 1200
        }

        opponent_rating = opponent_ratings.get(opponent_name, 1000)
        performance_results = [(opponent_rating, r) for r in results]
        performance_rating = self.elo_calculator.calculate_performance_rating(
            performance_results
        )

        # Count termination types
        termination_counts = {}
        for term in terminations:
            termination_counts[term] = termination_counts.get(term, 0) + 1

        eval_results = {
            'opponent': opponent.name,
            'opponent_rating': opponent_rating,
            'num_games': num_games,
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'win_rate': win_rate,
            'draw_rate': draw_rate,
            'loss_rate': loss_rate,
            'score': (wins + 0.5 * draws) / num_games,
            'performance_rating': performance_rating,
            'avg_moves': sum(move_counts) / num_games,
            'terminations': termination_counts
        }

        print(f"Results vs {opponent.name}:")
        print(f"  Win rate: {win_rate:.2%}")
        print(f"  Draw rate: {draw_rate:.2%}")
        print(f"  Loss rate: {loss_rate:.2%}")
        print(f"  Performance rating: {performance_rating:.0f}")

        return eval_results

    def full_evaluation(self) -> Dict:
        """
        Run full evaluation against all opponents.

        Returns:
            Complete evaluation results
        """
        print("=== Starting Full Evaluation ===")

        all_results = {}

        for opponent_name in self.config.OPPONENTS:
            results = self.evaluate_against_opponent(
                opponent_name,
                self.config.GAMES_PER_OPPONENT
            )
            all_results[opponent_name] = results

        # Calculate average performance rating
        performance_ratings = [
            r['performance_rating'] for r in all_results.values()
        ]
        avg_performance_rating = sum(performance_ratings) / len(performance_ratings)

        # Overall statistics
        total_games = sum(r['num_games'] for r in all_results.values())
        total_wins = sum(r['wins'] for r in all_results.values())
        total_draws = sum(r['draws'] for r in all_results.values())
        total_losses = sum(r['losses'] for r in all_results.values())

        summary = {
            'total_games': total_games,
            'total_wins': total_wins,
            'total_draws': total_draws,
            'total_losses': total_losses,
            'overall_win_rate': total_wins / total_games,
            'overall_draw_rate': total_draws / total_games,
            'overall_loss_rate': total_losses / total_games,
            'avg_performance_rating': avg_performance_rating,
            'opponent_results': all_results
        }

        print("\n=== Evaluation Summary ===")
        print(f"Total games: {total_games}")
        print(f"Overall win rate: {summary['overall_win_rate']:.2%}")
        print(f"Overall draw rate: {summary['overall_draw_rate']:.2%}")
        print(f"Average performance rating: {avg_performance_rating:.0f}")

        # Check milestones
        print("\n=== Milestones ===")
        for name, threshold in self.config.ELO_MILESTONES.items():
            achieved = avg_performance_rating >= threshold
            status = "✓" if achieved else "✗"
            print(f"{status} {name.capitalize()}: {threshold} Elo")

        return summary

    def quick_evaluation(self, num_games: int = 20) -> Dict:
        """
        Quick evaluation with fewer games (for fast feedback during training).

        Args:
            num_games: Number of games per opponent

        Returns:
            Quick evaluation results
        """
        print(f"=== Quick Evaluation ({num_games} games/opponent) ===")

        results = {}
        for opponent_name in ['random', 'greedy_material']:
            if opponent_name in self.opponents:
                results[opponent_name] = self.evaluate_against_opponent(
                    opponent_name, num_games
                )

        return results
