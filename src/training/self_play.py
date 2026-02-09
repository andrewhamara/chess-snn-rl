"""Self-play engine for generating training data."""

import torch
import chess
from typing import List, Tuple, Dict
from tqdm import tqdm

from src.chess_env.board import ChessBoard
from src.chess_env.encoder import BoardEncoder
from src.chess_env.move_encoding import MoveEncoder
from config.training_config import TrainingConfig


class SelfPlayEngine:
    """
    Generates self-play games using current model.
    """

    def __init__(self, model, device: str = "cpu", max_game_length: int = 200):
        """
        Initialize self-play engine.

        Args:
            model: ChessSNN model
            device: Device to run on
            max_game_length: Maximum moves per game
        """
        self.model = model
        self.device = device
        self.max_game_length = max_game_length

        self.board_encoder = BoardEncoder()
        self.move_encoder = MoveEncoder()

        self.model.eval()

    def play_game(self, temperature: float = 1.0,
                  store_states: bool = True, verbose: bool = False,
                  game_id: int = None) -> Dict:
        """
        Play one self-play game.

        Args:
            temperature: Sampling temperature for action selection
            store_states: Whether to store states for training
            verbose: Print game moves and details
            game_id: Game identifier for logging

        Returns:
            Dictionary with game data
        """
        board = ChessBoard()
        states = []
        actions = []
        legal_masks = []

        move_count = 0

        if verbose:
            game_str = f"[Game {game_id}]" if game_id else "[Game]"
            print(f"\n{game_str} Starting new game")
            print(board.board)

        with torch.no_grad():
            while not board.is_game_over() and move_count < self.max_game_length:
                # Encode current position
                legal_moves = board.get_legal_moves()
                state = self.board_encoder.encode(board.board, legal_moves)
                legal_mask = torch.from_numpy(
                    self.move_encoder.get_legal_action_mask(board.board)
                )

                # Move to device
                state = state.to(self.device)
                legal_mask = legal_mask.to(self.device)

                # Select action
                action, log_prob, value = self.model.select_action(
                    state, legal_mask, temperature=temperature
                )

                # Convert action to move
                move = self.move_encoder.action_to_move(action, board.board)

                if move is None or move not in legal_moves:
                    # Fallback to random legal move if model outputs illegal move
                    move = legal_moves[torch.randint(len(legal_moves), (1,)).item()]
                    action = self.move_encoder.move_to_action(move, board.board)
                    if verbose:
                        print(f"  WARNING: Illegal move, using random")

                # Store data
                if store_states:
                    states.append(state.cpu())
                    actions.append(action)
                    legal_masks.append(legal_mask.cpu())

                # Make move
                if verbose:
                    print(f"  Move {move_count + 1}: {move.uci()} (value: {value.item():.3f})")

                board.make_move(move)
                move_count += 1

                if verbose and move_count % 10 == 0:
                    print(f"  Position after {move_count} moves:")
                    print(board.board)

        # Get final outcome
        termination, outcome = board.get_outcome()

        # Calculate metrics
        legal_move_rate = 1.0  # We enforce legal moves in this implementation

        if verbose:
            print(f"\n  Game Over!")
            print(f"  Result: {termination} (outcome: {outcome})")
            print(f"  Total moves: {move_count}")
            print(f"  Final position:")
            print(board.board)

        return {
            'states': states,
            'actions': actions,
            'legal_masks': legal_masks,
            'outcome': outcome,
            'termination': termination,
            'move_count': move_count,
            'legal_move_rate': legal_move_rate
        }

    def generate_games(self, num_games: int, temperature: float = 1.0,
                      show_progress: bool = True, verbose: bool = False,
                      sample_every: int = 10) -> List[Dict]:
        """
        Generate multiple self-play games.

        Args:
            num_games: Number of games to generate
            temperature: Sampling temperature
            show_progress: Show progress bar
            verbose: Show detailed game logs
            sample_every: If verbose, show every Nth game

        Returns:
            List of game dictionaries
        """
        games = []

        iterator = range(num_games)
        if show_progress:
            iterator = tqdm(iterator, desc="Generating games",
                          bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

        for i in iterator:
            # Show verbose output for sample games
            show_this_game = verbose and (i % sample_every == 0)
            game = self.play_game(
                temperature=temperature,
                verbose=show_this_game,
                game_id=i+1
            )
            games.append(game)

            # Print running stats every N games
            if show_progress and (i + 1) % (num_games // 5 or 1) == 0:
                recent_games = games[-10:] if len(games) >= 10 else games
                avg_moves = sum(g['move_count'] for g in recent_games) / len(recent_games)
                outcomes = [g['outcome'] for g in recent_games]
                draw_rate = sum(1 for o in outcomes if abs(o) < 0.5) / len(outcomes)

                if isinstance(iterator, tqdm):
                    iterator.set_postfix({
                        'avg_moves': f'{avg_moves:.0f}',
                        'draw_rate': f'{draw_rate:.1%}'
                    })

        return games

    def get_game_stats(self, games: List[Dict]) -> Dict[str, float]:
        """
        Compute statistics from games.

        Args:
            games: List of game dictionaries

        Returns:
            Statistics dictionary
        """
        if len(games) == 0:
            return {}

        outcomes = [g['outcome'] for g in games]
        move_counts = [g['move_count'] for g in games]
        terminations = [g['termination'] for g in games]

        # Count termination types
        termination_counts = {}
        for term in terminations:
            if term:
                termination_counts[term] = termination_counts.get(term, 0) + 1

        # Count outcomes
        white_wins = sum(1 for o in outcomes if o > 0.5)
        draws = sum(1 for o in outcomes if abs(o) < 0.5)
        black_wins = sum(1 for o in outcomes if o < -0.5)

        total = len(games)

        stats = {
            'total_games': total,
            'avg_moves': sum(move_counts) / total,
            'white_win_rate': white_wins / total,
            'draw_rate': draws / total,
            'black_win_rate': black_wins / total,
            'avg_outcome': sum(outcomes) / total,
            'min_moves': min(move_counts),
            'max_moves': max(move_counts),
        }

        # Add termination stats
        for term, count in termination_counts.items():
            stats[f'term_{term}'] = count / total

        return stats
