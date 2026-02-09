"""Parallel self-play across multiple GPUs."""

import torch
import torch.multiprocessing as mp
from typing import List, Dict
from queue import Queue
import os
import numpy as np

from src.chess_env.board import ChessBoard
from src.chess_env.encoder import BoardEncoder
from src.chess_env.move_encoding import MoveEncoder


def convert_game_to_tensors(game: Dict) -> Dict:
    """
    Convert game data from numpy arrays back to PyTorch tensors.

    Args:
        game: Game dict with numpy arrays

    Returns:
        Game dict with PyTorch tensors
    """
    return {
        'states': [torch.from_numpy(s) if isinstance(s, np.ndarray) else s
                   for s in game['states']],
        'actions': game['actions'],
        'legal_masks': [torch.from_numpy(m) if isinstance(m, np.ndarray) else m
                       for m in game['legal_masks']],
        'outcome': game['outcome'],
        'termination': game['termination'],
        'move_count': game['move_count'],
        'legal_move_rate': game.get('legal_move_rate', 1.0)
    }


def worker_play_games(rank: int, model_state: dict, num_games: int,
                     temperature: float, result_queue: mp.Queue,
                     device_id: int):
    """
    Worker process for parallel game generation.

    Args:
        rank: Worker ID
        model_state: Model state dict
        num_games: Number of games this worker should generate
        temperature: Sampling temperature
        result_queue: Queue to put results
        device_id: GPU device ID
    """
    try:
        # Import here to avoid issues with multiprocessing
        from src.models.chess_snn import ChessSNN
        from config.model_config import ModelConfig
        import numpy as np

        # Set device for this worker
        device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

        # Create model and load state
        model = ChessSNN(ModelConfig()).to(device)
        model.load_state_dict(model_state)
        model.eval()

        # Create encoders
        board_encoder = BoardEncoder()
        move_encoder = MoveEncoder()

        # Generate games
        games = []
        for _ in range(num_games):
            game = play_single_game(model, board_encoder, move_encoder,
                                   temperature, device, max_moves=200)

            # Convert tensors to numpy arrays for safe multiprocessing
            game_serializable = {
                'states': [s.cpu().numpy() for s in game['states']],
                'actions': game['actions'],  # Already integers
                'legal_masks': [m.cpu().numpy() for m in game['legal_masks']],
                'outcome': game['outcome'],
                'termination': game['termination'],
                'move_count': game['move_count'],
                'legal_move_rate': game['legal_move_rate']
            }
            games.append(game_serializable)

        # Put results in queue
        result_queue.put((rank, games))

    except Exception as e:
        # Report errors back to main process
        import traceback
        error_msg = f"Worker {rank} failed: {str(e)}\n{traceback.format_exc()}"
        result_queue.put((rank, {'error': error_msg}))


def play_single_game(model, board_encoder: BoardEncoder,
                    move_encoder: MoveEncoder, temperature: float,
                    device: torch.device, max_moves: int = 200) -> Dict:
    """Play one self-play game."""
    board = ChessBoard()
    states = []
    actions = []
    legal_masks = []
    move_count = 0

    with torch.no_grad():
        while not board.is_game_over() and move_count < max_moves:
            # Encode current position
            legal_moves = board.get_legal_moves()
            state = board_encoder.encode(board.board, legal_moves)
            legal_mask = torch.from_numpy(
                move_encoder.get_legal_action_mask(board.board)
            )

            # Move to device
            state = state.to(device)
            legal_mask = legal_mask.to(device)

            # Select action
            action, _, _ = model.select_action(
                state, legal_mask, temperature=temperature
            )

            # Convert action to move
            move = move_encoder.action_to_move(action, board.board)

            if move is None or move not in legal_moves:
                # Fallback to random legal move
                move = legal_moves[torch.randint(len(legal_moves), (1,)).item()]
                action = move_encoder.move_to_action(move, board.board)

            # Store data
            states.append(state.cpu())
            actions.append(action)
            legal_masks.append(legal_mask.cpu())

            # Make move
            board.make_move(move)
            move_count += 1

    # Get final outcome
    termination, outcome = board.get_outcome()

    return {
        'states': states,
        'actions': actions,
        'legal_masks': legal_masks,
        'outcome': outcome,
        'termination': termination,
        'move_count': move_count,
        'legal_move_rate': 1.0
    }


class ParallelSelfPlayEngine:
    """
    Parallel self-play engine using multiple GPUs.
    """

    def __init__(self, model, gpu_ids: List[int] = None, max_game_length: int = 200):
        """
        Initialize parallel self-play engine.

        Args:
            model: ChessSNN model
            gpu_ids: List of GPU IDs to use (e.g., [0, 1, 2, 3])
                    If None, uses all available GPUs
            max_game_length: Maximum moves per game
        """
        self.model = model
        self.max_game_length = max_game_length

        # Determine GPUs to use
        if gpu_ids is None:
            num_gpus = torch.cuda.device_count()
            self.gpu_ids = list(range(num_gpus)) if num_gpus > 0 else [None]
        else:
            self.gpu_ids = gpu_ids

        self.num_workers = len(self.gpu_ids)

        print(f"Parallel self-play initialized with {self.num_workers} workers")
        print(f"GPU IDs: {self.gpu_ids}")

    def generate_games(self, num_games: int, temperature: float = 1.0,
                      show_progress: bool = True) -> List[Dict]:
        """
        Generate games in parallel across GPUs.

        Args:
            num_games: Total number of games to generate
            temperature: Sampling temperature
            show_progress: Show progress bar

        Returns:
            List of game dictionaries
        """
        if self.num_workers == 1:
            # Single GPU/CPU - use regular self-play
            from .self_play import SelfPlayEngine
            device = f"cuda:{self.gpu_ids[0]}" if self.gpu_ids[0] is not None else "cpu"
            engine = SelfPlayEngine(self.model, device, self.max_game_length)
            return engine.generate_games(num_games, temperature, show_progress)

        # Multi-GPU parallel generation
        return self._parallel_generate(num_games, temperature, show_progress)

    def _parallel_generate(self, num_games: int, temperature: float,
                          show_progress: bool) -> List[Dict]:
        """Generate games in parallel."""
        import time

        # Distribute games across workers
        games_per_worker = num_games // self.num_workers
        remainder = num_games % self.num_workers

        worker_games = [games_per_worker + (1 if i < remainder else 0)
                       for i in range(self.num_workers)]

        # Get model state dict
        model_state = self.model.state_dict()

        # Create result queue and progress tracking
        mp.set_start_method('spawn', force=True)
        result_queue = mp.Queue()

        # Shared counters for progress (using Manager for cross-process sharing)
        manager = mp.Manager()
        progress_dict = manager.dict()
        for i in range(self.num_workers):
            progress_dict[i] = 0

        # Launch workers
        processes = []
        for rank, (num_worker_games, gpu_id) in enumerate(zip(worker_games, self.gpu_ids)):
            if num_worker_games == 0:
                continue

            device_id = gpu_id if gpu_id is not None else 0
            p = mp.Process(
                target=worker_play_games,
                args=(rank, model_state, num_worker_games, temperature,
                     result_queue, device_id)
            )
            p.start()
            processes.append((p, rank))

        # Collect results from workers
        all_games = []
        worker_results = {}
        start_time = time.time()

        if show_progress:
            print(f"\nGenerating {num_games} games across {len(processes)} GPUs...")
            print(f"Games per GPU: {worker_games[:len(processes)]}")
            print("-" * 70)

        # Collect results as workers complete
        while len(worker_results) < len(processes):
            try:
                rank, result = result_queue.get(timeout=5)

                # Check for errors
                if isinstance(result, dict) and 'error' in result:
                    print(f"\n⚠️  ERROR in GPU {rank}:")
                    print(result['error'])
                    worker_results[rank] = []
                    continue

                # Store results
                worker_results[rank] = result

                if show_progress:
                    elapsed = time.time() - start_time
                    num_completed = len(worker_results)
                    total_collected = sum(len(games) for games in worker_results.values())

                    print(f"[{elapsed:6.1f}s] GPU {rank}: Completed {len(result)} games | "
                          f"Total: {num_completed}/{len(processes)} GPUs done | "
                          f"Progress: {total_collected}/{num_games} games")

            except Exception as e:
                # Timeout or other error - check if processes are still alive
                alive = sum(1 for p, _ in processes if p.is_alive())
                if alive == 0 and len(worker_results) >= len(processes):
                    break
                if alive == 0:
                    print(f"\n⚠️  All workers stopped but only {len(worker_results)}/{len(processes)} reported")
                    break

        # Combine all games
        for rank in sorted(worker_results.keys()):
            all_games.extend(worker_results[rank])

        # Wait for all processes
        for p, rank in processes:
            p.join()

        if show_progress:
            elapsed = time.time() - start_time
            print("-" * 70)
            print(f"Completed {len(all_games)} games in {elapsed:.1f}s "
                  f"({len(all_games)/elapsed:.1f} games/sec)")
            print()

        # Convert numpy arrays back to tensors for training
        all_games = [convert_game_to_tensors(game) for game in all_games]

        return all_games

    def get_game_stats(self, games: List[Dict]) -> Dict[str, float]:
        """Compute statistics from games."""
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
