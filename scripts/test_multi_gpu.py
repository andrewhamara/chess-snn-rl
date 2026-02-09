#!/usr/bin/env python3
"""Test multi-GPU training with visible progress."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.models.chess_snn import ChessSNN
from src.training.parallel_self_play import ParallelSelfPlayEngine
from config.model_config import ModelConfig


def main():
    print("="*70)
    print("  Testing Multi-GPU Self-Play with Progress Tracking")
    print("="*70)

    # Create model
    model = ChessSNN(ModelConfig())

    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"\nDetected {num_gpus} GPUs")

    if num_gpus == 0:
        print("No GPUs available, using CPU")
        gpu_ids = [None]
    else:
        # Use all available GPUs (or subset for testing)
        gpu_ids = list(range(min(num_gpus, 8)))
        print(f"Using GPUs: {gpu_ids}")

    # Create parallel self-play engine
    engine = ParallelSelfPlayEngine(model, gpu_ids=gpu_ids, max_game_length=200)

    # Generate games with progress tracking
    print("\n" + "="*70)
    print("  Generating 40 test games")
    print("="*70)

    games = engine.generate_games(
        num_games=40,
        temperature=1.0,
        show_progress=True
    )

    # Show statistics
    stats = engine.get_game_stats(games)

    print("\n" + "="*70)
    print("  Results")
    print("="*70)
    print(f"Total games generated: {len(games)}")
    print(f"Average moves per game: {stats['avg_moves']:.1f}")
    print(f"White win rate: {stats['white_win_rate']*100:.1f}%")
    print(f"Draw rate: {stats['draw_rate']*100:.1f}%")
    print(f"Black win rate: {stats['black_win_rate']*100:.1f}%")
    print("="*70)

    print("\n✓ Multi-GPU self-play test completed successfully!")


if __name__ == '__main__':
    main()
