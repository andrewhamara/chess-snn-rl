#!/usr/bin/env python3
"""Training script with verbose game logging."""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.chess_snn import ChessSNN
from src.training.trainer import Trainer
from config.training_config import TrainingConfig
from config.model_config import ModelConfig


def main():
    parser = argparse.ArgumentParser(description="Train Chess SNN with verbose logging")
    parser.add_argument('--iterations', type=int, default=10,
                       help='Number of training iterations')
    parser.add_argument('--games-per-iter', type=int, default=10,
                       help='Number of games per iteration')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cuda', 'cpu'],
                       help='Device to train on')
    parser.add_argument('--show-every', type=int, default=2,
                       help='Show detailed game every N games')

    args = parser.parse_args()

    # Update config
    config = TrainingConfig()
    config.GAMES_PER_ITERATION = args.games_per_iter
    config.DEVICE = args.device
    config.SAVE_DIR = "checkpoints_verbose"

    print("=== Chess SNN Training (Verbose Mode) ===")
    print(f"Training for {args.iterations} iterations")
    print(f"Games per iteration: {args.games_per_iter}")
    print(f"Showing detailed view every {args.show_every} games")
    print(f"Device: {args.device}")

    # Create model
    model = ChessSNN(ModelConfig())

    # Create trainer
    trainer = Trainer(model, config)

    # Load checkpoint if specified
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)

    # Modify self-play engine to be verbose
    original_generate = trainer.self_play_engine.generate_games

    def verbose_generate(num_games, temperature=1.0, show_progress=True):
        return original_generate(
            num_games,
            temperature,
            show_progress,
            verbose=True,
            sample_every=args.show_every
        )

    trainer.self_play_engine.generate_games = verbose_generate

    # Train
    try:
        trainer.train(args.iterations)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        trainer.save_checkpoint()
        print("Checkpoint saved")

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
