"""Main training script."""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.chess_snn import ChessSNN
from src.training.trainer import Trainer
from config.training_config import TrainingConfig
from config.model_config import ModelConfig


def main():
    parser = argparse.ArgumentParser(description="Train Chess SNN")
    parser.add_argument('--iterations', type=int, default=100,
                       help='Number of training iterations')
    parser.add_argument('--games-per-iter', type=int, default=200,
                       help='Number of games per iteration')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to train on')
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')

    args = parser.parse_args()

    # Update config with args
    config = TrainingConfig()
    config.GAMES_PER_ITERATION = args.games_per_iter
    config.DEVICE = args.device
    config.SAVE_DIR = args.save_dir

    print("=== Chess SNN Training ===")
    print(f"Training for {args.iterations} iterations")
    print(f"Games per iteration: {args.games_per_iter}")
    print(f"Device: {args.device}")
    print(f"Save directory: {args.save_dir}")

    # Create model
    model = ChessSNN(ModelConfig())

    # Create trainer
    trainer = Trainer(model, config)

    # Load checkpoint if specified
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)

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
