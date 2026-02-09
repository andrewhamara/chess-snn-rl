"""Multi-GPU training script with parallel self-play."""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.optim as optim
from tqdm import tqdm
import json

from src.models.chess_snn import ChessSNN
from src.training.replay_buffer import ReplayBuffer
from src.training.parallel_self_play import ParallelSelfPlayEngine
from src.training.loss import ChessLoss
from config.model_config import ModelConfig
from config.training_config import TrainingConfig


class MultiGPUTrainer:
    """Trainer with multi-GPU parallel self-play."""

    def __init__(self, gpu_ids: list, config: TrainingConfig = None):
        """
        Initialize multi-GPU trainer.

        Args:
            gpu_ids: List of GPU IDs to use (e.g., [0, 1, 2, 3, 4, 5, 6, 7])
            config: Training configuration
        """
        if config is None:
            config = TrainingConfig()
        self.config = config
        self.gpu_ids = gpu_ids

        # Primary device for training
        self.device = torch.device(f"cuda:{gpu_ids[0]}")
        print(f"Primary training device: {self.device}")
        print(f"Parallel self-play GPUs: {gpu_ids}")

        # Create model on primary device
        self.model = ChessSNN(ModelConfig()).to(self.device)

        # Create optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE
        )

        # Create replay buffer
        self.replay_buffer = ReplayBuffer(config.BUFFER_SIZE)

        # Create parallel self-play engine
        self.self_play_engine = ParallelSelfPlayEngine(
            self.model,
            gpu_ids=gpu_ids,
            max_game_length=config.MAX_GAME_LENGTH
        )

        # Create loss function
        self.loss_fn = ChessLoss(
            value_loss_coef=config.VALUE_LOSS_COEF,
            entropy_coef=config.ENTROPY_COEF
        )

        # Training state
        self.current_iteration = 0
        self.temperature = config.INITIAL_TEMPERATURE
        self.training_history = []

        # Create directories
        Path(config.SAVE_DIR).mkdir(exist_ok=True, parents=True)

    def train_iteration(self) -> dict:
        """Run one training iteration with parallel self-play."""
        print(f"\n=== Iteration {self.current_iteration + 1} ===")

        # 1. Generate self-play games in parallel
        print(f"Generating {self.config.GAMES_PER_ITERATION} games in parallel...")
        games = self.self_play_engine.generate_games(
            self.config.GAMES_PER_ITERATION,
            temperature=self.temperature
        )

        # Get game statistics
        game_stats = self.self_play_engine.get_game_stats(games)
        print(f"Game stats: {game_stats}")

        # 2. Add to replay buffer
        print("Adding games to replay buffer...")
        for game in games:
            self.replay_buffer.add_game(
                game['states'],
                game['actions'],
                game['outcome'],
                game['legal_masks']
            )

        buffer_stats = self.replay_buffer.get_stats()
        print(f"Buffer size: {len(self.replay_buffer)}")

        # 3. Train on replay buffer
        print(f"Training for {self.config.EPOCHS_PER_ITERATION} epochs...")
        epoch_losses = []

        self.model.train()
        for epoch in range(self.config.EPOCHS_PER_ITERATION):
            batch = self.replay_buffer.sample(self.config.BATCH_SIZE)

            # Move to device
            states = batch['states'].to(self.device)
            actions = batch['actions'].to(self.device)
            outcomes = batch['outcomes'].to(self.device)
            legal_masks = batch['legal_masks'].to(self.device) if batch['legal_masks'] is not None else None

            # Forward pass
            policy_logits, values = self.model(states, legal_masks)

            # Compute loss
            loss_dict = self.loss_fn(
                policy_logits, values, actions, outcomes, legal_masks
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss_dict['total_loss'].backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.GRADIENT_CLIP
            )

            self.optimizer.step()

            epoch_losses.append({k: v.item() if torch.is_tensor(v) else v
                               for k, v in loss_dict.items()})

        self.model.eval()

        # Average losses
        avg_losses = {
            k: sum(d[k] for d in epoch_losses) / len(epoch_losses)
            for k in epoch_losses[0].keys()
        }

        print(f"Training losses: {avg_losses}")

        # 4. Update temperature
        self.temperature = max(
            self.config.MIN_TEMPERATURE,
            self.temperature * self.config.TEMPERATURE_DECAY
        )

        # 5. Save checkpoint
        if (self.current_iteration + 1) % self.config.CHECKPOINT_INTERVAL == 0:
            self.save_checkpoint()

        # Compile iteration stats
        iteration_stats = {
            'iteration': self.current_iteration + 1,
            'temperature': self.temperature,
            'game_stats': game_stats,
            'buffer_stats': buffer_stats,
            'training_losses': avg_losses
        }

        self.training_history.append(iteration_stats)
        self.current_iteration += 1

        return iteration_stats

    def train(self, num_iterations: int):
        """Run full training loop."""
        print(f"Starting multi-GPU training for {num_iterations} iterations")
        print(f"GPUs: {self.gpu_ids}")

        for i in range(num_iterations):
            stats = self.train_iteration()
            self.save_stats()

        print("\nTraining complete!")
        self.save_checkpoint(final=True)

    def save_checkpoint(self, final: bool = False):
        """Save model checkpoint."""
        suffix = "final" if final else f"iter_{self.current_iteration}"
        checkpoint_path = Path(self.config.SAVE_DIR) / f"model_{suffix}.pt"

        checkpoint = {
            'iteration': self.current_iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'temperature': self.temperature,
            'training_history': self.training_history
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_iteration = checkpoint['iteration']
        self.temperature = checkpoint['temperature']
        self.training_history = checkpoint.get('training_history', [])

        print(f"Loaded checkpoint from iteration {self.current_iteration}")

    def save_stats(self):
        """Save training statistics."""
        stats_path = Path(self.config.SAVE_DIR) / "training_stats.json"

        with open(stats_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU Chess SNN Training")
    parser.add_argument('--iterations', type=int, default=100,
                       help='Number of training iterations')
    parser.add_argument('--games-per-iter', type=int, default=200,
                       help='Number of games per iteration')
    parser.add_argument('--gpus', type=str, default='0,1,2,3,4,5,6,7',
                       help='Comma-separated GPU IDs (e.g., "0,1,2,3")')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')

    args = parser.parse_args()

    # Parse GPU IDs
    gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]

    # Verify GPUs are available
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"CUDA available: {num_gpus} GPUs detected")
        for gpu_id in gpu_ids:
            if gpu_id >= num_gpus:
                print(f"Warning: GPU {gpu_id} not available")
                gpu_ids = [g for g in gpu_ids if g < num_gpus]
    else:
        print("CUDA not available, falling back to CPU")
        gpu_ids = [None]

    # Update config
    config = TrainingConfig()
    config.GAMES_PER_ITERATION = args.games_per_iter
    config.SAVE_DIR = args.save_dir

    print("\n=== Multi-GPU Chess SNN Training ===")
    print(f"Training for {args.iterations} iterations")
    print(f"Games per iteration: {args.games_per_iter}")
    print(f"GPUs: {gpu_ids}")
    print(f"Save directory: {args.save_dir}")

    # Create trainer
    trainer = MultiGPUTrainer(gpu_ids, config)

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
