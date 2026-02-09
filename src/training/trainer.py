"""Main training orchestrator."""

import torch
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import json

from src.models.chess_snn import ChessSNN
from .replay_buffer import ReplayBuffer
from .self_play import SelfPlayEngine
from .loss import ChessLoss
from config.model_config import ModelConfig
from config.training_config import TrainingConfig


class Trainer:
    """
    Main training loop orchestrator.
    """

    def __init__(self, model: ChessSNN = None, config: TrainingConfig = None):
        """
        Initialize trainer.

        Args:
            model: ChessSNN model (creates new if None)
            config: Training configuration
        """
        if config is None:
            config = TrainingConfig()
        self.config = config

        # Setup device
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Create or load model
        if model is None:
            model = ChessSNN()
        self.model = model.to(self.device)

        # Create optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE
        )

        # Create replay buffer
        self.replay_buffer = ReplayBuffer(config.BUFFER_SIZE)

        # Create self-play engine
        self.self_play_engine = SelfPlayEngine(
            self.model,
            device=self.device,
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
        """
        Run one training iteration.

        Returns:
            Dictionary with iteration statistics
        """
        print(f"\n=== Iteration {self.current_iteration + 1} ===")

        # 1. Generate self-play games
        print(f"Generating {self.config.GAMES_PER_ITERATION} games...")
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
        """
        Run full training loop.

        Args:
            num_iterations: Number of iterations to train
        """
        print(f"Starting training for {num_iterations} iterations")
        print(f"Config: {self.config.__dict__}")

        for i in range(num_iterations):
            stats = self.train_iteration()

            # Save stats
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
