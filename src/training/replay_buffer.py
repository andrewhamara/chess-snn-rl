"""Replay buffer for storing experiences."""

import numpy as np
import torch
from collections import deque
from typing import List, Tuple, Dict
import random


class ReplayBuffer:
    """
    Experience replay buffer for storing game positions.

    Stores tuples of (state, action, outcome) from self-play games.
    """

    def __init__(self, capacity: int):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of experiences to store
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, state: torch.Tensor, action: int, outcome: float,
            legal_mask: torch.Tensor = None):
        """
        Add experience to buffer.

        Args:
            state: Board state encoding [T, 13, 8, 8]
            action: Action taken
            outcome: Game outcome from this player's perspective {-1, 0, 1}
            legal_mask: Legal move mask [4672]
        """
        experience = {
            'state': state.cpu(),
            'action': action,
            'outcome': outcome,
            'legal_mask': legal_mask.cpu() if legal_mask is not None else None
        }
        self.buffer.append(experience)

    def add_game(self, states: List[torch.Tensor], actions: List[int],
                 outcome: float, legal_masks: List[torch.Tensor] = None):
        """
        Add full game to buffer.

        Args:
            states: List of board states
            actions: List of actions taken
            outcome: Final game outcome (from white's perspective)
            legal_masks: Optional list of legal move masks
        """
        # Add experiences with alternating outcome signs (white/black perspective)
        for i, (state, action) in enumerate(zip(states, actions)):
            # Alternate perspective: even indices = white, odd = black
            player_outcome = outcome if i % 2 == 0 else -outcome

            legal_mask = legal_masks[i] if legal_masks else None
            self.add(state, action, player_outcome, legal_mask)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample batch of experiences.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Dictionary with:
                - states: [B, T, 13, 8, 8]
                - actions: [B]
                - outcomes: [B]
                - legal_masks: [B, 4672] or None
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)

        experiences = random.sample(self.buffer, batch_size)

        states = torch.stack([exp['state'] for exp in experiences])
        actions = torch.tensor([exp['action'] for exp in experiences], dtype=torch.long)
        outcomes = torch.tensor([exp['outcome'] for exp in experiences], dtype=torch.float32)

        # Handle legal masks
        if experiences[0]['legal_mask'] is not None:
            legal_masks = torch.stack([exp['legal_mask'] for exp in experiences])
        else:
            legal_masks = None

        return {
            'states': states,
            'actions': actions,
            'outcomes': outcomes,
            'legal_masks': legal_masks
        }

    def __len__(self) -> int:
        """Get current buffer size."""
        return len(self.buffer)

    def clear(self):
        """Clear all experiences."""
        self.buffer.clear()

    def get_stats(self) -> Dict[str, float]:
        """
        Get buffer statistics.

        Returns:
            Dictionary with statistics
        """
        if len(self.buffer) == 0:
            return {
                'size': 0,
                'avg_outcome': 0.0,
                'win_rate': 0.0,
                'draw_rate': 0.0,
                'loss_rate': 0.0
            }

        outcomes = [exp['outcome'] for exp in self.buffer]
        avg_outcome = np.mean(outcomes)

        wins = sum(1 for o in outcomes if o > 0.5)
        draws = sum(1 for o in outcomes if abs(o) < 0.5)
        losses = sum(1 for o in outcomes if o < -0.5)

        total = len(outcomes)

        return {
            'size': total,
            'avg_outcome': avg_outcome,
            'win_rate': wins / total,
            'draw_rate': draws / total,
            'loss_rate': losses / total
        }
