"""Loss functions for training."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChessLoss:
    """
    Combined loss for chess SNN training.

    Loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy
    """

    def __init__(self, value_loss_coef: float = 1.0, entropy_coef: float = 0.01):
        """
        Initialize loss.

        Args:
            value_loss_coef: Coefficient for value loss
            entropy_coef: Coefficient for entropy bonus
        """
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

    def compute_loss(self, policy_logits: torch.Tensor, values: torch.Tensor,
                    actions: torch.Tensor, outcomes: torch.Tensor,
                    legal_masks: torch.Tensor = None) -> dict:
        """
        Compute total loss.

        Args:
            policy_logits: [B, 4672] - Policy logits
            values: [B, 1] - Value predictions
            actions: [B] - Actions taken
            outcomes: [B] - Game outcomes
            legal_masks: [B, 4672] - Legal move masks (optional)

        Returns:
            Dictionary with loss components
        """
        # Policy loss (REINFORCE with baseline)
        log_probs = F.log_softmax(policy_logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        # Advantage = outcome - value (as baseline)
        advantages = outcomes - values.squeeze(-1).detach()

        # Policy gradient loss
        policy_loss = -(action_log_probs * advantages).mean()

        # Value loss (MSE)
        value_loss = F.mse_loss(values.squeeze(-1), outcomes)

        # Entropy bonus (encourage exploration)
        probs = F.softmax(policy_logits, dim=-1)

        if legal_masks is not None:
            # Only consider legal moves for entropy
            probs = probs * legal_masks
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)

        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()

        # Total loss
        total_loss = (policy_loss +
                     self.value_loss_coef * value_loss -
                     self.entropy_coef * entropy)

        return {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': entropy,
            'mean_advantage': advantages.mean(),
            'mean_value': values.mean()
        }

    def __call__(self, *args, **kwargs):
        """Make loss callable."""
        return self.compute_loss(*args, **kwargs)
