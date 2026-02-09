"""Complete SNN model for chess with policy and value heads."""

import torch
import torch.nn as nn
from spikingjelly.activation_based import functional

from config.model_config import ModelConfig
from .snn_base import ConvLIFBlock, LinearLIFBlock, TemporalAvgPool, SpatialAvgPool


class ChessSNN(nn.Module):
    """
    Spiking Neural Network for chess with dual heads (policy + value).

    Architecture:
        Input [T, B, 13, 8, 8]
          ↓
        Conv blocks → [T, B, 256, 8, 8]
          ↓
        Spatial + Temporal pooling → [B, 256]
          ↓
        ├─→ Policy head → [B, 4672]
        └─→ Value head → [B, 1]
    """

    def __init__(self, config: ModelConfig = None):
        """Initialize ChessSNN."""
        super().__init__()

        if config is None:
            config = ModelConfig()

        self.config = config

        # Convolutional backbone
        self.conv1 = ConvLIFBlock(
            config.INPUT_CHANNELS,
            config.CONV_CHANNELS[0],
            kernel_size=config.KERNEL_SIZES[0],
            padding=config.KERNEL_SIZES[0] // 2,
            tau=config.NEURON_TAU
        )

        self.conv2 = ConvLIFBlock(
            config.CONV_CHANNELS[0],
            config.CONV_CHANNELS[1],
            kernel_size=config.KERNEL_SIZES[1],
            padding=config.KERNEL_SIZES[1] // 2,
            tau=config.NEURON_TAU
        )

        self.conv3 = ConvLIFBlock(
            config.CONV_CHANNELS[1],
            config.CONV_CHANNELS[2],
            kernel_size=config.KERNEL_SIZES[2],
            padding=config.KERNEL_SIZES[2] // 2,
            tau=config.NEURON_TAU
        )

        # Pooling layers
        self.spatial_pool = SpatialAvgPool()
        self.temporal_pool = TemporalAvgPool()

        # Policy head
        self.policy_fc1 = LinearLIFBlock(
            config.CONV_CHANNELS[2],
            config.POLICY_HIDDEN_DIM,
            tau=config.NEURON_TAU
        )
        self.policy_fc2 = nn.Linear(config.POLICY_HIDDEN_DIM, config.ACTION_SPACE_SIZE)

        # Value head
        self.value_fc1 = LinearLIFBlock(
            config.CONV_CHANNELS[2],
            config.VALUE_HIDDEN_DIM,
            tau=config.NEURON_TAU
        )
        self.value_fc2 = nn.Linear(config.VALUE_HIDDEN_DIM, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, legal_mask: torch.Tensor = None):
        """
        Forward pass.

        Args:
            x: Input spike trains [T, B, 13, 8, 8] or [B, T, 13, 8, 8]
            legal_mask: Optional legal move mask [B, 4672]

        Returns:
            policy_logits: [B, 4672]
            value: [B, 1]
        """
        # Handle both [T, B, ...] and [B, T, ...] formats
        if x.dim() == 5 and x.shape[0] != self.config.TIME_STEPS:
            # Assume [B, T, C, H, W], transpose to [T, B, C, H, W]
            x = x.transpose(0, 1)

        # Reset neuron states at start of forward pass
        functional.reset_net(self)

        # Convolutional backbone
        x = self.conv1(x)  # [T, B, 64, 8, 8]
        x = self.conv2(x)  # [T, B, 128, 8, 8]
        x = self.conv3(x)  # [T, B, 256, 8, 8]

        # Pooling
        x = self.spatial_pool(x)  # [T, B, 256]

        # Split into policy and value branches
        policy_features = x
        value_features = x

        # Policy head
        policy = self.policy_fc1(policy_features)  # [T, B, 1024]
        policy = self.temporal_pool(policy)         # [B, 1024]
        policy_logits = self.policy_fc2(policy)     # [B, 4672]

        # Value head
        value = self.value_fc1(value_features)      # [T, B, 128]
        value = self.temporal_pool(value)            # [B, 128]
        value = self.value_fc2(value)                # [B, 1]
        value = torch.tanh(value)                    # Bound to [-1, 1]

        # Apply legal move mask if provided
        if legal_mask is not None:
            # Mask illegal moves with large negative value
            policy_logits = policy_logits + (1.0 - legal_mask) * (-1e9)

        return policy_logits, value

    def get_policy_value(self, x: torch.Tensor, legal_mask: torch.Tensor = None):
        """
        Get policy distribution and value estimate.

        Args:
            x: Input spike trains [T, B, 13, 8, 8] or [B, T, 13, 8, 8]
            legal_mask: Optional legal move mask [B, 4672]

        Returns:
            policy_probs: [B, 4672] - Probability distribution over actions
            value: [B, 1] - Value estimate
        """
        policy_logits, value = self.forward(x, legal_mask)
        policy_probs = torch.softmax(policy_logits, dim=-1)
        return policy_probs, value

    def select_action(self, x: torch.Tensor, legal_mask: torch.Tensor,
                     temperature: float = 1.0, deterministic: bool = False):
        """
        Select action from policy.

        Args:
            x: Input spike trains [T, 13, 8, 8] or [1, T, 13, 8, 8]
            legal_mask: Legal move mask [4672] or [1, 4672]
            temperature: Sampling temperature
            deterministic: If True, select argmax action

        Returns:
            action: Selected action index
            log_prob: Log probability of selected action
            value: Value estimate
        """
        # Add batch dimension if needed
        if x.dim() == 4:
            x = x.unsqueeze(0)  # [1, T, 13, 8, 8]
        if legal_mask.dim() == 1:
            legal_mask = legal_mask.unsqueeze(0)  # [1, 4672]

        policy_logits, value = self.forward(x, legal_mask)

        if deterministic:
            action = policy_logits.argmax(dim=-1)
            log_prob = torch.log_softmax(policy_logits, dim=-1).gather(1, action.unsqueeze(-1))
        else:
            # Apply temperature
            policy_logits = policy_logits / temperature

            # Sample action
            policy_probs = torch.softmax(policy_logits, dim=-1)
            action = torch.multinomial(policy_probs, num_samples=1).squeeze(-1)
            log_prob = torch.log_softmax(policy_logits, dim=-1).gather(1, action.unsqueeze(-1))

        return action.item(), log_prob.squeeze(), value.squeeze()
