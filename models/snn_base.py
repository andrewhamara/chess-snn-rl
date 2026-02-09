"""Base SNN building blocks using SpikingJelly."""

import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate, layer


class ConvLIFBlock(nn.Module):
    """Convolutional block with LIF neurons."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, tau: float = 2.0):
        """
        Initialize Conv-LIF block.

        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Convolution kernel size
            stride: Convolution stride
            padding: Convolution padding
            tau: LIF time constant
        """
        super().__init__()

        # Use regular Conv2d and handle time dimension manually
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )

        self.bn = nn.BatchNorm2d(out_channels)

        self.lif = neuron.LIFNode(
            tau=tau,
            surrogate_function=surrogate.ATan(),
            step_mode='m'  # Multi-step mode
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [T, B, C, H, W]

        Returns:
            Output tensor [T, B, C', H', W']
        """
        # Reshape: [T, B, C, H, W] -> [T*B, C, H, W]
        T, B, C, H, W = x.shape
        x = x.reshape(T * B, C, H, W)

        # Apply conv and bn
        x = self.conv(x)
        x = self.bn(x)

        # Reshape back: [T*B, C', H', W'] -> [T, B, C', H', W']
        _, C_out, H_out, W_out = x.shape
        x = x.reshape(T, B, C_out, H_out, W_out)

        # Apply LIF
        x = self.lif(x)
        return x


class LinearLIFBlock(nn.Module):
    """Linear block with LIF neurons."""

    def __init__(self, in_features: int, out_features: int, tau: float = 2.0):
        """
        Initialize Linear-LIF block.

        Args:
            in_features: Input features
            out_features: Output features
            tau: LIF time constant
        """
        super().__init__()

        # Use regular Linear and handle time dimension manually
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.lif = neuron.LIFNode(
            tau=tau,
            surrogate_function=surrogate.ATan(),
            step_mode='m'
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [T, B, F]

        Returns:
            Output tensor [T, B, F']
        """
        # Reshape: [T, B, F] -> [T*B, F]
        T, B, F = x.shape
        x = x.reshape(T * B, F)

        # Apply linear
        x = self.linear(x)

        # Reshape back: [T*B, F'] -> [T, B, F']
        F_out = x.shape[1]
        x = x.reshape(T, B, F_out)

        # Apply LIF
        x = self.lif(x)
        return x


class TemporalAvgPool(nn.Module):
    """Average pooling over temporal dimension."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Average over time dimension.

        Args:
            x: Input tensor [T, B, ...]

        Returns:
            Output tensor [B, ...]
        """
        return x.mean(dim=0)


class SpatialAvgPool(nn.Module):
    """Global average pooling over spatial dimensions."""

    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Global average pool over H, W dimensions.

        Args:
            x: Input tensor [T, B, C, H, W]

        Returns:
            Output tensor [T, B, C]
        """
        T, B, C, H, W = x.shape
        x = x.reshape(T * B, C, H, W)
        x = self.pool(x)
        x = x.reshape(T, B, C)
        return x
