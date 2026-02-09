"""Board encoding to spike trains for SNN input."""

import chess
import numpy as np
import torch
from typing import Tuple

from config.model_config import ModelConfig


class BoardEncoder:
    """
    Encodes chess board to spike trains using rate coding.

    Output shape: [T=16, C=13, H=8, W=8]
    - 13 channels: 6 own pieces + 6 opponent pieces + 1 legal move mask
    - Rate coding: high rate (0.8-1.0) for presence, low (0.0-0.2) for absence
    """

    # Piece type to channel mapping
    PIECE_TO_CHANNEL = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5
    }

    def __init__(self, time_steps: int = ModelConfig.TIME_STEPS):
        """Initialize encoder with time steps."""
        self.time_steps = time_steps
        self.high_rate = 0.9  # Probability of spike when piece present
        self.low_rate = 0.1   # Probability of spike when piece absent

    def encode(self, board: chess.Board, legal_moves: list = None) -> torch.Tensor:
        """
        Encode board to spike train.

        Args:
            board: Chess board state
            legal_moves: Optional list of legal moves for masking

        Returns:
            Tensor of shape [T, 13, 8, 8] with binary spikes
        """
        # Create base representation [13, 8, 8] with firing rates
        rates = self._board_to_rates(board, legal_moves)

        # Convert rates to spike trains [T, 13, 8, 8]
        spikes = self._rates_to_spikes(rates)

        return spikes

    def _board_to_rates(self, board: chess.Board, legal_moves: list = None) -> np.ndarray:
        """
        Convert board to firing rate representation.

        Returns:
            Array of shape [13, 8, 8] with firing rates
        """
        rates = np.full((13, 8, 8), self.low_rate, dtype=np.float32)

        current_color = board.turn

        # Encode pieces (first 12 channels)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue

            file = chess.square_file(square)
            rank = chess.square_rank(square)

            piece_type = piece.piece_type
            channel_offset = 0 if piece.color == current_color else 6
            channel = channel_offset + self.PIECE_TO_CHANNEL[piece_type]

            rates[channel, rank, file] = self.high_rate

        # Encode legal move mask (channel 12)
        if legal_moves is not None:
            for move in legal_moves:
                to_square = move.to_square
                file = chess.square_file(to_square)
                rank = chess.square_rank(to_square)
                rates[12, rank, file] = self.high_rate

        return rates

    def _rates_to_spikes(self, rates: np.ndarray) -> torch.Tensor:
        """
        Convert firing rates to binary spike trains.

        Args:
            rates: Array of shape [13, 8, 8] with firing rates

        Returns:
            Tensor of shape [T, 13, 8, 8] with binary spikes
        """
        # Generate random values
        random_values = np.random.rand(self.time_steps, *rates.shape)

        # Compare with rates to generate spikes
        spikes = (random_values < rates).astype(np.float32)

        return torch.from_numpy(spikes)

    def encode_batch(self, boards: list, legal_moves_list: list = None) -> torch.Tensor:
        """
        Encode batch of boards.

        Args:
            boards: List of chess.Board objects
            legal_moves_list: Optional list of legal moves for each board

        Returns:
            Tensor of shape [B, T, 13, 8, 8]
        """
        batch_spikes = []

        for i, board in enumerate(boards):
            legal_moves = legal_moves_list[i] if legal_moves_list else None
            spikes = self.encode(board, legal_moves)
            batch_spikes.append(spikes)

        return torch.stack(batch_spikes, dim=0)

    def encode_position_deterministic(self, board: chess.Board,
                                     legal_moves: list = None) -> torch.Tensor:
        """
        Encode board with deterministic spike pattern (for testing/visualization).

        Instead of stochastic spikes, uses binary pattern based on rates.

        Returns:
            Tensor of shape [T, 13, 8, 8]
        """
        rates = self._board_to_rates(board, legal_moves)

        # Create deterministic pattern: first T/2 timesteps spike if rate > 0.5
        spikes = np.zeros((self.time_steps, *rates.shape), dtype=np.float32)
        spike_mask = rates > 0.5

        # Spike in first half of time window
        for t in range(self.time_steps // 2):
            spikes[t] = spike_mask.astype(np.float32)

        return torch.from_numpy(spikes)

    def visualize_encoding(self, board: chess.Board, legal_moves: list = None) -> str:
        """
        Create human-readable visualization of encoding.

        Returns:
            String representation of encoded channels
        """
        rates = self._board_to_rates(board, legal_moves)

        channel_names = [
            "Own Pawns", "Own Knights", "Own Bishops", "Own Rooks", "Own Queens", "Own King",
            "Opp Pawns", "Opp Knights", "Opp Bishops", "Opp Rooks", "Opp Queens", "Opp King",
            "Legal Moves"
        ]

        output = []
        output.append(f"Board Encoding (firing rates):\n")
        output.append(f"Turn: {'White' if board.turn == chess.WHITE else 'Black'}\n")

        for c, name in enumerate(channel_names):
            output.append(f"\nChannel {c} - {name}:")
            channel_data = rates[c]

            # Print board-like representation
            for rank in range(7, -1, -1):
                row = []
                for file in range(8):
                    rate = channel_data[rank, file]
                    if rate > 0.5:
                        row.append("■")
                    else:
                        row.append("·")
                output.append(" ".join(row))

        return "\n".join(output)
