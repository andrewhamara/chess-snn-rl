"""Tests for board encoding."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import chess
from src.chess_env.encoder import BoardEncoder
from config.model_config import ModelConfig


def test_encoder_shape():
    """Test that encoder produces correct output shape."""
    encoder = BoardEncoder()
    board = chess.Board()

    spikes = encoder.encode(board)

    assert spikes.shape == (ModelConfig.TIME_STEPS, 13, 8, 8)
    assert spikes.dtype == torch.float32
    print("✓ Encoder shape test passed")


def test_encoder_with_legal_moves():
    """Test encoder with legal move mask."""
    encoder = BoardEncoder()
    board = chess.Board()
    legal_moves = list(board.legal_moves)

    spikes = encoder.encode(board, legal_moves)

    assert spikes.shape == (ModelConfig.TIME_STEPS, 13, 8, 8)

    # Check that legal move channel has some activity
    legal_move_channel = spikes[:, 12, :, :]
    assert legal_move_channel.sum() > 0
    print("✓ Encoder with legal moves test passed")


def test_encoder_deterministic():
    """Test deterministic encoding."""
    encoder = BoardEncoder()
    board = chess.Board()

    spikes = encoder.encode_position_deterministic(board)

    assert spikes.shape == (ModelConfig.TIME_STEPS, 13, 8, 8)
    print("✓ Deterministic encoder test passed")


def test_encoder_batch():
    """Test batch encoding."""
    encoder = BoardEncoder()
    boards = [chess.Board() for _ in range(4)]

    batch_spikes = encoder.encode_batch(boards)

    assert batch_spikes.shape == (4, ModelConfig.TIME_STEPS, 13, 8, 8)
    print("✓ Batch encoder test passed")


if __name__ == '__main__':
    test_encoder_shape()
    test_encoder_with_legal_moves()
    test_encoder_deterministic()
    test_encoder_batch()
    print("\nAll encoder tests passed!")
