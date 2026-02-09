"""Tests for SNN model."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import chess
from src.models.chess_snn import ChessSNN
from src.chess_env.encoder import BoardEncoder
from src.chess_env.move_encoding import MoveEncoder
from config.model_config import ModelConfig


def test_model_forward():
    """Test model forward pass."""
    model = ChessSNN()
    encoder = BoardEncoder()
    board = chess.Board()

    # Encode board
    spikes = encoder.encode(board).unsqueeze(0)  # Add batch dim

    # Forward pass
    policy_logits, value = model(spikes)

    assert policy_logits.shape == (1, 4672)
    assert value.shape == (1, 1)
    assert -1 <= value.item() <= 1
    print("✓ Model forward pass test passed")


def test_model_with_mask():
    """Test model with legal move masking."""
    model = ChessSNN()
    encoder = BoardEncoder()
    move_encoder = MoveEncoder()
    board = chess.Board()

    spikes = encoder.encode(board).unsqueeze(0)
    legal_mask = torch.from_numpy(move_encoder.get_legal_action_mask(board)).unsqueeze(0)

    policy_logits, value = model(spikes, legal_mask)

    # Check that illegal moves have very low logits
    illegal_mask = (1 - legal_mask).bool()
    illegal_logits = policy_logits[illegal_mask]
    assert (illegal_logits < -1e8).all()
    print("✓ Model with mask test passed")


def test_model_action_selection():
    """Test action selection."""
    model = ChessSNN()
    encoder = BoardEncoder()
    move_encoder = MoveEncoder()
    board = chess.Board()

    spikes = encoder.encode(board)
    legal_mask = torch.from_numpy(move_encoder.get_legal_action_mask(board))

    # Test deterministic selection
    action, log_prob, value = model.select_action(spikes, legal_mask, deterministic=True)

    assert isinstance(action, int)
    assert 0 <= action < 4672
    assert -1 <= value.item() <= 1

    # Verify action is legal
    move = move_encoder.action_to_move(action, board)
    assert move is not None
    assert move in board.legal_moves
    print("✓ Model action selection test passed")


def test_model_batch():
    """Test model with batch input."""
    model = ChessSNN()
    encoder = BoardEncoder()

    boards = [chess.Board() for _ in range(4)]
    batch_spikes = encoder.encode_batch(boards)

    policy_logits, values = model(batch_spikes)

    assert policy_logits.shape == (4, 4672)
    assert values.shape == (4, 1)
    print("✓ Model batch test passed")


def test_gradient_flow():
    """Test that gradients flow through the model."""
    model = ChessSNN()
    encoder = BoardEncoder()
    board = chess.Board()

    spikes = encoder.encode(board).unsqueeze(0)

    policy_logits, value = model(spikes)

    # Compute dummy loss
    loss = policy_logits.sum() + value.sum()
    loss.backward()

    # Check that gradients exist
    has_gradients = False
    for param in model.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_gradients = True
            break

    assert has_gradients
    print("✓ Gradient flow test passed")


if __name__ == '__main__':
    test_model_forward()
    test_model_with_mask()
    test_model_action_selection()
    test_model_batch()
    test_gradient_flow()
    print("\nAll model tests passed!")
