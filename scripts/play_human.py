"""Interactive play against the trained agent."""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import chess
from src.models.chess_snn import ChessSNN
from src.chess_env.board import ChessBoard
from src.chess_env.encoder import BoardEncoder
from src.chess_env.move_encoding import MoveEncoder
from config.model_config import ModelConfig


def print_board(board: chess.Board):
    """Print board in a nice format."""
    print("\n" + str(board))
    print(f"\nFEN: {board.fen()}")
    print(f"Turn: {'White' if board.turn == chess.WHITE else 'Black'}")


def get_human_move(board: chess.Board) -> chess.Move:
    """Get move from human player."""
    legal_moves = list(board.legal_moves)

    print("\nLegal moves:")
    for i, move in enumerate(legal_moves):
        if (i + 1) % 8 == 0:
            print(move.uci())
        else:
            print(move.uci(), end="  ")
    print()

    while True:
        move_str = input("\nEnter your move (UCI format, e.g., e2e4): ").strip()

        if move_str.lower() == 'quit':
            return None

        try:
            move = chess.Move.from_uci(move_str)
            if move in legal_moves:
                return move
            else:
                print("Illegal move! Try again.")
        except:
            print("Invalid move format! Use UCI format (e.g., e2e4)")


def get_agent_move(model, board: chess.Board, encoder: BoardEncoder,
                   move_encoder: MoveEncoder, device: str) -> chess.Move:
    """Get move from agent."""
    legal_moves = list(board.legal_moves)
    state = encoder.encode(board, legal_moves)
    legal_mask = torch.from_numpy(move_encoder.get_legal_action_mask(board))

    state = state.to(device)
    legal_mask = legal_mask.to(device)

    with torch.no_grad():
        action, log_prob, value = model.select_action(
            state, legal_mask, temperature=0.1, deterministic=True
        )

    move = move_encoder.action_to_move(action, board)

    if move is None or move not in legal_moves:
        # Fallback to random
        print("Agent selected illegal move, choosing random...")
        move = legal_moves[0]

    print(f"\nAgent move: {move.uci()}")
    print(f"Agent value estimate: {value.item():.3f}")

    return move


def main():
    parser = argparse.ArgumentParser(description="Play against Chess SNN")
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--play-white', action='store_true',
                       help='Play as white (default: play as black)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to run on')

    args = parser.parse_args()

    # Load model
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = ChessSNN(ModelConfig())

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print("=== Play Against Chess SNN ===")
    print(f"Loaded model from iteration {checkpoint.get('iteration', 'unknown')}")
    print(f"You are playing as {'White' if args.play_white else 'Black'}")
    print("Enter 'quit' to exit")

    # Initialize
    board = ChessBoard()
    encoder = BoardEncoder()
    move_encoder = MoveEncoder()

    human_is_white = args.play_white

    # Game loop
    while not board.is_game_over():
        print_board(board.board)

        human_turn = (board.turn == chess.WHITE) == human_is_white

        if human_turn:
            print("\nYour turn:")
            move = get_human_move(board.board)
            if move is None:
                print("Game quit by user")
                break
        else:
            print("\nAgent's turn:")
            move = get_agent_move(model, board.board, encoder, move_encoder, device)

        board.make_move(move)

    # Game over
    if board.is_game_over():
        print_board(board.board)
        termination, outcome = board.get_outcome()
        print(f"\nGame over: {termination}")

        if outcome > 0:
            winner = "White"
        elif outcome < 0:
            winner = "Black"
        else:
            winner = "Draw"

        print(f"Result: {winner}")


if __name__ == '__main__':
    main()
