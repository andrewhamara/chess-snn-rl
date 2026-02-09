"""Baseline opponent implementations for evaluation."""

import chess
import random
from typing import List, Optional


class BaseOpponent:
    """Base class for opponents."""

    def __init__(self, name: str):
        self.name = name

    def select_move(self, board: chess.Board) -> chess.Move:
        """Select a move given the board state."""
        raise NotImplementedError

    def reset(self):
        """Reset opponent state between games."""
        pass


class RandomPlayer(BaseOpponent):
    """
    Selects moves uniformly at random from legal moves.
    Baseline Elo: ~0
    """

    def __init__(self):
        super().__init__("Random")

    def select_move(self, board: chess.Board) -> chess.Move:
        """Select random legal move."""
        legal_moves = list(board.legal_moves)
        return random.choice(legal_moves)


class GreedyMaterialPlayer(BaseOpponent):
    """
    Greedy player that maximizes immediate material gain.
    Baseline Elo: ~800

    Piece values:
    - Pawn: 1
    - Knight/Bishop: 3
    - Rook: 5
    - Queen: 9
    """

    PIECE_VALUES = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0  # King capture ends game
    }

    def __init__(self):
        super().__init__("GreedyMaterial")

    def select_move(self, board: chess.Board) -> chess.Move:
        """Select move that maximizes material gain."""
        legal_moves = list(board.legal_moves)

        # Evaluate each move
        best_moves = []
        best_value = float('-inf')

        for move in legal_moves:
            value = self._evaluate_move(board, move)

            if value > best_value:
                best_value = value
                best_moves = [move]
            elif value == best_value:
                best_moves.append(move)

        # Return random among best moves
        return random.choice(best_moves)

    def _evaluate_move(self, board: chess.Board, move: chess.Move) -> float:
        """
        Evaluate move value based on material.

        Returns:
            Material value change (positive = gain, negative = loss)
        """
        value = 0.0

        # Capture value
        if board.is_capture(move):
            captured_piece = board.piece_at(move.to_square)
            if captured_piece:
                value += self.PIECE_VALUES[captured_piece.piece_type]

        # Promotion value
        if move.promotion:
            value += self.PIECE_VALUES[move.promotion] - self.PIECE_VALUES[chess.PAWN]

        # Simple check bonus
        board_copy = board.copy()
        board_copy.push(move)
        if board_copy.is_check():
            value += 0.1

        # Checkmate bonus
        if board_copy.is_checkmate():
            value += 1000

        return value


class MinimaxPlayer(BaseOpponent):
    """
    Minimax player with alpha-beta pruning.
    Baseline Elo: ~1200 (depth 2), ~1600 (depth 3)
    """

    PIECE_VALUES = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000
    }

    # Position bonuses (simplified)
    PAWN_TABLE = [
        0,  0,  0,  0,  0,  0,  0,  0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
        5,  5, 10, 25, 25, 10,  5,  5,
        0,  0,  0, 20, 20,  0,  0,  0,
        5, -5,-10,  0,  0,-10, -5,  5,
        5, 10, 10,-20,-20, 10, 10,  5,
        0,  0,  0,  0,  0,  0,  0,  0
    ]

    def __init__(self, depth: int = 2):
        super().__init__(f"Minimax-D{depth}")
        self.depth = depth

    def select_move(self, board: chess.Board) -> chess.Move:
        """Select move using minimax with alpha-beta pruning."""
        best_move = None
        best_value = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        for move in board.legal_moves:
            board.push(move)
            value = -self._minimax(board, self.depth - 1, -beta, -alpha, False)
            board.pop()

            if value > best_value:
                best_value = value
                best_move = move

            alpha = max(alpha, value)

        return best_move if best_move else random.choice(list(board.legal_moves))

    def _minimax(self, board: chess.Board, depth: int,
                alpha: float, beta: float, maximizing: bool) -> float:
        """
        Minimax with alpha-beta pruning.

        Returns:
            Evaluation score from current player's perspective
        """
        if depth == 0 or board.is_game_over():
            return self._evaluate_board(board)

        if maximizing:
            max_eval = float('-inf')
            for move in board.legal_moves:
                board.push(move)
                eval_score = self._minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in board.legal_moves:
                board.push(move)
                eval_score = self._minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval

    def _evaluate_board(self, board: chess.Board) -> float:
        """
        Evaluate board position.

        Returns:
            Score from current player's perspective (positive = better)
        """
        if board.is_checkmate():
            return -20000
        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        score = 0
        current_player = board.turn

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue

            # Material value
            value = self.PIECE_VALUES[piece.piece_type]

            # Position bonus for pawns
            if piece.piece_type == chess.PAWN:
                if piece.color == chess.WHITE:
                    value += self.PAWN_TABLE[square]
                else:
                    value += self.PAWN_TABLE[chess.square_mirror(square)]

            # Add or subtract based on piece color
            if piece.color == current_player:
                score += value
            else:
                score -= value

        return score
