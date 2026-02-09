"""Chess board wrapper using python-chess."""

import chess
from typing import List, Optional, Tuple


class ChessBoard:
    """Wrapper around python-chess Board with RL-specific methods."""

    def __init__(self, fen: Optional[str] = None):
        """Initialize board from FEN or starting position."""
        self.board = chess.Board(fen) if fen else chess.Board()

    def reset(self) -> None:
        """Reset to starting position."""
        self.board.reset()

    def get_legal_moves(self) -> List[chess.Move]:
        """Get list of legal moves in current position."""
        return list(self.board.legal_moves)

    def make_move(self, move: chess.Move) -> bool:
        """
        Make a move on the board.

        Returns:
            True if move was legal and made, False otherwise.
        """
        if move in self.board.legal_moves:
            self.board.push(move)
            return True
        return False

    def is_game_over(self) -> bool:
        """Check if game is over."""
        return self.board.is_game_over()

    def get_result(self) -> float:
        """
        Get game result from WHITE's perspective.

        Returns:
            1.0 for white win, 0.0 for draw, -1.0 for black win
        """
        if not self.board.is_game_over():
            return 0.0

        result = self.board.result()
        if result == "1/2-1/2":
            return 0.0
        elif result == "1-0":
            return 1.0  # White won
        elif result == "0-1":
            return -1.0  # Black won
        else:
            return 0.0  # Unknown result, treat as draw

    def get_outcome(self) -> Tuple[Optional[str], float]:
        """
        Get game outcome details.

        Returns:
            (termination_reason, result_value)
        """
        if not self.board.is_game_over():
            return None, 0.0

        outcome = self.board.outcome()
        result_value = self.get_result()

        if outcome.termination == chess.Termination.CHECKMATE:
            reason = "checkmate"
        elif outcome.termination == chess.Termination.STALEMATE:
            reason = "stalemate"
        elif outcome.termination == chess.Termination.INSUFFICIENT_MATERIAL:
            reason = "insufficient_material"
        elif outcome.termination == chess.Termination.SEVENTYFIVE_MOVES:
            reason = "75_move_rule"
        elif outcome.termination == chess.Termination.FIVEFOLD_REPETITION:
            reason = "repetition"
        else:
            reason = "unknown"

        return reason, result_value

    def copy(self) -> 'ChessBoard':
        """Create a copy of the board."""
        return ChessBoard(self.board.fen())

    def fen(self) -> str:
        """Get FEN representation."""
        return self.board.fen()

    def __str__(self) -> str:
        """String representation of board."""
        return str(self.board)

    @property
    def turn(self) -> chess.Color:
        """Get current turn."""
        return self.board.turn

    @property
    def fullmove_number(self) -> int:
        """Get fullmove number."""
        return self.board.fullmove_number

    @property
    def halfmove_clock(self) -> int:
        """Get halfmove clock."""
        return self.board.halfmove_clock
