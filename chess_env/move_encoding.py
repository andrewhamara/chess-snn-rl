"""Move encoding/decoding for action space."""

import chess
import numpy as np
from typing import List, Tuple, Optional


class MoveEncoder:
    """
    Encodes chess moves to action indices and vice versa.

    Action space: 4672 = 64 from_squares × 73 move types
    Move types:
    - 56 queen moves (8 directions × 7 distances)
    - 8 knight moves
    - 9 underpromotions (3 directions × 3 piece types: N, B, R)
    """

    # Direction vectors for queen moves (includes rook + bishop)
    QUEEN_DIRECTIONS = [
        (-1, 0), (1, 0), (0, -1), (0, 1),  # Rook directions
        (-1, -1), (-1, 1), (1, -1), (1, 1)  # Bishop directions
    ]

    # Knight move offsets
    KNIGHT_MOVES = [
        (-2, -1), (-2, 1), (-1, -2), (-1, 2),
        (1, -2), (1, 2), (2, -1), (2, 1)
    ]

    # Underpromotion directions (forward-left, forward, forward-right)
    UNDERPROMOTION_DIRECTIONS = [(-1, 1), (0, 1), (1, 1)]
    UNDERPROMOTION_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]

    def __init__(self):
        """Initialize move encoder with lookup tables."""
        self.action_to_move_cache = {}
        self.move_to_action_cache = {}
        self._build_move_type_mappings()

    def _build_move_type_mappings(self):
        """Build mappings between move types and indices."""
        self.move_type_to_idx = {}
        self.idx_to_move_type = {}

        idx = 0

        # Queen moves (56 total)
        for direction in self.QUEEN_DIRECTIONS:
            for distance in range(1, 8):
                self.move_type_to_idx[(direction, distance, None)] = idx
                self.idx_to_move_type[idx] = (direction, distance, None)
                idx += 1

        # Knight moves (8 total)
        for knight_move in self.KNIGHT_MOVES:
            self.move_type_to_idx[(knight_move, 1, None)] = idx
            self.idx_to_move_type[idx] = (knight_move, 1, None)
            idx += 1

        # Underpromotions (9 total)
        for direction in self.UNDERPROMOTION_DIRECTIONS:
            for piece in self.UNDERPROMOTION_PIECES:
                self.move_type_to_idx[(direction, 1, piece)] = idx
                self.idx_to_move_type[idx] = (direction, 1, piece)
                idx += 1

        assert idx == 73, f"Expected 73 move types, got {idx}"

    def move_to_action(self, move: chess.Move, board: chess.Board) -> Optional[int]:
        """
        Convert a chess.Move to an action index.

        Args:
            move: The chess move
            board: Current board state (needed for context)

        Returns:
            Action index [0, 4671] or None if invalid
        """
        from_square = move.from_square
        to_square = move.to_square

        # Get from/to coordinates
        from_file, from_rank = chess.square_file(from_square), chess.square_rank(from_square)
        to_file, to_rank = chess.square_file(to_square), chess.square_rank(to_square)

        # Calculate direction
        d_file = to_file - from_file
        d_rank = to_rank - from_rank

        # Check if it's a knight move
        if (d_file, d_rank) in self.KNIGHT_MOVES:
            # Knight move
            direction = (d_file, d_rank)
            distance = 1
            promotion = None
        elif move.promotion and move.promotion != chess.QUEEN:
            # Underpromotion
            direction = (np.sign(d_file), np.sign(d_rank))
            distance = 1
            promotion = move.promotion
        else:
            # Regular move or queen promotion
            max_abs = max(abs(d_file), abs(d_rank))
            if max_abs == 0:
                return None
            direction = (d_file // max_abs if d_file != 0 else 0,
                        d_rank // max_abs if d_rank != 0 else 0)
            distance = max_abs
            promotion = None

        move_type_key = (direction, distance, promotion)

        if move_type_key not in self.move_type_to_idx:
            return None

        move_type_idx = self.move_type_to_idx[move_type_key]
        action = from_square * 73 + move_type_idx

        return action

    def action_to_move(self, action: int, board: chess.Board) -> Optional[chess.Move]:
        """
        Convert action index to chess.Move.

        Args:
            action: Action index [0, 4671]
            board: Current board state (needed to validate)

        Returns:
            chess.Move or None if invalid
        """
        if action < 0 or action >= 4672:
            return None

        from_square = action // 73
        move_type_idx = action % 73

        if move_type_idx not in self.idx_to_move_type:
            return None

        direction, distance, promotion = self.idx_to_move_type[move_type_idx]

        # Calculate to_square
        from_file, from_rank = chess.square_file(from_square), chess.square_rank(from_square)
        to_file = from_file + direction[0] * distance
        to_rank = from_rank + direction[1] * distance

        # Check bounds
        if not (0 <= to_file < 8 and 0 <= to_rank < 8):
            return None

        to_square = chess.square(to_file, to_rank)

        # Handle queen promotions (default promotion)
        if promotion is None and to_rank in [0, 7]:
            piece = board.piece_at(from_square)
            if piece and piece.piece_type == chess.PAWN:
                promotion = chess.QUEEN

        move = chess.Move(from_square, to_square, promotion=promotion)

        return move if move in board.legal_moves else None

    def get_legal_action_mask(self, board: chess.Board) -> np.ndarray:
        """
        Get binary mask for legal actions.

        Args:
            board: Current board state

        Returns:
            Binary mask [4672] where 1 = legal, 0 = illegal
        """
        mask = np.zeros(4672, dtype=np.float32)

        for move in board.legal_moves:
            action = self.move_to_action(move, board)
            if action is not None:
                mask[action] = 1.0

        return mask

    def encode_moves(self, moves: List[chess.Move], board: chess.Board) -> List[int]:
        """Encode list of moves to action indices."""
        actions = []
        for move in moves:
            action = self.move_to_action(move, board)
            if action is not None:
                actions.append(action)
        return actions

    def decode_actions(self, actions: List[int], board: chess.Board) -> List[chess.Move]:
        """Decode list of action indices to moves."""
        moves = []
        for action in actions:
            move = self.action_to_move(action, board)
            if move is not None:
                moves.append(move)
        return moves
