"""Evaluation settings."""

class EvalConfig:
    # Elo system
    INITIAL_ELO = 1000
    K_FACTOR = 32

    # Evaluation games
    GAMES_PER_OPPONENT = 100
    MAX_GAME_LENGTH = 200

    # Opponent types
    OPPONENTS = [
        "random",
        "greedy_material",
        "minimax_depth2"
    ]

    # Performance thresholds
    LEGAL_MOVE_THRESHOLD = 0.95
    RANDOM_WIN_RATE_THRESHOLD = 0.70
    GREEDY_WIN_RATE_THRESHOLD = 0.60
    MINIMAX_WIN_RATE_THRESHOLD = 0.50

    # Elo milestones
    ELO_MILESTONES = {
        "beginner": 600,
        "intermediate": 1000,
        "advanced": 1200
    }
