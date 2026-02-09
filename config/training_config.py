"""Training hyperparameters."""

class TrainingConfig:
    # Self-play
    GAMES_PER_ITERATION = 200
    MAX_GAME_LENGTH = 200  # Moves

    # Replay buffer
    BUFFER_SIZE = 100_000
    BATCH_SIZE = 256

    # Training
    LEARNING_RATE = 1e-4
    EPOCHS_PER_ITERATION = 20
    GRADIENT_CLIP = 1.0

    # Loss coefficients
    VALUE_LOSS_COEF = 1.0
    ENTROPY_COEF = 0.01

    # Exploration
    INITIAL_TEMPERATURE = 1.0
    TEMPERATURE_DECAY = 0.995
    MIN_TEMPERATURE = 0.1

    # Checkpointing
    CHECKPOINT_INTERVAL = 10  # iterations
    SAVE_DIR = "checkpoints"

    # Logging
    LOG_INTERVAL = 1  # iterations
    TENSORBOARD_DIR = "runs"

    # Device
    DEVICE = "cuda"  # or "cpu"
