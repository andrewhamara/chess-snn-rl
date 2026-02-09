"""SNN model architecture configuration."""

class ModelConfig:
    # Input encoding
    TIME_STEPS = 16
    INPUT_CHANNELS = 13  # 6 own + 6 opponent + 1 legal move mask
    BOARD_SIZE = 8

    # Neuron model
    NEURON_TAU = 2.0
    SURROGATE_ALPHA = 2.0  # ATan surrogate gradient parameter

    # Convolutional layers
    CONV_CHANNELS = [64, 128, 256]
    KERNEL_SIZES = [3, 3, 3]

    # Policy head
    POLICY_HIDDEN_DIM = 1024
    ACTION_SPACE_SIZE = 4672  # 64 from_squares × 73 move types

    # Value head
    VALUE_HIDDEN_DIM = 128

    # Initialization
    INIT_GAIN = 1.0
