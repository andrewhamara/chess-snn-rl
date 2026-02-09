#!/usr/bin/env python3
"""Quick test for multiprocessing tensor serialization fix."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from src.models.chess_snn import ChessSNN
from src.training.parallel_self_play import ParallelSelfPlayEngine
from config.model_config import ModelConfig


def main():
    print("=" * 70)
    print("  Testing Multi-GPU Fix (Tensor Serialization)")
    print("=" * 70)

    # Create model
    model = ChessSNN(ModelConfig())

    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"\nDetected {num_gpus} GPUs")

    if num_gpus == 0:
        print("No GPUs available, testing with CPU")
        gpu_ids = [None, None]  # Test with 2 CPU workers
    else:
        # Use 2 GPUs for quick test
        gpu_ids = list(range(min(num_gpus, 2)))
        print(f"Using GPUs: {gpu_ids}")

    # Create parallel self-play engine
    engine = ParallelSelfPlayEngine(model, gpu_ids=gpu_ids, max_game_length=50)

    # Generate small number of games with progress tracking
    print("\n" + "=" * 70)
    print("  Generating 10 test games")
    print("=" * 70)

    try:
        games = engine.generate_games(
            num_games=10,
            temperature=1.0,
            show_progress=True
        )

        print("\n" + "=" * 70)
        print("  Results")
        print("=" * 70)
        print(f"Total games generated: {len(games)}")

        # Verify games have correct structure
        if len(games) > 0:
            game = games[0]
            print(f"Game structure check:")
            print(f"  - states: {len(game['states'])} positions")
            print(f"  - actions: {len(game['actions'])} actions")
            print(f"  - legal_masks: {len(game['legal_masks'])} masks")
            print(f"  - outcome: {game['outcome']}")
            print(f"  - move_count: {game['move_count']}")

            # Check that tensors are properly converted
            state_type = type(game['states'][0])
            print(f"\n  State type: {state_type}")
            if torch.is_tensor(game['states'][0]):
                print("  ✓ States are PyTorch tensors (correctly converted)")
            else:
                print("  ✗ States are not tensors!")

        print("\n" + "=" * 70)
        print("✓ Multi-GPU test completed successfully!")
        print("=" * 70)

    except Exception as e:
        print("\n" + "=" * 70)
        print("✗ Test FAILED with error:")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
