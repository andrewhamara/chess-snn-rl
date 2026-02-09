#!/usr/bin/env python3
"""Installation verification script."""

import sys

def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} (OK)")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} (Need 3.8+)")
        return False

def check_imports():
    """Check all required imports."""
    print("\nChecking required packages...")

    packages = [
        ("torch", "PyTorch"),
        ("spikingjelly", "SpikingJelly"),
        ("chess", "python-chess"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("matplotlib", "Matplotlib"),
        ("tensorboard", "TensorBoard"),
        ("yaml", "PyYAML"),
        ("tqdm", "tqdm"),
        ("pandas", "Pandas"),
    ]

    all_ok = True
    for package, name in packages:
        try:
            __import__(package)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} (missing)")
            all_ok = False

    return all_ok

def check_cuda():
    """Check CUDA availability."""
    print("\nChecking CUDA/GPU support...")
    try:
        import torch

        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            print(f"✓ CUDA available with {num_gpus} GPU(s)")

            for i in range(num_gpus):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"  GPU {i}: {name} ({mem:.1f} GB)")

            return True
        else:
            print("⚠ CUDA not available (CPU mode only)")
            return False
    except Exception as e:
        print(f"✗ Error checking CUDA: {e}")
        return False

def run_quick_tests():
    """Run quick functionality tests."""
    print("\nRunning quick tests...")

    try:
        # Test encoder
        print("Testing board encoder...")
        import torch
        import chess
        from src.chess_env.encoder import BoardEncoder

        encoder = BoardEncoder()
        board = chess.Board()
        spikes = encoder.encode(board)

        assert spikes.shape == (16, 13, 8, 8), "Encoder shape mismatch"
        print("✓ Board encoder works")

        # Test move encoder
        print("Testing move encoder...")
        from src.chess_env.move_encoding import MoveEncoder

        move_encoder = MoveEncoder()
        legal_moves = list(board.legal_moves)
        move = legal_moves[0]

        action = move_encoder.move_to_action(move, board)
        decoded = move_encoder.action_to_move(action, board)

        assert decoded == move, "Move encoding/decoding mismatch"
        print("✓ Move encoder works")

        # Test model
        print("Testing SNN model...")
        from src.models.chess_snn import ChessSNN

        model = ChessSNN()
        spikes_batch = spikes.unsqueeze(0)
        policy_logits, value = model(spikes_batch)

        assert policy_logits.shape == (1, 4672), "Policy shape mismatch"
        assert value.shape == (1, 1), "Value shape mismatch"
        assert -1 <= value.item() <= 1, "Value out of bounds"
        print("✓ SNN model works")

        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all verification checks."""
    print("="*60)
    print("Chess SNN RL - Installation Verification")
    print("="*60)

    checks = [
        ("Python Version", check_python_version),
        ("Package Imports", check_imports),
        ("CUDA Support", check_cuda),
        ("Functionality Tests", run_quick_tests),
    ]

    results = {}
    for name, check_fn in checks:
        results[name] = check_fn()

    print("\n" + "="*60)
    print("Summary")
    print("="*60)

    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    if all(results.values()):
        print("\n🎉 All checks passed! You're ready to train.")
        print("\nQuick start:")
        print("  python scripts/train.py --iterations 10 --games-per-iter 50 --device cuda")
        return 0
    else:
        print("\n⚠ Some checks failed. Please install missing packages:")
        print("  pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())
