"""Evaluation script for benchmarking."""

import sys
import argparse
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.models.chess_snn import ChessSNN
from src.evaluation.evaluator import Evaluator
from config.eval_config import EvalConfig
from config.model_config import ModelConfig


def main():
    parser = argparse.ArgumentParser(description="Evaluate Chess SNN")
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--games', type=int, default=100,
                       help='Number of games per opponent')
    parser.add_argument('--quick', action='store_true',
                       help='Quick evaluation (20 games vs random + greedy)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to run on')
    parser.add_argument('--output', type=str, default='eval_results.json',
                       help='Output file for results')

    args = parser.parse_args()

    print("=== Chess SNN Evaluation ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")

    # Load model
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = ChessSNN(ModelConfig())

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Loaded model from iteration {checkpoint.get('iteration', 'unknown')}")

    # Create evaluator
    config = EvalConfig()
    config.GAMES_PER_OPPONENT = args.games
    evaluator = Evaluator(model, device=str(device), config=config)

    # Run evaluation
    if args.quick:
        results = evaluator.quick_evaluation(num_games=20)
    else:
        results = evaluator.full_evaluation()

    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
