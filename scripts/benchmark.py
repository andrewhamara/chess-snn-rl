#!/usr/bin/env python3
"""Command-line benchmarking script with text output."""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import json
from datetime import datetime

from src.models.chess_snn import ChessSNN
from src.evaluation.evaluator import Evaluator
from config.eval_config import EvalConfig
from config.model_config import ModelConfig


def print_separator(char='=', length=70):
    """Print a separator line."""
    print(char * length)


def print_section(title):
    """Print a section header."""
    print()
    print_separator()
    print(f"  {title}")
    print_separator()


def print_results_table(results):
    """Print results in a formatted table."""
    print()
    print("OPPONENT RESULTS:")
    print_separator('-')
    print(f"{'Opponent':<20} {'Games':<8} {'W-D-L':<15} {'Win %':<10} {'Elo':<10}")
    print_separator('-')

    for opponent_name, data in results.items():
        wins = data['wins']
        draws = data['draws']
        losses = data['losses']
        total = data['num_games']
        win_rate = data['win_rate'] * 100
        perf_elo = int(data['performance_rating'])

        wdl = f"{wins}-{draws}-{losses}"

        print(f"{data['opponent']:<20} {total:<8} {wdl:<15} {win_rate:>6.1f}%   {perf_elo:<10}")

    print_separator('-')


def print_summary(summary):
    """Print evaluation summary."""
    print_section("EVALUATION SUMMARY")

    total = summary['total_games']
    wins = summary['total_wins']
    draws = summary['total_draws']
    losses = summary['total_losses']

    print(f"Total Games:     {total}")
    print(f"Total Wins:      {wins} ({summary['overall_win_rate']*100:.1f}%)")
    print(f"Total Draws:     {draws} ({summary['overall_draw_rate']*100:.1f}%)")
    print(f"Total Losses:    {losses} ({summary['overall_loss_rate']*100:.1f}%)")
    print(f"Avg Perf Elo:    {int(summary['avg_performance_rating'])}")


def print_milestones(summary, config):
    """Print milestone achievements."""
    print_section("ELO MILESTONES")

    elo = summary['avg_performance_rating']

    for name, threshold in config.ELO_MILESTONES.items():
        achieved = "✓" if elo >= threshold else "✗"
        status = "ACHIEVED" if elo >= threshold else "not yet"
        print(f"{achieved} {name.capitalize():<15} {threshold} Elo  [{status}]")


def print_detailed_stats(results):
    """Print detailed statistics per opponent."""
    print_section("DETAILED STATISTICS")

    for opponent_name, data in results.items():
        print(f"\n{data['opponent']}:")
        print(f"  Games:           {data['num_games']}")
        print(f"  Win Rate:        {data['win_rate']*100:.1f}%")
        print(f"  Draw Rate:       {data['draw_rate']*100:.1f}%")
        print(f"  Loss Rate:       {data['loss_rate']*100:.1f}%")
        print(f"  Score:           {data['score']:.3f}")
        print(f"  Performance Elo: {int(data['performance_rating'])}")
        print(f"  Avg Moves:       {data['avg_moves']:.1f}")

        if data.get('terminations'):
            print(f"  Terminations:")
            for term, rate in data['terminations'].items():
                print(f"    {term}: {rate*100:.1f}%")


def save_results_txt(summary, output_path):
    """Save results to a text file."""
    with open(output_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("  CHESS SNN EVALUATION RESULTS\n")
        f.write("="*70 + "\n\n")

        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Games: {summary['total_games']}\n")
        f.write(f"Average Performance Elo: {int(summary['avg_performance_rating'])}\n\n")

        f.write("OPPONENT RESULTS:\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Opponent':<20} {'Games':<8} {'W-D-L':<15} {'Win %':<10} {'Elo':<10}\n")
        f.write("-"*70 + "\n")

        for opponent_name, data in summary['opponent_results'].items():
            wins = data['wins']
            draws = data['draws']
            losses = data['losses']
            total = data['num_games']
            win_rate = data['win_rate'] * 100
            perf_elo = int(data['performance_rating'])
            wdl = f"{wins}-{draws}-{losses}"

            f.write(f"{data['opponent']:<20} {total:<8} {wdl:<15} {win_rate:>6.1f}%   {perf_elo:<10}\n")

        f.write("-"*70 + "\n")

    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Chess SNN model (CLI-friendly)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick benchmark (20 games)
  python scripts/benchmark.py --checkpoint model.pt --quick

  # Full benchmark (100 games)
  python scripts/benchmark.py --checkpoint model.pt --games 100

  # Save to file
  python scripts/benchmark.py --checkpoint model.pt --output results.txt
        """
    )

    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--games', type=int, default=100,
                       help='Games per opponent (default: 100)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode (20 games vs random + greedy)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to run on')
    parser.add_argument('--output', type=str, default=None,
                       help='Save results to text file')
    parser.add_argument('--json', type=str, default=None,
                       help='Save results to JSON file')
    parser.add_argument('--quiet', action='store_true',
                       help='Minimal output (just final summary)')

    args = parser.parse_args()

    if not args.quiet:
        print_section("CHESS SNN BENCHMARK")
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Device: {args.device}")
        print(f"Mode: {'Quick (20 games)' if args.quick else f'Full ({args.games} games)'}")

    # Load model
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = ChessSNN(ModelConfig())

    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        iteration = checkpoint.get('iteration', 'unknown')

        if not args.quiet:
            print(f"Loaded model from iteration: {iteration}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return 1

    model = model.to(device)
    model.eval()

    # Create evaluator
    config = EvalConfig()
    config.GAMES_PER_OPPONENT = 20 if args.quick else args.games
    evaluator = Evaluator(model, device=str(device), config=config)

    # Run evaluation
    if not args.quiet:
        print_section("RUNNING EVALUATION")

    if args.quick:
        summary = evaluator.quick_evaluation(num_games=20)
    else:
        summary = evaluator.full_evaluation()

    # Print results
    if not args.quiet:
        print_results_table(summary.get('opponent_results', summary))
        print_summary(summary)
        print_milestones(summary, config)
        print_detailed_stats(summary.get('opponent_results', summary))
    else:
        # Quiet mode - just summary
        print(f"Avg Elo: {int(summary['avg_performance_rating'])}")
        print(f"Win rate: {summary['overall_win_rate']*100:.1f}%")

    # Save results
    if args.output:
        save_results_txt(summary, args.output)

    if args.json:
        with open(args.json, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"JSON results saved to: {args.json}")

    if not args.quiet:
        print_section("BENCHMARK COMPLETE")

    return 0


if __name__ == '__main__':
    sys.exit(main())
