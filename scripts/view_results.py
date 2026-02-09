#!/usr/bin/env python3
"""View training/evaluation results from JSON files (CLI-friendly)."""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime


def print_separator(char='=', length=70):
    """Print separator line."""
    print(char * length)


def print_section(title):
    """Print section header."""
    print()
    print_separator()
    print(f"  {title}")
    print_separator()


def view_training_stats(json_path):
    """View training statistics."""
    with open(json_path) as f:
        data = json.load(f)

    print_section("TRAINING STATISTICS")
    print(f"Total iterations: {len(data)}")

    if len(data) == 0:
        print("No training data found.")
        return

    latest = data[-1]
    print(f"Latest iteration: {latest['iteration']}")
    print(f"Temperature: {latest['temperature']:.3f}")

    # Game stats
    print("\nLatest Game Stats:")
    game_stats = latest['game_stats']
    print(f"  Total games:     {game_stats['total_games']}")
    print(f"  Avg moves:       {game_stats['avg_moves']:.1f}")
    print(f"  White win rate:  {game_stats['white_win_rate']*100:.1f}%")
    print(f"  Draw rate:       {game_stats['draw_rate']*100:.1f}%")
    print(f"  Black win rate:  {game_stats['black_win_rate']*100:.1f}%")

    # Training losses
    print("\nLatest Training Losses:")
    losses = latest['training_losses']
    print(f"  Total loss:      {losses['total_loss']:.4f}")
    print(f"  Policy loss:     {losses['policy_loss']:.4f}")
    print(f"  Value loss:      {losses['value_loss']:.4f}")
    print(f"  Entropy:         {losses['entropy']:.4f}")

    # Buffer stats
    print("\nReplay Buffer:")
    buffer_stats = latest['buffer_stats']
    print(f"  Size:            {buffer_stats['size']}")
    print(f"  Win rate:        {buffer_stats['win_rate']*100:.1f}%")

    # Training progression
    print_section("TRAINING PROGRESSION")
    print(f"{'Iter':<8} {'Avg Moves':<12} {'Draw Rate':<12} {'Total Loss':<12}")
    print_separator('-')

    for i, entry in enumerate(data):
        if i % (len(data) // 10 or 1) == 0 or i == len(data) - 1:
            iter_num = entry['iteration']
            avg_moves = entry['game_stats']['avg_moves']
            draw_rate = entry['game_stats']['draw_rate'] * 100
            total_loss = entry['training_losses']['total_loss']

            print(f"{iter_num:<8} {avg_moves:<12.1f} {draw_rate:<11.1f}% {total_loss:<12.4f}")


def view_eval_results(json_path):
    """View evaluation results."""
    with open(json_path) as f:
        data = json.load(f)

    print_section("EVALUATION RESULTS")

    # Summary
    print(f"Total games:     {data['total_games']}")
    print(f"Total wins:      {data['total_wins']} ({data['overall_win_rate']*100:.1f}%)")
    print(f"Total draws:     {data['total_draws']} ({data['overall_draw_rate']*100:.1f}%)")
    print(f"Total losses:    {data['total_losses']} ({data['overall_loss_rate']*100:.1f}%)")
    print(f"Avg Perf Elo:    {int(data['avg_performance_rating'])}")

    # Opponent results
    print_section("OPPONENT RESULTS")
    print(f"{'Opponent':<20} {'Games':<8} {'W-D-L':<15} {'Win %':<10} {'Elo':<10}")
    print_separator('-')

    for opponent_name, opp_data in data['opponent_results'].items():
        wins = opp_data['wins']
        draws = opp_data['draws']
        losses = opp_data['losses']
        total = opp_data['num_games']
        win_rate = opp_data['win_rate'] * 100
        perf_elo = int(opp_data['performance_rating'])
        wdl = f"{wins}-{draws}-{losses}"

        print(f"{opp_data['opponent']:<20} {total:<8} {wdl:<15} {win_rate:>6.1f}%   {perf_elo:<10}")

    print_separator('-')


def view_progress(json_path, last_n=10):
    """View last N iterations progress."""
    with open(json_path) as f:
        data = json.load(f)

    recent = data[-last_n:] if len(data) > last_n else data

    print_section(f"LAST {len(recent)} ITERATIONS")
    print(f"{'Iter':<6} {'Games':<8} {'Avg Moves':<10} {'Draw%':<8} {'Loss':<10}")
    print_separator('-')

    for entry in recent:
        iter_num = entry['iteration']
        games = entry['game_stats']['total_games']
        avg_moves = entry['game_stats']['avg_moves']
        draw_rate = entry['game_stats']['draw_rate'] * 100
        total_loss = entry['training_losses']['total_loss']

        print(f"{iter_num:<6} {games:<8} {avg_moves:<10.1f} {draw_rate:<7.1f}% {total_loss:<10.4f}")


def compare_checkpoints(paths):
    """Compare multiple evaluation results."""
    print_section("CHECKPOINT COMPARISON")

    results = []
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        results.append({
            'path': Path(path).name,
            'elo': int(data['avg_performance_rating']),
            'win_rate': data['overall_win_rate'] * 100
        })

    print(f"{'Checkpoint':<30} {'Elo':<10} {'Win Rate':<10}")
    print_separator('-')

    for r in sorted(results, key=lambda x: x['elo'], reverse=True):
        print(f"{r['path']:<30} {r['elo']:<10} {r['win_rate']:<9.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="View training/evaluation results (CLI-friendly)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View training progress
  python scripts/view_results.py --training checkpoints/training_stats.json

  # View evaluation results
  python scripts/view_results.py --eval results.json

  # View last 5 iterations
  python scripts/view_results.py --training training_stats.json --last 5

  # Compare multiple evaluations
  python scripts/view_results.py --compare eval1.json eval2.json eval3.json
        """
    )

    parser.add_argument('--training', type=str,
                       help='View training statistics JSON')
    parser.add_argument('--eval', type=str,
                       help='View evaluation results JSON')
    parser.add_argument('--last', type=int, default=10,
                       help='Show last N iterations (default: 10)')
    parser.add_argument('--compare', nargs='+',
                       help='Compare multiple evaluation JSONs')

    args = parser.parse_args()

    if args.training:
        view_training_stats(args.training)
        if len(json.load(open(args.training))) > 0:
            print()
            view_progress(args.training, args.last)

    elif args.eval:
        view_eval_results(args.eval)

    elif args.compare:
        compare_checkpoints(args.compare)

    else:
        parser.print_help()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
