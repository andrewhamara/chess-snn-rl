#!/usr/bin/env python3
"""Real-time training monitor with live statistics."""

import sys
import time
import json
from pathlib import Path
from datetime import datetime, timedelta

try:
    import curses
except ImportError:
    curses = None


def format_duration(seconds):
    """Format duration in seconds to human readable."""
    return str(timedelta(seconds=int(seconds)))


def monitor_with_curses(stats_file, refresh_interval=5):
    """Monitor with curses (if available)."""
    def draw(stdscr):
        curses.curs_set(0)  # Hide cursor
        stdscr.nodelay(1)   # Non-blocking input

        last_data = None
        start_time = time.time()

        while True:
            stdscr.clear()

            # Read latest data
            try:
                with open(stats_file) as f:
                    data = json.load(f)
                    if len(data) > 0:
                        last_data = data[-1]
            except (FileNotFoundError, json.JSONDecodeError):
                stdscr.addstr(0, 0, "Waiting for training to start...")
                stdscr.refresh()
                time.sleep(refresh_interval)
                continue

            if not last_data:
                stdscr.addstr(0, 0, "No data yet...")
                stdscr.refresh()
                time.sleep(refresh_interval)
                continue

            # Header
            height, width = stdscr.getmaxyx()
            elapsed = time.time() - start_time

            stdscr.addstr(0, 0, "="*min(70, width-1))
            stdscr.addstr(1, 0, "  CHESS SNN TRAINING MONITOR")
            stdscr.addstr(2, 0, "="*min(70, width-1))
            stdscr.addstr(3, 0, f"File: {stats_file}")
            stdscr.addstr(4, 0, f"Monitoring time: {format_duration(elapsed)}")
            stdscr.addstr(5, 0, f"Last update: {datetime.now().strftime('%H:%M:%S')}")

            # Current iteration
            row = 7
            iter_num = last_data['iteration']
            temp = last_data['temperature']

            stdscr.addstr(row, 0, f"Current Iteration: {iter_num}")
            row += 1
            stdscr.addstr(row, 0, f"Temperature: {temp:.4f}")
            row += 2

            # Game stats
            game_stats = last_data['game_stats']
            stdscr.addstr(row, 0, "GAME STATISTICS:")
            row += 1
            stdscr.addstr(row, 0, "-"*min(70, width-1))
            row += 1
            stdscr.addstr(row, 0, f"  Games:         {game_stats['total_games']}")
            row += 1
            stdscr.addstr(row, 0, f"  Avg moves:     {game_stats['avg_moves']:.1f}")
            row += 1
            stdscr.addstr(row, 0, f"  White wins:    {game_stats['white_win_rate']*100:.1f}%")
            row += 1
            stdscr.addstr(row, 0, f"  Draws:         {game_stats['draw_rate']*100:.1f}%")
            row += 1
            stdscr.addstr(row, 0, f"  Black wins:    {game_stats['black_win_rate']*100:.1f}%")
            row += 2

            # Training losses
            losses = last_data['training_losses']
            stdscr.addstr(row, 0, "TRAINING LOSSES:")
            row += 1
            stdscr.addstr(row, 0, "-"*min(70, width-1))
            row += 1
            stdscr.addstr(row, 0, f"  Total loss:    {losses['total_loss']:.6f}")
            row += 1
            stdscr.addstr(row, 0, f"  Policy loss:   {losses['policy_loss']:.6f}")
            row += 1
            stdscr.addstr(row, 0, f"  Value loss:    {losses['value_loss']:.6f}")
            row += 1
            stdscr.addstr(row, 0, f"  Entropy:       {losses['entropy']:.6f}")
            row += 2

            # Buffer stats
            buffer = last_data['buffer_stats']
            stdscr.addstr(row, 0, "REPLAY BUFFER:")
            row += 1
            stdscr.addstr(row, 0, "-"*min(70, width-1))
            row += 1
            stdscr.addstr(row, 0, f"  Size:          {buffer['size']}")
            row += 1
            stdscr.addstr(row, 0, f"  Win rate:      {buffer['win_rate']*100:.1f}%")
            row += 2

            # Progress
            if len(data) > 1:
                prev = data[-2]
                stdscr.addstr(row, 0, "PROGRESS:")
                row += 1
                stdscr.addstr(row, 0, "-"*min(70, width-1))
                row += 1

                moves_delta = game_stats['avg_moves'] - prev['game_stats']['avg_moves']
                loss_delta = losses['total_loss'] - prev['training_losses']['total_loss']

                stdscr.addstr(row, 0, f"  Moves change:  {moves_delta:+.1f}")
                row += 1
                stdscr.addstr(row, 0, f"  Loss change:   {loss_delta:+.6f}")
                row += 2

            # Footer
            if row < height - 2:
                stdscr.addstr(row, 0, "="*min(70, width-1))
                row += 1
                stdscr.addstr(row, 0, "Press 'q' to quit | Refreshing every 5s")

            stdscr.refresh()

            # Check for quit
            key = stdscr.getch()
            if key == ord('q'):
                break

            time.sleep(refresh_interval)

    curses.wrapper(draw)


def monitor_simple(stats_file, refresh_interval=5):
    """Simple monitoring without curses."""
    print("="*70)
    print("  CHESS SNN TRAINING MONITOR (Simple Mode)")
    print("="*70)
    print(f"Monitoring: {stats_file}")
    print(f"Refresh interval: {refresh_interval}s")
    print("Press Ctrl+C to stop")
    print("="*70)

    try:
        while True:
            try:
                with open(stats_file) as f:
                    data = json.load(f)
                    if len(data) == 0:
                        print("Waiting for training data...")
                        time.sleep(refresh_interval)
                        continue

                    last = data[-1]

                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Iteration {last['iteration']}")
                    print("-"*70)

                    game_stats = last['game_stats']
                    print(f"Games: {game_stats['total_games']} | "
                          f"Avg moves: {game_stats['avg_moves']:.1f} | "
                          f"Draws: {game_stats['draw_rate']*100:.1f}%")

                    losses = last['training_losses']
                    print(f"Loss: {losses['total_loss']:.4f} | "
                          f"Policy: {losses['policy_loss']:.4f} | "
                          f"Value: {losses['value_loss']:.4f}")

                    buffer = last['buffer_stats']
                    print(f"Buffer: {buffer['size']} | Win rate: {buffer['win_rate']*100:.1f}%")

            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Waiting for file... ({e})")

            time.sleep(refresh_interval)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Monitor training progress in real-time"
    )
    parser.add_argument('--file', type=str,
                       default='checkpoints/training_stats.json',
                       help='Training stats JSON file')
    parser.add_argument('--interval', type=int, default=5,
                       help='Refresh interval in seconds')
    parser.add_argument('--simple', action='store_true',
                       help='Use simple mode (no curses)')

    args = parser.parse_args()

    stats_file = Path(args.file)

    if not stats_file.exists():
        print(f"Warning: {stats_file} does not exist yet")
        print("Will wait for it to be created...")

    if curses and not args.simple:
        try:
            monitor_with_curses(stats_file, args.interval)
        except Exception as e:
            print(f"Curses failed: {e}")
            print("Falling back to simple mode...")
            monitor_simple(stats_file, args.interval)
    else:
        monitor_simple(stats_file, args.interval)


if __name__ == '__main__':
    main()
