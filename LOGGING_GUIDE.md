# Detailed Logging & Monitoring Guide

## 🎮 See What's Happening in Games

### Problem
Default training shows progress bars but not actual game moves.

### Solution: Verbose Training Mode

---

## 📺 Verbose Training (See Games Being Played)

### Basic Usage

```bash
python scripts/train_verbose.py \
    --iterations 5 \
    --games-per-iter 10 \
    --show-every 2
```

### Output Example

```
=== Chess SNN Training (Verbose Mode) ===
Training for 5 iterations
Games per iteration: 10
Showing detailed view every 2 games
Device: cpu

=== Iteration 1 ===
Generating 10 games...

[Game 1] Starting new game
r n b q k b n r
p p p p p p p p
. . . . . . . .
. . . . . . . .
. . . . . . . .
. . . . . . . .
P P P P P P P P
R N B Q K B N R

  Move 1: e2e4 (value: 0.012)
  Move 2: e7e5 (value: -0.008)
  Move 3: g1f3 (value: 0.015)
  Move 4: b8c6 (value: -0.011)
  Move 5: f1c4 (value: 0.023)
  Move 6: g8f6 (value: -0.019)
  Move 7: d2d3 (value: 0.008)
  Move 8: f8e7 (value: -0.005)
  Move 9: e1g1 (value: 0.031)
  Move 10: d7d6 (value: -0.012)

  Position after 10 moves:
r . b q k . . r
p p p . b p p p
. . n p . n . .
. . . . p . . .
. . B . P . . .
. . . P . N . .
P P P . . P P P
R N B Q . R K .

  Move 11: b1c3 (value: 0.021)
  ...

  Game Over!
  Result: checkmate (outcome: 1.0)
  Total moves: 47
  Final position:
r . b q . . k r
p p p . . p p p
. . n . . n . .
. . . . . . . .
. . . . . Q . .
. . . . . N . .
P P P . . P P P
. . . . . R K .

Game stats: {'avg_moves': 82.4, 'white_win_rate': 0.3, 'draw_rate': 0.5}
```

### Parameters

```bash
python scripts/train_verbose.py \
    --iterations 10          # Number of training iterations
    --games-per-iter 20      # Games per iteration
    --show-every 3           # Show every 3rd game in detail
    --device cuda            # Use GPU
    --checkpoint model.pt    # Resume from checkpoint
```

---

## 📊 Real-Time Monitoring

### Watch Training Progress Live

```bash
python scripts/watch_training.py --file checkpoints/training_stats.json
```

### Output (Live Updating)

```
======================================================================
  CHESS SNN TRAINING MONITOR
======================================================================
File: checkpoints/training_stats.json
Monitoring time: 0:15:23
Last update: 14:32:45

Current Iteration: 23
Temperature: 0.7854

GAME STATISTICS:
----------------------------------------------------------------------
  Games:         200
  Avg moves:     98.3
  White wins:    42.5%
  Draws:         45.2%
  Black wins:    12.3%

TRAINING LOSSES:
----------------------------------------------------------------------
  Total loss:    -0.123456
  Policy loss:   -0.098234
  Value loss:    0.234567
  Entropy:       2.345678

REPLAY BUFFER:
----------------------------------------------------------------------
  Size:          45600
  Win rate:      54.2%

PROGRESS:
----------------------------------------------------------------------
  Moves change:  -8.2
  Loss change:   -0.012345

======================================================================
Press 'q' to quit | Refreshing every 5s
```

### Options

```bash
# Custom file and refresh rate
python scripts/watch_training.py \
    --file checkpoints_8gpu/training_stats.json \
    --interval 10

# Simple mode (no curses, for remote SSH)
python scripts/watch_training.py \
    --file training_stats.json \
    --simple
```

### Simple Mode Output

```
======================================================================
  CHESS SNN TRAINING MONITOR (Simple Mode)
======================================================================
Monitoring: checkpoints/training_stats.json
Refresh interval: 5s
Press Ctrl+C to stop
======================================================================

[14:32:45] Iteration 23
----------------------------------------------------------------------
Games: 200 | Avg moves: 98.3 | Draws: 45.2%
Loss: -0.1235 | Policy: -0.0982 | Value: 0.2346
Buffer: 45600 | Win rate: 54.2%

[14:32:50] Iteration 23
----------------------------------------------------------------------
Games: 200 | Avg moves: 98.3 | Draws: 45.2%
Loss: -0.1235 | Policy: -0.0982 | Value: 0.2346
Buffer: 45600 | Win rate: 54.2%
```

---

## 📈 Improved Progress Bars

### New Features in Regular Training

Training now shows **real-time statistics** in progress bar:

```bash
python scripts/train.py --iterations 100 --games-per-iter 200
```

**Old output:**
```
Generating games: 100%|██████| 200/200 [06:40<00:00,  2.00s/it]
```

**New output:**
```
Generating games: 100%|██████| 200/200 [06:40<00:00] avg_moves=95.2 draw_rate=42.5%
```

Now you can see:
- Average moves per game (decreases as agent improves)
- Draw rate (should decrease over training)

---

## 🔍 Detailed Game Logging During Multi-GPU Training

### Add Verbose Mode to Multi-GPU

Edit your training script to add verbose logging:

```python
# In scripts/train_multi_gpu.py or your custom script
# After creating trainer

# Sample every 20th game with detailed output
trainer.self_play_engine.verbose = True
trainer.self_play_engine.sample_every = 20
```

Or create a wrapper script:

```bash
# train_8gpu_verbose.sh
#!/bin/bash
python scripts/train_verbose.py \
    --iterations 200 \
    --games-per-iter 100 \
    --show-every 10 \
    --device cuda \
    --save-dir checkpoints_verbose
```

---

## 📊 Monitor From Remote/Cluster

### Option 1: SSH + watch_training.py

```bash
# From local machine
ssh cluster "cd chess-snn-rl && python scripts/watch_training.py --file checkpoints/training_stats.json --simple"
```

### Option 2: Periodic Updates

```bash
# On cluster, create monitor script
cat > monitor.sh << 'EOF'
#!/bin/bash
while true; do
    clear
    echo "=== Training Status $(date) ==="
    python scripts/view_results.py --training checkpoints/training_stats.json --last 3
    sleep 30
done
EOF

chmod +x monitor.sh
./monitor.sh
```

### Option 3: Log to File

```bash
# On cluster
python scripts/watch_training.py --simple > training_monitor.log 2>&1 &

# View from local machine
ssh cluster "tail -f chess-snn-rl/training_monitor.log"
```

---

## 🎯 Recommended Workflows

### For Development (See Everything)

```bash
# Small run with full visibility
python scripts/train_verbose.py \
    --iterations 5 \
    --games-per-iter 10 \
    --show-every 2 \
    --device cuda

# In another terminal
python scripts/watch_training.py --file checkpoints_verbose/training_stats.json
```

### For Production (Monitor Progress)

```bash
# Start training
python scripts/train_multi_gpu.py \
    --iterations 200 \
    --games-per-iter 1600 \
    --gpus 0,1,2,3,4,5,6,7

# In another terminal/tmux pane
python scripts/watch_training.py --file checkpoints_8gpu/training_stats.json
```

### For Cluster (Headless)

```bash
# Submit job
sbatch train_8gpu.sh

# Monitor remotely
ssh cluster "cd chess-snn-rl && python scripts/watch_training.py --file checkpoints_8gpu/training_stats.json --simple"

# Or check periodically
ssh cluster "cd chess-snn-rl && python scripts/view_results.py --training checkpoints_8gpu/training_stats.json --last 5"
```

---

## 📝 Understanding the Output

### Game Moves

```
Move 15: e4e5 (value: 0.034)
```

- **e4e5**: UCI move notation (from e4 to e5)
- **value: 0.034**: Agent's position evaluation
  - Positive = good for current player
  - Negative = bad for current player
  - Range: [-1, 1]

### Game Outcomes

```
Result: checkmate (outcome: 1.0)
```

- **checkmate**: How game ended
- **outcome: 1.0**: Result from current player's perspective
  - 1.0 = Win
  - 0.0 = Draw
  - -1.0 = Loss

### Training Metrics

```
avg_moves=95.2
```
- **Good**: Decreasing over time (games getting more decisive)
- **Bad**: Stuck at ~200 (hitting move limit = weak play)

```
draw_rate=42.5%
```
- **Early training**: 80-90% draws (exploring)
- **Mid training**: 40-60% draws (learning tactics)
- **Late training**: 30-40% draws (decisive play)

---

## 🔧 Troubleshooting

### "No curses module"

Use simple mode:
```bash
python scripts/watch_training.py --simple
```

### "File not found"

Training hasn't started yet. The file is created after first iteration.

### "Nothing happening"

Check if training is actually running:
```bash
ps aux | grep train
nvidia-smi  # Check GPU usage
```

---

## 💡 Tips

1. **Use verbose mode for debugging**
   - See if agent is making reasonable moves
   - Check if illegal moves are being selected
   - Understand early game patterns

2. **Use watch_training for long runs**
   - Monitor without stopping training
   - See progress in real-time
   - Catch issues early

3. **Use simple mode on cluster**
   - Works over SSH
   - No special terminal requirements
   - Easy to log to file

4. **Monitor draw rate decline**
   - Key indicator of learning
   - Should drop from 90% → 40% over training

5. **Check average moves**
   - Should decrease over training
   - 200 moves = hitting limit (bad)
   - 60-100 moves = decisive games (good)

---

**Now you can see exactly what's happening during training! 🎮**
