# Quick Start Guide

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Verify installation:**
```bash
python tests/test_encoder.py
python tests/test_model.py
python tests/test_evaluation.py
```

## Training

### Quick Training (10 iterations for testing)
```bash
python scripts/train.py --iterations 10 --games-per-iter 50
```

### Full Training (100 iterations)
```bash
python scripts/train.py --iterations 100 --games-per-iter 200 --device cuda
```

### Resume Training
```bash
python scripts/train.py --checkpoint checkpoints/model_iter_50.pt --iterations 50
```

## Evaluation

### Quick Evaluation (20 games)
```bash
python scripts/evaluate.py --checkpoint checkpoints/model_final.pt --quick
```

### Full Evaluation (100 games per opponent)
```bash
python scripts/evaluate.py --checkpoint checkpoints/model_final.pt --games 100
```

## Interactive Play

### Play as Black against the agent
```bash
python scripts/play_human.py --checkpoint checkpoints/model_final.pt
```

### Play as White
```bash
python scripts/play_human.py --checkpoint checkpoints/model_final.pt --play-white
```

## Project Structure

```
chess-snn-rl/
├── config/              # Configuration files
│   ├── model_config.py      # SNN architecture
│   ├── training_config.py   # Training hyperparameters
│   └── eval_config.py       # Evaluation settings
│
├── src/
│   ├── chess_env/       # Chess environment
│   │   ├── board.py         # Board wrapper
│   │   ├── encoder.py       # Spike train encoding
│   │   └── move_encoding.py # Action space mapping
│   │
│   ├── models/          # SNN models
│   │   ├── snn_base.py      # Base SNN blocks
│   │   └── chess_snn.py     # Complete model
│   │
│   ├── training/        # Training infrastructure
│   │   ├── replay_buffer.py # Experience replay
│   │   ├── self_play.py     # Self-play engine
│   │   ├── loss.py          # Loss functions
│   │   └── trainer.py       # Training orchestrator
│   │
│   └── evaluation/      # Evaluation system
│       ├── elo_calculator.py # Elo rating
│       ├── opponents.py      # Baseline opponents
│       └── evaluator.py      # Match runner
│
├── scripts/             # Executable scripts
│   ├── train.py             # Training script
│   ├── evaluate.py          # Evaluation script
│   └── play_human.py        # Interactive play
│
└── tests/               # Unit tests
    ├── test_encoder.py
    ├── test_model.py
    └── test_evaluation.py
```

## Training Configuration

Key hyperparameters in `config/training_config.py`:

- `GAMES_PER_ITERATION`: 200 games per iteration
- `BATCH_SIZE`: 256 experiences per batch
- `LEARNING_RATE`: 1e-4
- `EPOCHS_PER_ITERATION`: 20 training epochs
- `TEMPERATURE`: Starts at 1.0, decays to 0.1

## Model Architecture

From `config/model_config.py`:

- **Input**: [T=16, C=13, H=8, W=8] spike trains
- **Convolutions**: 64 → 128 → 256 channels
- **LIF neurons**: tau=2.0, ATan surrogate
- **Policy head**: 4672 actions (legal moves masked)
- **Value head**: Position evaluation [-1, 1]

## Expected Performance

### Short-term (10-20 iterations)
- Legal move rate: >95%
- Win rate vs random: >50%
- Elo: ~500-600

### Medium-term (50-100 iterations)
- Legal move rate: ~100%
- Win rate vs random: >70%
- Win rate vs greedy: >40%
- Elo: ~800-1000

### Long-term (200+ iterations)
- Win rate vs greedy: >60%
- Win rate vs minimax-d2: >30%
- Elo: >1200

## Monitoring Training

Training statistics are saved to:
- `checkpoints/training_stats.json` - Full training history
- Checkpoints saved every 10 iterations

## Troubleshooting

### CUDA out of memory
Reduce batch size or games per iteration:
```bash
python scripts/train.py --games-per-iter 100
```

### Slow training
Use fewer games per iteration or fewer epochs:
Edit `config/training_config.py` to adjust `EPOCHS_PER_ITERATION`

### Model outputs illegal moves
This should be rare due to legal move masking. Check:
1. Move encoder is working correctly
2. Legal masks are applied during action selection

## Next Steps

1. Run initial training for 10 iterations
2. Quick evaluate to verify model is learning
3. Scale up to 100+ iterations
4. Full evaluation against all opponents
5. Analyze results and iterate on architecture/hyperparameters
