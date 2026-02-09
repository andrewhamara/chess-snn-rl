# GPU Quick Start - 8 GPU Setup

## Transfer to Your GPU Server

```bash
# On your local machine
cd /path/to/chess-snn-rl
tar -czf chess-snn-rl.tar.gz .

# Transfer to GPU server
scp chess-snn-rl.tar.gz user@gpu-server:/path/to/destination/

# On GPU server
tar -xzf chess-snn-rl.tar.gz
cd chess-snn-rl
```

## Setup on GPU Server

```bash
# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install spikingjelly python-chess numpy scipy matplotlib tensorboard pyyaml tqdm pandas pytest

# Verify GPUs
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
nvidia-smi
```

## Single Command Training (8 GPUs)

```bash
# Full training run - 200 iterations with 1600 games/iter
python scripts/train_multi_gpu.py \
    --iterations 200 \
    --games-per-iter 1600 \
    --gpus 0,1,2,3,4,5,6,7 \
    --save-dir checkpoints_8gpu
```

## Quick Test Run First (Verify Setup)

```bash
# Test with 2 iterations to verify everything works
python scripts/train_multi_gpu.py \
    --iterations 2 \
    --games-per-iter 160 \
    --gpus 0,1,2,3,4,5,6,7 \
    --save-dir test_checkpoints

# If successful, run full training
```

## Monitor Training

```bash
# Terminal 1: Watch GPU utilization
watch -n 1 nvidia-smi

# Terminal 2: Monitor training progress
tail -f checkpoints_8gpu/training_stats.json

# Or grep for key metrics
grep -E "iteration|avg_moves|white_win_rate|total_loss" checkpoints_8gpu/training_stats.json
```

## Expected Performance (8 GPUs)

- **Games/Iteration**: 1600 (200 per GPU)
- **Time/Iteration**: ~4-6 minutes (vs ~30 min on single GPU)
- **Total Training Time**: 200 iterations × 5 min = ~16-20 hours
- **Final Elo**: 1200-1400 (after 200 iterations)

## Resume if Interrupted

```bash
python scripts/train_multi_gpu.py \
    --checkpoint checkpoints_8gpu/model_iter_100.pt \
    --iterations 100 \
    --games-per-iter 1600 \
    --gpus 0,1,2,3,4,5,6,7 \
    --save-dir checkpoints_8gpu
```

## Evaluate After Training

```bash
# Full evaluation
python scripts/evaluate.py \
    --checkpoint checkpoints_8gpu/model_final.pt \
    --games 100 \
    --device cuda

# Quick evaluation (20 games)
python scripts/evaluate.py \
    --checkpoint checkpoints_8gpu/model_final.pt \
    --quick \
    --device cuda
```

## Checkpoints

Saved automatically every 10 iterations:
- `checkpoints_8gpu/model_iter_10.pt`
- `checkpoints_8gpu/model_iter_20.pt`
- ...
- `checkpoints_8gpu/model_final.pt`

## Troubleshooting

### CUDA Out of Memory
```python
# Edit config/training_config.py
BATCH_SIZE = 128  # Reduce from 256
```

### Multiprocessing Issues
```bash
# Set before running
export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1
```

### Check GPU Memory
```bash
# See memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

## Optimal Configuration Matrix

| Scenario | GPUs | Games/Iter | Iterations | Time | Target Elo |
|----------|------|------------|------------|------|------------|
| Quick Test | 4 | 400 | 10 | 1 hour | ~400 |
| Development | 4 | 800 | 50 | 4 hours | ~800 |
| Production | 8 | 1600 | 200 | 18 hours | 1200+ |
| Maximum | 8 | 3200 | 500 | 48 hours | 1400+ |

## Single GPU vs Multi-GPU

```bash
# Single GPU (baseline)
python scripts/train.py \
    --device cuda \
    --iterations 100 \
    --games-per-iter 200

# 8 GPUs (8x games, similar time)
python scripts/train_multi_gpu.py \
    --gpus 0,1,2,3,4,5,6,7 \
    --iterations 100 \
    --games-per-iter 1600
```

**Result**: 8x more training data in the same wall-clock time!
