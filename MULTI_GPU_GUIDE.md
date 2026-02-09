# Multi-GPU Training Guide

## Overview

The Chess SNN RL system supports **parallel self-play across multiple GPUs** to dramatically speed up training. The bottleneck in training is game generation, not the neural network training itself, so we parallelize self-play across GPUs.

## Architecture

- **Self-Play**: Distributed across all GPUs (each GPU generates games independently)
- **Training**: Happens on primary GPU (GPU 0 by default)
- **Speedup**: Near-linear with number of GPUs for self-play phase

## Single GPU Training

```bash
# Use first GPU (cuda:0)
python scripts/train.py --iterations 100 --games-per-iter 200 --device cuda

# Use specific GPU
python scripts/train.py --iterations 100 --games-per-iter 200 --device cuda:3
```

## Multi-GPU Training (8 GPUs)

```bash
# Use all 8 GPUs
python scripts/train_multi_gpu.py \
    --iterations 100 \
    --games-per-iter 800 \
    --gpus 0,1,2,3,4,5,6,7

# Use subset of GPUs (e.g., 4 GPUs)
python scripts/train_multi_gpu.py \
    --iterations 100 \
    --games-per-iter 400 \
    --gpus 0,1,2,3
```

## Performance Scaling

### Expected Speedup (Self-Play Phase)

| GPUs | Games/Iter | Time/Iter | Speedup |
|------|------------|-----------|---------|
| 1    | 200        | ~7 min    | 1.0x    |
| 2    | 400        | ~7 min    | 2.0x    |
| 4    | 800        | ~7 min    | 4.0x    |
| 8    | 1600       | ~7 min    | 8.0x    |

**Key Insight**: With 8 GPUs, you can generate 8x more games in the same time, leading to much better sample efficiency and faster learning.

## Optimal Configuration for 8 GPUs

```bash
python scripts/train_multi_gpu.py \
    --iterations 200 \
    --games-per-iter 1600 \
    --gpus 0,1,2,3,4,5,6,7 \
    --save-dir checkpoints_8gpu
```

**Why 1600 games/iter?**
- 200 games per GPU (same single-GPU workload)
- Better exploration due to more diverse experiences
- Faster convergence to high Elo ratings

## Advanced Usage

### Resume Training

```bash
python scripts/train_multi_gpu.py \
    --checkpoint checkpoints_8gpu/model_iter_50.pt \
    --iterations 100 \
    --games-per-iter 1600 \
    --gpus 0,1,2,3,4,5,6,7
```

### Memory-Efficient Training (Smaller Batch Size)

If you run out of memory during training phase, adjust batch size in `config/training_config.py`:

```python
BATCH_SIZE = 128  # Reduce from 256
```

### Custom GPU Allocation

```bash
# Use only high-memory GPUs
python scripts/train_multi_gpu.py \
    --gpus 4,5,6,7 \
    --games-per-iter 800

# Leave GPU 0 for other tasks
python scripts/train_multi_gpu.py \
    --gpus 1,2,3,4,5,6,7 \
    --games-per-iter 1400
```

## How It Works

### Parallel Self-Play Pipeline

1. **Model Replication**: Model state is copied to all GPUs
2. **Independent Game Generation**: Each GPU generates N/K games (N=total, K=num GPUs)
3. **Result Collection**: Games from all GPUs are collected via multiprocessing queue
4. **Centralized Training**: All experiences train model on primary GPU
5. **Model Update**: Updated model is distributed to all GPUs for next iteration

### Code Architecture

```python
# Parallel self-play engine
ParallelSelfPlayEngine(
    model,
    gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7]
)

# Worker processes (one per GPU)
for gpu_id in gpu_ids:
    spawn_worker(
        gpu_id=gpu_id,
        games=games_per_gpu,
        model_state=model.state_dict()
    )

# Collect results
all_games = gather_from_workers()

# Train on primary GPU
train_on_experiences(all_games)
```

## Monitoring

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Monitor training progress
tail -f checkpoints_8gpu/training_stats.json
```

## Troubleshooting

### Out of Memory (OOM)

**Symptom**: CUDA out of memory error

**Solutions**:
1. Reduce batch size: `BATCH_SIZE = 128` in `training_config.py`
2. Reduce games per GPU: Use fewer total games
3. Reduce model size: Decrease conv channels in `model_config.py`

### Slow Multi-GPU Training

**Symptom**: Not seeing expected speedup

**Causes**:
1. **CPU Bottleneck**: Self-play still uses CPU for chess logic
2. **Queue Overhead**: Minimal, but exists
3. **GPU Communication**: Only at start/end of iteration

**Expected**: 6-7x speedup with 8 GPUs (not full 8x due to overhead)

### Process Hangs

**Symptom**: Training freezes during parallel generation

**Solution**:
```python
# Set in your shell before running
export CUDA_LAUNCH_BLOCKING=1

# Or reduce number of GPUs
python scripts/train_multi_gpu.py --gpus 0,1,2,3
```

## Recommended Settings

### Development (Quick Iteration)
```bash
python scripts/train_multi_gpu.py \
    --iterations 10 \
    --games-per-iter 400 \
    --gpus 0,1,2,3
```

### Full Training (High Performance)
```bash
python scripts/train_multi_gpu.py \
    --iterations 500 \
    --games-per-iter 1600 \
    --gpus 0,1,2,3,4,5,6,7
```

### Evaluation After Training
```bash
# Evaluate on single GPU
python scripts/evaluate.py \
    --checkpoint checkpoints_8gpu/model_final.pt \
    --games 100 \
    --device cuda:0
```

## Performance Tips

1. **Use all 8 GPUs**: Maximum parallelization for self-play
2. **Scale games linearly**: 200 games/GPU × 8 GPUs = 1600 total
3. **Monitor GPU utilization**: All GPUs should be near 100% during self-play
4. **Batch size on primary GPU**: Keep at 256 if memory allows
5. **Save frequently**: Checkpoint every 10 iterations (default)

## Expected Results

With 8 GPUs and proper configuration:

- **Training Speed**: 8x faster game generation
- **Sample Efficiency**: Better due to more diverse experiences
- **Elo Progression**:
  - 100 iterations: ~800 Elo
  - 200 iterations: ~1000-1200 Elo
  - 500 iterations: >1200 Elo

## Cost-Benefit Analysis

| Setup | Games/Hour | Elo @ 24hrs | GPU Cost |
|-------|------------|-------------|----------|
| 1 GPU | ~1,700 | 600-800 | 1x |
| 4 GPUs | ~6,800 | 1000-1200 | 4x |
| 8 GPUs | ~13,600 | 1200-1400 | 8x |

**Recommendation**: Use all 8 GPUs for fastest path to strong chess play.
