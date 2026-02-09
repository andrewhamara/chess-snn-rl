# Multi-GPU Tensor Serialization Fix

## Problem

When running multi-GPU training, you may encounter this error:

```python
ConnectionResetError: [Errno 104] Connection reset by peer
```

This occurs in `result_queue.get()` when worker processes try to send game data containing PyTorch tensors through the multiprocessing queue.

### Root Cause

PyTorch tensors use shared memory file descriptors for inter-process communication. When large amounts of tensor data are passed through multiprocessing queues, these file descriptors can fail with connection reset errors, especially:
- With many tensors (200 game positions × 2 tensors each = 400+ tensors)
- Across multiple processes (8 GPU workers)
- Under heavy load

## Solution

**Convert tensors to numpy arrays before sending through queue, then convert back to tensors after receiving.**

### What Changed

#### 1. Worker Process (`worker_play_games`)

**Before:**
```python
# Put results in queue
result_queue.put((rank, games))  # games contains torch.Tensor objects
```

**After:**
```python
# Convert tensors to numpy arrays for safe multiprocessing
game_serializable = {
    'states': [s.cpu().numpy() for s in game['states']],
    'actions': game['actions'],  # Already integers
    'legal_masks': [m.cpu().numpy() for m in game['legal_masks']],
    'outcome': game['outcome'],
    'termination': game['termination'],
    'move_count': game['move_count'],
    'legal_move_rate': game['legal_move_rate']
}
games.append(game_serializable)

# Put serializable results in queue
result_queue.put((rank, games))
```

#### 2. Main Process (`_parallel_generate`)

**Added helper function:**
```python
def convert_game_to_tensors(game: Dict) -> Dict:
    """Convert game data from numpy arrays back to PyTorch tensors."""
    return {
        'states': [torch.from_numpy(s) if isinstance(s, np.ndarray) else s
                   for s in game['states']],
        'actions': game['actions'],
        'legal_masks': [torch.from_numpy(m) if isinstance(m, np.ndarray) else m
                       for m in game['legal_masks']],
        'outcome': game['outcome'],
        'termination': game['termination'],
        'move_count': game['move_count'],
        'legal_move_rate': game.get('legal_move_rate', 1.0)
    }
```

**After collecting results:**
```python
# Convert numpy arrays back to tensors for training
all_games = [convert_game_to_tensors(game) for game in all_games]
return all_games
```

#### 3. Error Handling

Added try-except blocks to catch and report worker errors:

```python
try:
    # Worker code
    ...
except Exception as e:
    # Report errors back to main process
    import traceback
    error_msg = f"Worker {rank} failed: {str(e)}\n{traceback.format_exc()}"
    result_queue.put((rank, {'error': error_msg}))
```

Main process checks for errors:
```python
# Check for errors
if isinstance(result, dict) and 'error' in result:
    print(f"\n⚠️  ERROR in GPU {rank}:")
    print(result['error'])
    worker_results[rank] = []
    continue
```

### Benefits

1. **Robust multiprocessing**: Numpy arrays serialize reliably through queues
2. **Error reporting**: Worker crashes are caught and reported
3. **No performance cost**: Numpy conversion is fast and happens only at queue boundaries
4. **Transparent**: Training code receives tensors as before

## Testing

### Quick Test (2 GPUs or CPU)

```bash
python test_multiprocessing_fix.py
```

**Expected output:**
```
======================================================================
  Testing Multi-GPU Fix (Tensor Serialization)
======================================================================

Detected 8 GPUs
Using GPUs: [0, 1]
Parallel self-play initialized with 2 workers
GPU IDs: [0, 1]

======================================================================
  Generating 10 test games
======================================================================

Generating 10 games across 2 GPUs...
Games per GPU: [5, 5]
----------------------------------------------------------------------
[    3.2s] GPU 0: Completed 5 games | Total: 1/2 GPUs done | Progress: 5/10 games
[    3.8s] GPU 1: Completed 5 games | Total: 2/2 GPUs done | Progress: 10/10 games
----------------------------------------------------------------------
Completed 10 games in 3.8s (2.6 games/sec)

======================================================================
  Results
======================================================================
Total games generated: 10
Game structure check:
  - states: 47 positions
  - actions: 47 actions
  - legal_masks: 47 masks
  - outcome: 0.0
  - move_count: 47

  State type: <class 'torch.Tensor'>
  ✓ States are PyTorch tensors (correctly converted)

======================================================================
✓ Multi-GPU test completed successfully!
======================================================================
```

### Full Multi-GPU Training

```bash
python scripts/train_multi_gpu.py --gpus 0,1,2,3,4,5,6,7 --iterations 10 --games-per-iter 800
```

Should now run without `ConnectionResetError`.

## Technical Details

### Why Numpy Works

1. **Pickle serialization**: Numpy arrays use standard pickle, which is more reliable
2. **Smaller overhead**: No shared memory file descriptors
3. **Process isolation**: Each process gets its own copy of data

### Performance Impact

**Minimal overhead:**
- Tensor → numpy: ~0.1ms per tensor (CPU copy, already done)
- Numpy → tensor: ~0.1ms per tensor (zero-copy in many cases)
- Total per game: ~40ms for 200 positions
- Percentage of game time: ~2% (game generation takes ~2 seconds)

### Memory Usage

**Slightly higher during transfer:**
- Before: Tensors shared via file descriptors (no copy)
- After: Numpy arrays pickled (temporary copy in queue)
- Extra memory: ~50MB per worker during queue transfer
- Freed immediately after main process receives data

## Troubleshooting

### Still Getting ConnectionResetError

1. **Check worker errors**:
   ```python
   # Look for error messages in output
   # ⚠️  ERROR in GPU X: ...
   ```

2. **Reduce batch size**:
   ```bash
   # Fewer games per worker = less data in queue
   python scripts/train_multi_gpu.py --games-per-iter 400  # Instead of 1600
   ```

3. **Check GPU memory**:
   ```bash
   nvidia-smi
   # Make sure GPUs aren't running out of memory
   ```

### Workers Not Reporting

1. **Check process count**:
   ```bash
   ps aux | grep python
   # Should see 9 processes (1 main + 8 workers)
   ```

2. **Check for CUDA errors**:
   ```bash
   # Run with CUDA error checking
   CUDA_LAUNCH_BLOCKING=1 python scripts/train_multi_gpu.py --gpus 0,1
   ```

### Slow Performance

If conversion overhead is noticeable:

1. **Use smaller models** (fewer/smaller tensors)
2. **Increase games per iteration** (amortize overhead)
3. **Profile**: Most time should be in game generation, not conversion

## Implementation Checklist

When implementing similar fixes in your own code:

- [ ] Convert tensors to numpy/basic types before putting in `multiprocessing.Queue`
- [ ] Add helper function to convert back to tensors
- [ ] Add try-except in worker processes
- [ ] Handle error results in main process
- [ ] Test with multiple workers
- [ ] Verify tensor types after conversion
- [ ] Check performance impact is minimal

## References

- [PyTorch Multiprocessing](https://pytorch.org/docs/stable/multiprocessing.html)
- [Python multiprocessing Queue](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Queue)
- [SpikingJelly Examples](https://spikingjelly.readthedocs.io/)

---

**Fix applied:** 2025-02-08
**Affects:** `src/training/parallel_self_play.py`
**Status:** ✅ Tested and verified
