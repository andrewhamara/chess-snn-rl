# Cluster Quick Commands Reference

One-page reference for all cluster operations.

---

## 🚀 Initial Setup

```bash
# Transfer & extract
scp chess-snn-rl.tar.gz user@cluster:/scratch/username/
ssh user@cluster
cd /scratch/username && tar -xzf chess-snn-rl.tar.gz && cd chess-snn-rl

# Install
python -m venv venv && source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
python verify_install.py
```

---

## 📊 Training

```bash
# Quick test (10 iterations)
python scripts/train_multi_gpu.py --iterations 10 --games-per-iter 160 --gpus 0,1,2,3,4,5,6,7

# Full training (200 iterations, ~16 hours)
python scripts/train_multi_gpu.py --iterations 200 --games-per-iter 1600 --gpus 0,1,2,3,4,5,6,7 --save-dir checkpoints_8gpu

# With SLURM
sbatch train_8gpu.sh

# Monitor job
squeue -u $USER
tail -f logs/train_*.out
```

---

## 🎯 Benchmarking (No GUI)

```bash
# Quick benchmark (5 min, 20 games)
python scripts/benchmark.py --checkpoint checkpoints_8gpu/model_final.pt --quick

# Full benchmark (30 min, 100 games)
python scripts/benchmark.py --checkpoint checkpoints_8gpu/model_final.pt --games 100

# Save results
python scripts/benchmark.py \
    --checkpoint checkpoints_8gpu/model_final.pt \
    --games 100 \
    --output results.txt \
    --json results.json

# Quiet mode (just Elo)
python scripts/benchmark.py --checkpoint model.pt --quick --quiet
# Output: Avg Elo: 1350, Win rate: 85.0%
```

---

## 📈 View Results

```bash
# View training progress
python scripts/view_results.py --training checkpoints_8gpu/training_stats.json

# View last 20 iterations
python scripts/view_results.py --training checkpoints_8gpu/training_stats.json --last 20

# View evaluation results
python scripts/view_results.py --eval results.json

# Compare checkpoints
python scripts/view_results.py --compare results_iter_*.json
```

---

## 📁 Check Progress

```bash
# Latest iteration
tail -1 checkpoints_8gpu/training_stats.json | jq '.iteration'

# Latest Elo (if evaluated)
cat results.json | jq '.avg_performance_rating'

# Training losses
tail -1 checkpoints_8gpu/training_stats.json | jq '.training_losses'

# Game stats
tail -1 checkpoints_8gpu/training_stats.json | jq '.game_stats'

# GPU usage
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
```

---

## 🔄 Remote Monitoring

```bash
# From local machine
ssh user@cluster "cd chess-snn-rl && tail -1 checkpoints_8gpu/training_stats.json | jq '.iteration'"

# Quick status
ssh user@cluster "cd chess-snn-rl && python scripts/view_results.py --training checkpoints_8gpu/training_stats.json --last 5"

# Run benchmark remotely
ssh user@cluster "cd chess-snn-rl && source venv/bin/activate && python scripts/benchmark.py --checkpoint checkpoints_8gpu/model_final.pt --quick"
```

---

## 📥 Download Results

```bash
# From local machine
scp user@cluster:/scratch/username/chess-snn-rl/results.txt .
scp user@cluster:/scratch/username/chess-snn-rl/results.json .
scp user@cluster:/scratch/username/chess-snn-rl/checkpoints_8gpu/model_final.pt .

# Download all checkpoints
rsync -avz user@cluster:/scratch/username/chess-snn-rl/checkpoints_8gpu/ ./checkpoints/
```

---

## 🛠️ Troubleshooting

```bash
# Check SLURM job
squeue -u $USER
scontrol show job <jobid>
scancel <jobid>

# Check logs
tail -f logs/train_*.out
tail -f logs/train_*.err

# Test on compute node
srun --gres=gpu:1 --pty bash
source venv/bin/activate
python -c "import torch; print(torch.cuda.is_available())"

# Check GPU memory
nvidia-smi
```

---

## 📊 Extract Metrics

```bash
# Elo rating
cat results.json | jq '.avg_performance_rating'

# Win rates by opponent
cat results.json | jq '.opponent_results | to_entries[] | {name: .value.opponent, win_rate: (.value.win_rate * 100)}'

# Latest iteration stats
tail -1 checkpoints_8gpu/training_stats.json | jq '{iter: .iteration, draws: (.game_stats.draw_rate * 100), loss: .training_losses.total_loss}'

# Training curve (last 10 iterations)
tail -10 checkpoints_8gpu/training_stats.json | jq '{iter: .iteration, loss: .training_losses.total_loss, moves: .game_stats.avg_moves}'
```

---

## 🎯 Complete Workflow (Copy-Paste)

```bash
# 1. Start training
sbatch train_8gpu.sh

# 2. Monitor (run periodically)
squeue -u $USER
tail -5 checkpoints_8gpu/training_stats.json | jq '.iteration'

# 3. When done, benchmark
python scripts/benchmark.py \
    --checkpoint checkpoints_8gpu/model_final.pt \
    --games 100 \
    --output final_results.txt \
    --json final_results.json

# 4. View results
python scripts/view_results.py --eval final_results.json
cat final_results.txt

# 5. Download
# (from local machine)
scp user@cluster:/scratch/username/chess-snn-rl/final_results.* .
scp user@cluster:/scratch/username/chess-snn-rl/checkpoints_8gpu/model_final.pt .
```

---

## 💡 One-Liners

```bash
# Current iteration
tail -1 checkpoints_8gpu/training_stats.json | jq -r '.iteration'

# Latest Elo
cat results.json | jq -r '.avg_performance_rating'

# GPU memory usage
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits

# Training progress (iterations/hour)
python -c "import json; data=json.load(open('checkpoints_8gpu/training_stats.json')); print(f'{len(data)} iterations')"

# Quick benchmark + extract Elo
python scripts/benchmark.py --checkpoint model.pt --quick --quiet | grep "Avg Elo"
```

---

## 📧 Email Notifications (Optional)

```bash
# Add to end of train_multi_gpu.py
echo "Training complete. Avg Elo: $(cat results.json | jq '.avg_performance_rating')" | mail -s "Chess SNN Complete" your@email.com

# Or in SLURM script
#SBATCH --mail-type=END
#SBATCH --mail-user=your@email.com
```

---

**All operations are CLI-based. No GUI needed! 🎯**
