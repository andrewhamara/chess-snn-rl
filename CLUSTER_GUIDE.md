# Cluster Deployment & Benchmarking Guide (No GUI)

Complete guide for running Chess SNN RL on a headless GPU cluster.

---

## 📦 Transfer to Cluster

### 1. Package Repository

```bash
# On local machine
cd /path/to/chess-snn-rl
tar -czf chess-snn-rl.tar.gz \
    --exclude='checkpoints' \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    .

# Check size
ls -lh chess-snn-rl.tar.gz
```

### 2. Transfer to Cluster

```bash
# Using scp
scp chess-snn-rl.tar.gz user@cluster.edu:/scratch/username/

# Using rsync (faster for updates)
rsync -avz --progress chess-snn-rl/ user@cluster.edu:/scratch/username/chess-snn-rl/
```

### 3. Setup on Cluster

```bash
# SSH to cluster
ssh user@cluster.edu

# Extract
cd /scratch/username
tar -xzf chess-snn-rl.tar.gz
cd chess-snn-rl

# Create environment
module load python/3.10  # or your cluster's module system
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Verify
python verify_install.py
```

---

## 🚀 Training on Cluster

### Using SLURM

Create a job script: `train_8gpu.sh`

```bash
#!/bin/bash
#SBATCH --job-name=chess-snn
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# Load modules
module load cuda/11.8
module load python/3.10

# Activate environment
source venv/bin/activate

# Run training
python scripts/train_multi_gpu.py \
    --iterations 200 \
    --games-per-iter 1600 \
    --gpus 0,1,2,3,4,5,6,7 \
    --save-dir checkpoints_8gpu

# Run benchmark when done
python scripts/benchmark.py \
    --checkpoint checkpoints_8gpu/model_final.pt \
    --games 100 \
    --output final_results.txt \
    --json final_results.json
```

Submit job:
```bash
mkdir -p logs
sbatch train_8gpu.sh
```

### Using PBS/Torque

Create `train_8gpu.pbs`:

```bash
#!/bin/bash
#PBS -N chess-snn
#PBS -l nodes=1:ppn=32:gpus=8
#PBS -l walltime=24:00:00
#PBS -l mem=128gb
#PBS -o logs/train_$PBS_JOBID.out
#PBS -e logs/train_$PBS_JOBID.err

cd $PBS_O_WORKDIR
source venv/bin/activate

python scripts/train_multi_gpu.py \
    --iterations 200 \
    --games-per-iter 1600 \
    --gpus 0,1,2,3,4,5,6,7 \
    --save-dir checkpoints_8gpu
```

Submit:
```bash
qsub train_8gpu.pbs
```

### Direct Interactive Session

```bash
# Request GPU node
srun --gres=gpu:8 --mem=128G --time=24:00:00 --pty bash

# Run training
source venv/bin/activate
python scripts/train_multi_gpu.py \
    --iterations 200 \
    --games-per-iter 1600 \
    --gpus 0,1,2,3,4,5,6,7
```

---

## 📊 Benchmarking (No GUI Required)

### 1. Quick Benchmark (5 minutes)

```bash
python scripts/benchmark.py \
    --checkpoint checkpoints_8gpu/model_final.pt \
    --quick \
    --output quick_results.txt
```

**Output:**
```
======================================================================
  CHESS SNN BENCHMARK
======================================================================
Checkpoint: checkpoints_8gpu/model_final.pt
Device: cuda
Mode: Quick (20 games)

======================================================================
  RUNNING EVALUATION
======================================================================

Evaluating against Random (20 games)...
Results vs Random:
  Win rate: 95.00%
  Draw rate: 5.00%
  Performance rating: 1450

Evaluating against GreedyMaterial (20 games)...
Results vs GreedyMaterial:
  Win rate: 75.00%
  Draw rate: 20.00%
  Performance rating: 1250

OPPONENT RESULTS:
----------------------------------------------------------------------
Opponent             Games    W-D-L           Win %      Elo
----------------------------------------------------------------------
Random               20       19-1-0           95.0%     1450
GreedyMaterial       20       15-4-1           75.0%     1250
----------------------------------------------------------------------

======================================================================
  EVALUATION SUMMARY
======================================================================
Total Games:     40
Total Wins:      34 (85.0%)
Total Draws:     5 (12.5%)
Total Losses:    1 (2.5%)
Avg Perf Elo:    1350

======================================================================
  ELO MILESTONES
======================================================================
✓ Beginner        600 Elo  [ACHIEVED]
✓ Intermediate    1000 Elo  [ACHIEVED]
✓ Advanced        1200 Elo  [ACHIEVED]

======================================================================
  BENCHMARK COMPLETE
======================================================================
```

### 2. Full Benchmark (30 minutes)

```bash
python scripts/benchmark.py \
    --checkpoint checkpoints_8gpu/model_final.pt \
    --games 100 \
    --output full_results.txt \
    --json full_results.json
```

### 3. Benchmark Multiple Checkpoints

```bash
# Benchmark different iterations
for i in 50 100 150 200; do
    python scripts/benchmark.py \
        --checkpoint checkpoints_8gpu/model_iter_$i.pt \
        --quick \
        --json results_iter_$i.json \
        --quiet
done

# Compare results
python scripts/view_results.py --compare results_iter_*.json
```

**Output:**
```
======================================================================
  CHECKPOINT COMPARISON
======================================================================
Checkpoint                     Elo        Win Rate
----------------------------------------------------------------------
results_iter_200.json          1350       85.0%
results_iter_150.json          1180       72.5%
results_iter_100.json          950        61.2%
results_iter_50.json           720        48.7%
```

---

## 📈 Monitor Training Progress

### 1. View Training Stats

```bash
# View latest progress
python scripts/view_results.py --training checkpoints_8gpu/training_stats.json

# View last 20 iterations
python scripts/view_results.py --training checkpoints_8gpu/training_stats.json --last 20
```

**Output:**
```
======================================================================
  TRAINING STATISTICS
======================================================================
Total iterations: 150
Latest iteration: 150
Temperature: 0.215

Latest Game Stats:
  Total games:     1600
  Avg moves:       82.3
  White win rate:  48.2%
  Draw rate:       38.5%
  Black win rate:  13.3%

Latest Training Losses:
  Total loss:      -0.2341
  Policy loss:     -0.1876
  Value loss:      0.3421
  Entropy:         1.8732

Replay Buffer:
  Size:            100000
  Win rate:        51.2%

======================================================================
  TRAINING PROGRESSION
======================================================================
Iter     Avg Moves    Draw Rate    Total Loss
----------------------------------------------------------------------
1        194.1        94.0%        -0.0152
15       178.3        87.2%        0.0234
30       156.7        76.4%        -0.0567
45       132.1        65.8%        -0.0892
60       108.5        54.2%        -0.1234
75       95.2         47.6%        -0.1567
90       88.7         42.1%        -0.1823
105      84.3         39.5%        -0.2045
120      81.6         37.8%        -0.2187
135      79.4         36.2%        -0.2289
150      82.3         38.5%        -0.2341
```

### 2. Monitor During Training

```bash
# In separate terminal/tmux pane
watch -n 30 'python scripts/view_results.py --training checkpoints_8gpu/training_stats.json --last 5'

# Or use tail on JSON
tail -f checkpoints_8gpu/training_stats.json | grep -E "iteration|total_loss|avg_moves"
```

### 3. Check GPU Utilization

```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Log to file
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 60 > gpu_usage.log
```

---

## 📁 Results Analysis

### 1. View Evaluation Results

```bash
python scripts/view_results.py --eval full_results.json
```

### 2. Extract Key Metrics

```bash
# Get Elo rating
cat full_results.json | jq '.avg_performance_rating'

# Get win rates
cat full_results.json | jq '.opponent_results | to_entries[] | {opponent: .key, win_rate: .value.win_rate}'

# Get summary
cat full_results.json | jq '{elo: .avg_performance_rating, wins: .total_wins, games: .total_games}'
```

### 3. Create Summary Report

```bash
# Generate text report
cat > generate_report.sh << 'EOF'
#!/bin/bash
echo "===== CHESS SNN TRAINING REPORT ====="
echo ""
echo "Training Stats:"
python scripts/view_results.py --training checkpoints_8gpu/training_stats.json --last 10
echo ""
echo "Evaluation Results:"
python scripts/view_results.py --eval full_results.json
EOF

chmod +x generate_report.sh
./generate_report.sh > final_report.txt

# View report
cat final_report.txt
```

---

## 🔄 Remote Monitoring

### 1. Using SSH

```bash
# From local machine, check progress
ssh user@cluster.edu "cd chess-snn-rl && cat checkpoints_8gpu/training_stats.json | tail -50"

# Run benchmark remotely
ssh user@cluster.edu "cd chess-snn-rl && source venv/bin/activate && python scripts/benchmark.py --checkpoint checkpoints_8gpu/model_final.pt --quick"
```

### 2. Using tmux/screen

```bash
# On cluster, start training in tmux
tmux new -s chess-training
source venv/bin/activate
python scripts/train_multi_gpu.py --iterations 200 --gpus 0,1,2,3,4,5,6,7

# Detach: Ctrl+B, then D

# Reattach later
tmux attach -t chess-training
```

### 3. Automated Progress Reports

Create `monitor.sh`:

```bash
#!/bin/bash
# Send progress updates via email/file

while true; do
    # Get latest stats
    python scripts/view_results.py \
        --training checkpoints_8gpu/training_stats.json \
        --last 1 > progress_update.txt

    # Timestamp
    echo "Updated: $(date)" >> progress_update.txt

    # Email (if configured)
    # mail -s "Chess SNN Progress" your@email.com < progress_update.txt

    # Or copy to accessible location
    cp progress_update.txt /path/to/web/accessible/location/

    sleep 3600  # Every hour
done
```

---

## 📤 Retrieve Results

### 1. Download Results

```bash
# From local machine
scp user@cluster.edu:/scratch/username/chess-snn-rl/full_results.json .
scp user@cluster.edu:/scratch/username/chess-snn-rl/final_report.txt .
scp user@cluster.edu:/scratch/username/chess-snn-rl/checkpoints_8gpu/model_final.pt .
```

### 2. Download Checkpoints

```bash
# Download specific checkpoint
scp user@cluster.edu:/scratch/username/chess-snn-rl/checkpoints_8gpu/model_final.pt models/

# Download all checkpoints
rsync -avz user@cluster.edu:/scratch/username/chess-snn-rl/checkpoints_8gpu/ local_checkpoints/
```

---

## 🎯 Complete Workflow Example

```bash
# 1. Submit training job
sbatch train_8gpu.sh

# 2. Monitor job
squeue -u $USER
tail -f logs/train_*.out

# 3. Check progress periodically
ssh cluster "cd chess-snn-rl && python scripts/view_results.py --training checkpoints_8gpu/training_stats.json --last 5"

# 4. After training completes, benchmark
ssh cluster "cd chess-snn-rl && source venv/bin/activate && python scripts/benchmark.py --checkpoint checkpoints_8gpu/model_final.pt --games 100 --output results.txt --json results.json"

# 5. View results
ssh cluster "cd chess-snn-rl && cat results.txt"

# 6. Download everything
scp cluster:/scratch/username/chess-snn-rl/results.* .
scp cluster:/scratch/username/chess-snn-rl/checkpoints_8gpu/model_final.pt .

# 7. View locally
cat results.txt
```

---

## 🔧 Troubleshooting on Cluster

### Check Job Status

```bash
# SLURM
squeue -u $USER
scancel <jobid>  # Cancel if needed

# PBS
qstat -u $USER
qdel <jobid>
```

### Check Logs

```bash
# View output
tail -f logs/train_*.out

# View errors
tail -f logs/train_*.err

# Check GPU usage
grep -i "cuda\|gpu\|memory" logs/train_*.err
```

### Verify Installation

```bash
# On compute node
srun --gres=gpu:1 --pty bash
source venv/bin/activate
python verify_install.py
```

---

## 📊 Expected Output

After successful training and benchmarking, you should have:

1. **Training Stats** (`training_stats.json`)
   - 200 iterations of training data
   - Loss curves, game statistics
   - Buffer statistics

2. **Benchmark Results** (`results.txt`, `results.json`)
   - Performance Elo: ~1200-1400
   - Win rates vs all opponents
   - Detailed statistics

3. **Model Checkpoints** (`checkpoints_8gpu/`)
   - `model_final.pt` - Final trained model
   - `model_iter_*.pt` - Intermediate checkpoints

4. **Reports**
   - Text summaries
   - JSON for further analysis
   - GPU usage logs

---

## 💡 Tips for Cluster Usage

1. **Use tmux/screen** - Don't lose sessions
2. **Set up email notifications** - Get updates when jobs complete
3. **Save everything** - Logs, stats, checkpoints
4. **Test first** - Run 1-2 iterations to verify setup
5. **Monitor GPU usage** - Ensure all GPUs are utilized
6. **Backup results** - Download important results regularly

---

**All benchmarking can be done via command-line! No GUI required! 🚀**
