# Installation Guide

## Quick Install (5 minutes)

### 1. Clone & Setup

```bash
git clone https://github.com/yourusername/chess-snn-rl.git
cd chess-snn-rl

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

**For CUDA 11.8 (Recommended):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

**For CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**For CPU Only:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python verify_install.py
```

**Expected output:**
```
============================================================
Chess SNN RL - Installation Verification
============================================================
Checking Python version...
✓ Python 3.10.x (OK)

Checking required packages...
✓ PyTorch
✓ SpikingJelly
✓ python-chess
✓ NumPy
✓ SciPy
✓ Matplotlib
✓ TensorBoard
✓ PyYAML
✓ tqdm
✓ Pandas

Checking CUDA/GPU support...
✓ CUDA available with 8 GPU(s)
  GPU 0: NVIDIA RTX 4090 (24.0 GB)
  GPU 1: NVIDIA RTX 4090 (24.0 GB)
  ...

Running quick tests...
Testing board encoder...
✓ Board encoder works
Testing move encoder...
✓ Move encoder works
Testing SNN model...
✓ SNN model works

============================================================
Summary
============================================================
✓ PASS: Python Version
✓ PASS: Package Imports
✓ PASS: CUDA Support
✓ PASS: Functionality Tests

🎉 All checks passed! You're ready to train.
```

### 4. Start Training

```bash
# Quick test
python scripts/train.py --iterations 10 --games-per-iter 50 --device cuda

# Full training (8 GPUs)
python scripts/train_multi_gpu.py \
    --iterations 200 \
    --games-per-iter 1600 \
    --gpus 0,1,2,3,4,5,6,7
```

---

## Detailed Installation

### System Requirements

**Minimum:**
- Python 3.8+
- 8GB RAM
- 4GB VRAM (for GPU training)
- 10GB disk space

**Recommended:**
- Python 3.10+
- 16GB RAM
- 8GB+ VRAM per GPU
- 50GB disk space (for checkpoints)
- Multiple NVIDIA GPUs for parallel training

### GPU Setup

#### Check CUDA Version

```bash
nvidia-smi
```

Look for "CUDA Version" in the output. Install PyTorch with matching CUDA version.

#### Common CUDA Versions

| CUDA Version | PyTorch Index URL |
|--------------|-------------------|
| 11.7 | `https://download.pytorch.org/whl/cu117` |
| 11.8 | `https://download.pytorch.org/whl/cu118` |
| 12.1 | `https://download.pytorch.org/whl/cu121` |

### Package Details

All packages in `requirements.txt`:

```
# Core Deep Learning
torch>=2.0.0,<2.3.0          # PyTorch
torchvision>=0.15.0,<0.18.0  # Vision utilities
torchaudio>=2.0.0,<2.3.0     # Audio utilities

# Spiking Neural Networks
spikingjelly>=0.0.0.0.14     # SNN framework

# Chess Engine
python-chess>=1.999          # Chess library

# Scientific Computing
numpy>=1.24.0,<2.0.0         # Numerical arrays
scipy>=1.10.0,<2.0.0         # Scientific computing

# Visualization & Monitoring
matplotlib>=3.7.0,<4.0.0     # Plotting
tensorboard>=2.13.0,<3.0.0   # Training visualization
seaborn>=0.12.0              # Statistical plots

# Progress & Logging
tqdm>=4.65.0                 # Progress bars
pyyaml>=6.0                  # YAML config

# Data Processing
pandas>=2.0.0,<3.0.0         # Data frames

# Testing
pytest>=7.3.0,<8.0.0         # Unit testing
```

---

## Troubleshooting Installation

### Issue: `torch` not found

```bash
# Reinstall PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: `spikingjelly` import error

```bash
# Install directly from PyPI
pip install spikingjelly --upgrade
```

### Issue: CUDA not available

**Check:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**If False:**
1. Verify NVIDIA drivers: `nvidia-smi`
2. Check CUDA installation
3. Reinstall PyTorch with correct CUDA version
4. Use CPU mode if GPU unavailable: `--device cpu`

### Issue: Memory errors during install

```bash
# Use pip with no cache
pip install --no-cache-dir -r requirements.txt
```

### Issue: Permission errors

```bash
# Use user install
pip install --user -r requirements.txt

# Or use sudo (not recommended)
sudo pip install -r requirements.txt
```

---

## Alternative Installation Methods

### Using Conda

```bash
# Create conda environment
conda create -n chess-snn python=3.10
conda activate chess-snn

# Install PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other packages
pip install -r requirements.txt
```

### Using Docker

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "scripts/train.py"]
```

```bash
# Build and run
docker build -t chess-snn-rl .
docker run --gpus all chess-snn-rl
```

---

## Verification Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created
- [ ] PyTorch with CUDA support installed
- [ ] All requirements installed (`pip list`)
- [ ] CUDA available (`python -c "import torch; print(torch.cuda.is_available())"`)
- [ ] GPU count correct (`python -c "import torch; print(torch.cuda.device_count())"`)
- [ ] All tests pass (`python verify_install.py`)
- [ ] Sample training runs (`python scripts/train.py --iterations 1 --games-per-iter 10`)

---

## Next Steps

After successful installation:

1. **Read Documentation:**
   - `README.md` - Main documentation
   - `QUICKSTART.md` - Quick start guide
   - `MULTI_GPU_GUIDE.md` - Multi-GPU setup

2. **Run Tests:**
   ```bash
   python tests/test_encoder.py
   python tests/test_model.py
   python tests/test_evaluation.py
   ```

3. **Start Training:**
   ```bash
   # CPU (testing)
   python scripts/train.py --iterations 10 --device cpu

   # Single GPU
   python scripts/train.py --iterations 100 --device cuda

   # 8 GPUs
   python scripts/train_multi_gpu.py --gpus 0,1,2,3,4,5,6,7 --iterations 200
   ```

4. **Monitor Progress:**
   ```bash
   tail -f checkpoints/training_stats.json
   ```

---

## Support

If you encounter issues:

1. Check this installation guide
2. Review troubleshooting section in `README.md`
3. Run `python verify_install.py` for diagnostics
4. Check existing [GitHub issues](https://github.com/yourusername/chess-snn-rl/issues)
5. Open a new issue with error logs

---

**Happy training! 🚀**
