# Event Deflare - 3D UNet Implementation

**Event-based flare removal system using TrueResidualUNet3D.**

---

## Installation

### Requirements

- Python 3.9
- CUDA-capable GPU (4GB+ VRAM recommended)
- Conda or Miniconda

### Quick Install

```bash
# Create environment
conda create -n event_deflare python=3.9 \
  pytorch torchvision pytorch-cuda=12.1 \
  pytorch-3dunet=1.9.1 \
  numpy scipy matplotlib pandas h5py opencv scikit-image \
  pyyaml tqdm tensorboard \
  -c pytorch -c nvidia -c conda-forge -y

# Activate environment
conda activate event_deflare

# Verify installation
python -c "import torch; from pytorch3dunet.unet3d.model import ResidualUNet3D; print('✅ Installation successful')"
```

---

## Data Preparation

### Input Format

Event data in HDF5 format:

```
file.h5
└── events/
    ├── t: int64 array (timestamps in microseconds)
    ├── x: uint16 array (x-coordinates, 0-639)
    ├── y: uint16 array (y-coordinates, 0-479)
    └── p: int8 array (polarity, -1 or 1)
```

### Dataset Structure

```
data/
├── train/
│   ├── noisy/          # Input events with flare
│   └── clean/          # Ground truth (deflared events)
└── test/
    ├── noisy/
    └── clean/
```

Update paths in `configs/train_config.yaml`:

```yaml
loaders:
  train_noisy_dir: "data/train/noisy"
  train_clean_dir: "data/train/clean"
  val_noisy_dir: "data/test/noisy"
  val_clean_dir: "data/test/clean"
```

---

## Usage

### Training

```bash
conda activate event_deflare

# Standard training
python main.py train --config configs/train_config.yaml

# Debug mode (with visualization)
python main.py train --config configs/train_config.yaml --debug
```

### Testing

```bash
# Evaluate on test set
python main.py test --config configs/test_config.yaml

# Baseline (encode-decode only)
python main.py test --config configs/test_config.yaml --baseline

# With visualization
python main.py test --config configs/test_config.yaml --debug
```

### Inference

**Single File**:
```bash
python inference_single.py --input path/to/file.h5 --mode normal
```

**Batch Processing**:
```bash
python main.py inference --config configs/inference_config.yaml
```

---

## Configuration

### Model Size

Edit `configs/train_config.yaml`:

```yaml
model:
  name: TrueResidualUNet3D
  f_maps: [32, 64, 128, 256]  # 7M params (standard)
  num_levels: 4
  # For more capacity: [64, 128, 256, 512] → 28M params
```

### Data Processing

```yaml
loaders:
  sensor_size: [480, 640]
  segment_duration_us: 20000  # 20ms per segment
  num_bins: 8                 # 8 temporal bins
  num_segments: 5             # 100ms file → 5 segments
```

---

## Troubleshooting

**Import Error**:
```bash
conda install -c conda-forge pytorch-3dunet=1.9.1
```

**CUDA Out of Memory**:
```yaml
# Reduce model size in config
f_maps: [32, 64, 128]
```

**File Not Found**:
- Check data paths in `configs/*.yaml`
- Verify H5 file format

**Windows Paths**:
```bash
# Use forward slashes
python inference_single.py --input "C:/data/file.h5"
```

---

## Quick Reference

```bash
# Install
conda create -n event_deflare python=3.9 pytorch pytorch-cuda=12.1 pytorch-3dunet=1.9.1 -c pytorch -c nvidia -c conda-forge -y

# Train
python main.py train --config configs/train_config.yaml

# Test
python main.py test --config configs/test_config.yaml

# Inference
python inference_single.py --input file.h5 --mode normal
```
