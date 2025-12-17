# Installation

The codebase consists of two independent components:
- **Dataset Generator** (`main/`): Simulates event-based lens flare data
- **E-DeflareNet** (`Unet_main/`): Neural network for flare removal

## Prerequisites

- Python 3.9-3.10
- CUDA-capable GPU (recommended)
- Git

## Step 1: Dataset Generator Setup

```bash
cd main

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch, h5py, cv2; print('✅ Dataset generator ready')"
```

## Step 2: E-DeflareNet Setup

```bash
cd ../Unet_main

# Create conda environment
conda create -n event_deflare python=3.9 \
  pytorch torchvision pytorch-cuda=12.1 \
  pytorch-3dunet=1.9.1 \
  numpy scipy matplotlib pandas h5py opencv scikit-image \
  pyyaml tqdm tensorboard \
  -c pytorch -c nvidia -c conda-forge -y

# Activate environment
conda activate event_deflare

# Verify installation
python -c "from pytorch3dunet.unet3d.model import ResidualUNet3D; print('✅ E-DeflareNet ready')"
```

## Quick Verification

```bash
# Test dataset generator
cd main
python main.py --test --num-sequences 2

# Test model inference (after training or downloading pretrained weights)
cd ../Unet_main
conda activate event_deflare
python inference_single.py --input <path_to_h5_file> --mode normal
```

## Troubleshooting

**CUDA version mismatch**: Replace `pytorch-cuda=12.1` with your CUDA version (e.g., `11.8`)

**pytorch-3dunet not found**:
```bash
conda install -c conda-forge pytorch-3dunet=1.9.1
```

**Import errors**: Ensure you're in the correct directory and environment before running scripts.
