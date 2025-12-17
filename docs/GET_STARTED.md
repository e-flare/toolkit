# Getting Started

Complete workflow for using the E-Deflare framework.

## Option A: Use Pre-Generated Dataset (Recommended)

### 1. Download Dataset & Pretrained Model

**Download Dataset** from [HuggingFace](https://huggingface.co/datasets/E-Deflare/data):
- **E-Flare-2.7K**: Training/validation set (simulated)
- **E-Flare-R**: Real-world test set

Organize as (matching config paths):
```
Unet_main/data/
├── background_with_flare_events/         # Training: events with flare
├── background_with_light_events/         # Training: ground truth (deflared)
├── background_with_flare_events_test/    # Validation/test: events with flare
└── background_with_light_events_test/    # Validation/test: ground truth
```

**Download Pretrained Model**:
```bash
# Option 1: Baidu Netdisk (faster for users in China)
# Download from: https://pan.baidu.com/s/1MG9SG3ZMjRWDALSC5f43hg?pwd=ejsj (Code: ejsj)

# Option 2: HuggingFace (international users)
# Download checkpoint.pth from: https://huggingface.co/datasets/E-Deflare/data/tree/main

# Place checkpoint in:
mkdir -p Unet_main/checkpoints/pretrained
# Move downloaded file to: Unet_main/checkpoints/pretrained/checkpoint.pth
```

Update model path in `configs/test_config.yaml` or `configs/inference_config.yaml`:
```yaml
model:
  path: checkpoints/pretrained/checkpoint.pth
```

### 2. Train E-DeflareNet (Optional)

```bash
cd Unet_main
conda activate event_deflare

# Update data paths in configs/train_config.yaml
# Then start training
python main.py train --config configs/train_config.yaml
```

### 3. Evaluate with Pretrained Model

```bash
cd Unet_main
conda activate event_deflare

# E-Flare-2.7K test set
python main.py test --config configs/test_config.yaml

# E-Flare-R (real-world)
# Update test_config.yaml with E-Flare-R paths, then:
python main.py test --config configs/test_config.yaml
```

### 4. Inference on New Data

```bash
# Single file inference with pretrained model
python inference_single.py --input path/to/event.h5 --mode normal

# Batch processing
python main.py inference --config configs/inference_config.yaml
```

**Note**: Steps 2 (training) and 3-4 (evaluation/inference) are independent. You can skip training and directly use the pretrained model for evaluation or inference.

---

## Option B: Generate Custom Dataset

### 1. Prepare Source Data

See `main/DATA_SETUP.md` for:
- Flare7K dataset setup
- DSEC background event data preparation

### 2. Generate Training Data

```bash
cd main

# Generate full dataset (2545 train + 175 test samples)
python main.py

# Generate small test dataset
python main.py --test --num-sequences 10

# Debug mode (with visualizations)
python main.py --debug
```

Output: H5 files in `main/output/data/`

### 3. Organize for Training

```bash
# Move generated data to Unet_main (match config paths)
cd ..
mkdir -p Unet_main/data/background_with_flare_events
mkdir -p Unet_main/data/background_with_light_events

# Copy training data
cp main/output/data/full_scene_events/*.h5 Unet_main/data/background_with_flare_events/
cp main/output/data/background_with_light_events/*.h5 Unet_main/data/background_with_light_events/
```

### 4. Train & Evaluate

Follow steps 2-4 from Option A.

---

## Quick Reference

```bash
# Setup (one-time) - see INSTALL.md for full commands
cd main && pip install -r requirements.txt
cd ../Unet_main && conda create -n event_deflare python=3.9 pytorch pytorch-cuda=12.1 pytorch-3dunet -c pytorch -c nvidia -c conda-forge -y
conda activate event_deflare

# Generate data (optional)
cd main && python main.py --test --num-sequences 50

# Train model
cd Unet_main && python main.py train --config configs/train_config.yaml

# Test model
python main.py test --config configs/test_config.yaml

# Inference
python inference_single.py --input file.h5 --mode normal
```

## Configuration Tips

**Dataset Generator** (`main/configs/config.yaml`):
- `num_sequences`: Number of samples to generate
- `sensor_size`: Event camera resolution (default: [480, 640])
- `duration_us`: Sequence duration in microseconds

**E-DeflareNet** (`Unet_main/configs/train_config.yaml`):
- `f_maps`: Network capacity ([32,64,128,256] for 7M params)
- `num_bins`: Temporal bins for voxel grid (default: 8)
- `segment_duration_us`: Time window per sample (default: 20ms)

## Troubleshooting

**Generator fails**: Check Flare7K path in `main/configs/config.yaml`

**Training OOM**: Reduce `f_maps` or batch size in `train_config.yaml`

**Inference error**: Verify H5 file format contains `/events/{t,x,y,p}` groups
