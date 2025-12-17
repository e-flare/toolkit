# EventMamba-FX: Two-Step Event-Based Lens Flare Generator

Core implementation of the two-step event-based lens flare generation system. Generates realistic DVS event data by combining physical lens flare simulation with event-based camera simulation.

## Installation

### Requirements
- Python 3.10+
- CUDA-capable GPU (recommended for GLSL reflection flare generation)

### Setup

```bash
# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

See `DATA_SETUP.md` for detailed instructions on:
- Downloading and organizing Flare7K dataset
- Preparing DSEC background event data
- Required directory structure

## Usage

### Basic Usage

```bash
# Complete pipeline (Step 1: flare generation + Step 2: event composition)
python main.py

# With debug visualizations (recommended for first run)
python main.py --debug
```

### Step-by-Step Execution

```bash
# Step 1 only: Generate flare events
python main.py --step 1 --debug

# Step 2 only: Compose events (requires Step 1 first)
python main.py --step 2 --debug
```

### Test Mode

```bash
# Generate test dataset (10 sequences, upper flare position, fixed seed)
python main.py --test --debug

# Custom number of test sequences
python main.py --test --num-sequences 50

# Custom flare position: 'upper' (default), 'center', or 'random'
python main.py --test --flare-position center
```

## Configuration

Edit `configs/config.yaml` to customize:
- Number of training/validation/test sequences
- DVS simulator parameters
- Output paths
- Flare synthesis parameters

## Output Structure

```
output/
├── data/                           # Generated event data (H5 format)
│   ├── flare_events/              # Step 1: Flare events (scattering + reflection)
│   ├── light_source_events/       # Step 1: Light source events
│   ├── background_with_light_events/  # Step 2: Background + light source
│   └── full_scene_events/         # Step 2: Full scene (background + light + flare)
└── debug/                         # Debug visualizations (--debug mode only)
    ├── flare_generation/
    ├── light_source_generation/
    └── event_composition/
```

Test mode outputs to `output/test/` with the same structure.

## Output Format

All event data is saved in standard DVS H5 format:
- `/events/t` - timestamps (microseconds)
- `/events/x` - x coordinates
- `/events/y` - y coordinates
- `/events/p` - polarity (1/-1)

## Citation

Please refer to the paper for detailed methodology and technical explanations.
