# Data Preparation Guide

This document describes the required dataset organization for EventMamba-FX.

## Required Datasets

### 1. Flare7K Dataset

Download the Flare7K dataset and organize as follows:

```
Flare7Kpp/
└── Flare7Kpp/
    ├── Scattering_Flare/
    │   ├── Compound_Flare/          # Used for training/validation
    │   │   ├── 1.png
    │   │   ├── 2.png
    │   │   └── ...
    │   └── Glare_with_shimmer/      # Used for test mode
    │       ├── 1.png
    │       ├── 2.png
    │       └── ...
    └── Light_Source/                # Corresponding light source images
        ├── 1.png                    # Same filenames as Scattering_Flare
        ├── 2.png
        └── ...
```

**Update config.yaml:**
```yaml
data:
  flare7k_path: "/path/to/Flare7Kpp/Flare7Kpp"
```

### 2. DSEC Background Event Data

Prepare background event data in H5 format:

```
data/
└── bg_events/
    ├── zurich_city_00_a.h5
    ├── zurich_city_01_a.h5
    ├── interlaken_00_c.h5
    └── ...
```

Each H5 file should contain DVS events in standard format:
- `/events/t` - timestamps (microseconds)
- `/events/x` - x coordinates
- `/events/y` - y coordinates
- `/events/p` - polarity (1/-1)

**Notes:**
- DSEC event data is publicly available from the DSEC dataset
- Extract and convert to H5 format with the above structure
- Multiple files will be sampled randomly during generation

## Directory Structure Checklist

Before running the system, ensure:

- [ ] Flare7K dataset downloaded and organized
- [ ] `flare7k_path` in config.yaml points to correct directory
- [ ] Background event H5 files placed in `data/bg_events/`
- [ ] At least one H5 file exists in `data/bg_events/`
- [ ] DVS-Voltmeter simulator path is correct in config.yaml

## Verify Setup

Run a quick test to verify data paths:

```bash
python main.py --test --debug --num-sequences 1
```

This will generate one test sequence and should complete without errors if data is properly configured.
