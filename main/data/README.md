# Data Directory

This directory contains input data for EventMamba-FX.

## Background Events

Place DSEC background event H5 files in `bg_events/`:

```
data/
└── bg_events/
    ├── zurich_city_00_a.h5
    ├── zurich_city_01_a.h5
    └── ... (other DSEC event files)
```

Each H5 file should contain standard DVS event format:
- `/events/t` - timestamps (microseconds)
- `/events/x` - x coordinates
- `/events/y` - y coordinates
- `/events/p` - polarity (1/-1)

## Flare7K Dataset

The Flare7K dataset path is configured in `configs/config.yaml`:

```yaml
data:
  flare7k_path: "/path/to/Flare7Kpp/Flare7Kpp"
```

See `DATA_SETUP.md` for detailed dataset preparation instructions.
