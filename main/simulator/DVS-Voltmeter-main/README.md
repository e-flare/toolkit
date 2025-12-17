# DVS-Voltmeter Event Simulator

This is the DVS-Voltmeter physical event simulator embedded in EventMamba-FX.

## Description

DVS-Voltmeter simulates realistic DVS (Dynamic Vision Sensor) events from image sequences using a physics-based Brownian Motion with Drift model.

**Reference:** DVS-Voltmeter - Stochastic Process-based Event Simulator for Dynamic Vision Sensors

## Usage

This simulator is automatically invoked by the EventMamba-FX system. Direct usage is not required.

The simulator is called from `src/dvs_flare_integration.py` with pre-configured parameters from `configs/config.yaml`.

## Configuration

DVS parameters are configured in the main `configs/config.yaml`:

```yaml
data:
  event_simulator:
    dvs_voltmeter:
      simulator_path: "simulator/DVS-Voltmeter-main"
      timeout_sec: 60
      parameters:
        k1_range: [5.0, 16.0]  # Randomized k1 parameter
        # ... see config.yaml for full parameters
```

## Files

- `main.py` - Simulator entry point (called by EventMamba-FX)
- `src/simulator.py` - Core DVS simulation logic
- `src/simulator_utils.py` - Utility functions
- `src/config.py` - Simulator configuration

## License

See original DVS-Voltmeter repository for license information.
