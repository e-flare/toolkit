# License

## Primary License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](../LICENSE) file for full terms.

**Copyright © 2025 E-Deflare Authors**

## Third-Party Components

This codebase integrates several third-party components with their own licenses:

### 1. DVS-Voltmeter Event Simulator
- **Location**: `main/simulator/DVS-Voltmeter-main/`
- **Description**: Physics-based event camera simulator
- **License**: Check original repository for license terms
- **Reference**: [DVS-Voltmeter](https://github.com/neuromorphicsystems/event-simulator) (if applicable)

### 2. pytorch-3dunet
- **Usage**: 3D U-Net backbone in `Unet_main/`
- **License**: MIT License
- **Installation**: `conda install -c conda-forge pytorch-3dunet=1.9.1`
- **Reference**: [pytorch-3dunet](https://github.com/wolny/pytorch-3dunet)

### 3. Flare7K Dataset
- **Usage**: Source data for flare synthesis (`main/` data preparation)
- **License**: Check [Flare7K dataset page](https://github.com/ykdai/Flare7K) for usage terms
- **Citation**: Required if using generated data based on Flare7K

### 4. DSEC Dataset
- **Usage**: Background event sequences for dataset generation
- **License**: Check [DSEC dataset page](https://dsec.ifi.uzh.ch/) for terms
- **Citation**: Required if using DSEC-derived event data

## Commercial Use

While the core E-Deflare framework code is under Apache 2.0 (permissive for commercial use), ensure compliance with:
- Third-party component licenses (pytorch-3dunet MIT, DVS-Voltmeter terms)
- Dataset licenses (Flare7K, DSEC usage restrictions if applicable)
- Citation requirements from all referenced works

**Recommendation**: Review all upstream licenses before commercial deployment.

## Citation Requirement

If you use this codebase, datasets, or models in your work, please cite:

```bibtex
@article{han2025e-deflare,
    title   = {Learning to Remove Lens Flare in Event Camera},
    author  = {Haiqian Han and Lingdong Kong and Jianing Li and Ao Liang and Chengtao Zhu and Jiacheng Lyu and Lai Xing Ng and Xiangyang Ji and Wei Tsang Ooi and Benoit R. Cottereau},
    journal = {arXiv preprint arXiv:2512.09016},
    year    = {2025}
}
```

---

**Disclaimer**: This document provides a summary. Users are responsible for verifying license compliance for their specific use case.
