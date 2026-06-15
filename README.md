# HeimDAS

Distributed Acoustic Sensing (DAS) produces massive volumes of strain-rate data
from fibre-optic cables. Manually scanning this data for events of interest is
impractical. HeimDAS automates the process: it learns what normal cable behaviour
looks like, sets a statistically principled detection threshold, and flags
anomalous events without requiring labelled training data or manual tuning.

## How it works

A convolutional autoencoder reconstructs short signal patches. Patches that
reconstruct poorly are anomalous. The detection threshold is set automatically
using Extreme Value Theory (Generalised Pareto Distribution), and refits hourly
to track environmental drift. Detected events are segmented, georeferenced in
distance and time, and exported as JSON plus publication-quality figures.

## Quickstart

```bash
git clone git@github.com:your-user/HeimDAS.git
cd HeimDAS
python -m venv .venv && source .venv/bin/activate
pip install -e .
heimdas /path/to/hdf5/data --output-dir ./results --verbose
```

Add `--device cuda` for GPU acceleration.

## Usage

```bash
# Default: 60 min calibration, output to ./output/
heimdas /path/to/hdf5/data

# Custom settings
heimdas /path/to/hdf5/data \
    --output-dir ./results \
    --calibration-minutes 30 \
    --cable-name "My Cable" \
    --device cuda \
    --batch-size 128 \
    --verbose
```

Each run creates a timestamped folder containing:
- `hour_NNN.png` -- two-panel figures (raw waterfall + detection overlay)
- `detections_hour_NNN.json` -- event metadata with physical coordinates
- `threshold_history.png` -- threshold evolution over time

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATA_DIR` | required | Directory of `.hdf5` DAS files |
| `--output-dir` | `./output` | Output directory |
| `--calibration-minutes` | `60` | Calibration window (minutes) |
| `--cable-name` | auto | Cable name for plot titles |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--batch-size` | `64` | Inference batch size |
| `--q` | `1e-3` | False-alarm probability |
| `--cca-stride` | `90` | Temporal pooling stride |
| `--verbose` | off | Debug logging |

## Input format

A directory of HDF5 files (one per acquisition burst, typically 10 s each).
Each file must contain:

```
/data          (n_samples, n_channels)  int16 or float32
/header/time   UNIX timestamp of first sample
/header/dx     Channel spacing in metres
/header/dt     Sample interval in seconds
```

Files are sorted by filename. Any native sample rate is supported (data is
anti-alias filtered and resampled to 400 Hz internally).

## Pipeline

1. **Autoencoder inference** -- reconstructs patches; high residual = anomaly
2. **Anti-aliased resampling** -- FIR decimation to 400 Hz
3. **GPD calibration** -- automatic threshold from Extreme Value Theory
4. **Adaptive refit** -- hourly threshold update to track drift
5. **Segmentation** -- hysteresis, CCA, morphology, proximity merging
6. **Visualization** -- SciencePlots figures with physical axes

## Project structure

```
heimdas/
    cli.py          CLI and pipeline orchestration
    config.py       Configuration dataclass
    loader.py       HDF5 discovery and loading
    preprocess.py   Resampling and normalization
    model.py        ConvAutoencoder architecture
    calibration.py  GPD threshold fitting
    detection.py    Event segmentation
    visualize.py    Plotting
    weights/        Pretrained model (bundled)
```

## Requirements

- Python >= 3.9
- PyTorch >= 2.1
- All other dependencies install automatically via `pip install -e .`

## License

MIT
