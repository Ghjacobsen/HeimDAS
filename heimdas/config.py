"""Pipeline configuration for HeimDAS.

All tuneable parameters are collected in a single PipelineConfig dataclass.
Every field has a sensible default derived from extensive experimentation on
Storebælt (Config B) and SHEFA fibre-optic cables. Below is a per-stage
explanation of what each group of parameters controls.

Preprocessing
-------------
The autoencoder was trained on data sampled at 400 Hz. Input data at any
native sample rate is decimated (or interpolated) to this target before
inference. The patch_size is fixed at 128×128 — this is baked into the
pretrained weights and cannot be changed without retraining.

Edge channels (first/last `edge_crop` channels) are excluded from
calibration because coupling artefacts at cable terminations produce
extreme residuals that would bias the GPD fit.

Calibration (GPD / Extreme Value Theory)
-----------------------------------------
The pipeline derives its detection threshold τ from data using the
Generalised Pareto Distribution (GPD), fitted via the Grimshaw Maximum
Likelihood Estimator. The false-alarm probability `q` sets the return
period T = 1/q: on average, 1 in T normal samples exceeds τ. The initial
quantile `q_init` determines how much of the residual distribution is
treated as "tail" for GPD fitting — higher values focus on fewer extreme
samples (more precise tail but less stable), lower values include more
bulk data (more stable but risks violating the GPD assumption).

Before fitting, residuals above `calibration_clip_sigma` × MAD are
clipped to prevent rare instrument glitches from dominating the tail.

Rolling Threshold
-----------------
After initial calibration on the first `calibration_minutes` of data,
the threshold is refit every hour using the PREVIOUS hour's residuals.
This adapts to changing environmental conditions (temperature, tides,
anthropogenic noise) without peeking into the future.

Temporal Compression
--------------------
Before Connected Component Analysis (CCA), the residual is temporally
compressed by max-pooling blocks of `cca_stride` samples into single
pixels. At 400 Hz with stride=90, this produces ~4.4 detection pixels
per second — enough to resolve events lasting ≥0.2 s while keeping CCA
tractable. Increasing cca_stride compresses more aggressively (faster,
fewer components) but loses ability to separate temporally close events.

Detection & Segmentation
-------------------------
After compression, detection proceeds via:

1. Median filter (suppresses salt-and-pepper noise).
2. Hysteresis thresholding: pixels above τ are "seeds"; all connected
   pixels above τ × hysteresis_ratio are included. Lower ratio =
   events expand further (captures weak tails), higher = tighter bounds.
3. Morphological opening (removes thin filaments) + closing (fills gaps).
4. CCA labels connected regions. Components smaller than k_min pixels
   are discarded.
5. Proximity merging: nearby components (within gap_t seconds temporally
   AND gap_x metres spatially) are merged via Union-Find.
6. Display filter: only events exceeding min_display_area AND
   min_display_spatial_px are rendered in colour. Smaller events appear
   in gray. This is purely cosmetic — it does NOT affect detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PipelineConfig:
    """Complete set of pipeline hyperparameters.

    Every field includes a comment explaining its purpose and the effect
    of changing it from the default value.
    """

    # ─── Preprocessing ────────────────────────────────────────────────────

    # Target sample rate (Hz). All input is resampled to this before inference.
    # The autoencoder was trained at 400 Hz. Lowering reduces temporal resolution
    # and may miss short events; raising above 400 is not recommended.
    target_fs: float = 400.0

    # Side length of square patches fed to the autoencoder.
    # MUST remain 128 to match pretrained weights. Changing requires retraining.
    patch_size: int = 128

    # Channels excluded from EACH cable edge during calibration.
    # Edge channels have poor coupling and extreme residuals that bias GPD.
    # Increase for noisier edges; decrease only if cable termination is clean.
    edge_crop: int = 128

    # ─── Calibration (GPD / EVT) ─────────────────────────────────────────

    # Duration (minutes) of the initial quiet-period calibration window.
    # Shorter = noisier threshold estimate. Longer = more stable but needs
    # more quiet data upfront before detection begins.
    calibration_minutes: int = 60

    # False-alarm probability for GPD threshold. T = 1/q = expected number of
    # normal samples per false exceedance. Smaller q = higher threshold =
    # fewer false alarms but may miss weak events. Larger = more sensitive.
    q: float = 1e-3

    # Initial quantile for GPD tail extraction. Only the top (1 - q_init)
    # fraction of residuals is modelled as GPD. Higher = fewer tail samples,
    # sharper focus on extremes but less stable. Lower = more data in tail,
    # more stable fit but may violate GPD assumption.
    q_init: float = 0.85

    # Residuals above median + clip_sigma × MAD are clipped before GPD fit.
    # Prevents rare instrument glitches from dominating the tail.
    # Lower = more aggressive clipping (safer but potentially loses real signal).
    calibration_clip_sigma: float = 50.0

    # Factor for detecting broken/dead channels. Channels whose median residual
    # exceeds this factor × global median are excluded from calibration.
    calibration_outlier_factor: float = 10.0

    # Maximum number of residual samples used for GPD fit (memory cap).
    # If the calibration window exceeds this, rows are randomly subsampled.
    max_calibration_samples: int = 20_000_000

    # ─── Rolling Threshold ────────────────────────────────────────────────

    # At each hour boundary, refit threshold using the PREVIOUS hour's residual.
    # Value of 1 = use only last hour (no sliding window). Adapts to environmental
    # drift without lookahead.
    rolling_window_hours: int = 1

    # ─── Temporal Compression ─────────────────────────────────────────────

    # Max-pool block size (samples → 1 detection pixel). At 400 Hz, stride=90
    # yields ~4.4 pixels/s. Larger = more compression, merges nearby transients,
    # faster but loses resolution for events shorter than stride/fs seconds.
    # Smaller = finer resolution but exponentially more CCA components.
    cca_stride: int = 90

    # ─── Detection & Segmentation ─────────────────────────────────────────

    # Extend threshold = τ × hysteresis_ratio. Seeds are pixels > τ, extended
    # to all connected pixels > τ × ratio. Lower = events expand further
    # (captures weak tails). Higher = tighter event boundaries.
    hysteresis_ratio: float = 0.5

    # Median filter kernel (time, space) applied before thresholding.
    # Suppresses salt-and-pepper noise. Larger = more smoothing but may
    # erode edges of small events. (3,3) is minimal smoothing.
    median_kernel: tuple[int, int] = (3, 3)

    # Morphological binary opening kernel (time, space). Removes thin noise
    # filaments surviving thresholding. Larger = removes more thin structures
    # but may eliminate legitimate narrow events.
    open_kernel: tuple[int, int] = (3, 5)

    # Morphological binary closing kernel (time, space). Fills small internal
    # gaps in event blobs. Larger = more gap-filling but may merge distinct
    # nearby events into one.
    close_kernel: tuple[int, int] = (5, 7)

    # Minimum component area (pixels) after CCA, BEFORE proximity merging.
    # Higher = discards more small fragments early (faster merge, cleaner).
    # Lower = retains fragments that may merge into real events.
    k_min: int = 30

    # Maximum temporal gap (seconds) for proximity merging. Components whose
    # bounding boxes are within this gap in time are merged. Larger = connects
    # temporally separated bursts. Smaller = only merges very close components.
    gap_t_seconds: float = 5.0

    # Maximum spatial gap (metres) for proximity merging. Larger = connects
    # spatially separated clusters (useful for diffuse seismic arrivals).
    # Smaller = only merges spatially adjacent blobs.
    gap_x_metres: float = 80.0

    # Minimum area (pixels) for an event to be rendered in COLOUR on the
    # detection map. Events below this threshold are shown in gray.
    # Purely cosmetic — does NOT affect detection logic.
    min_display_area: int = 5000

    # Minimum spatial span (channels) for colour rendering.
    # Filters out narrow vertical streaks from colour highlighting.
    # Purely cosmetic — does NOT affect detection logic.
    min_display_spatial_px: int = 30

    # ─── Output ───────────────────────────────────────────────────────────

    # Directory where hourly PNG visualizations are saved.
    output_dir: Path = field(default_factory=lambda: Path("output"))

    # Inference batch size (patches per forward pass). Reduce if GPU OOM.
    batch_size: int = 64
