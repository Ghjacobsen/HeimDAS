"""Signal preprocessing for HeimDAS.

This module handles two critical transformations applied before the data
reaches the autoencoder:

1. Temporal resampling — The autoencoder was trained on 400 Hz data.
   Cables running at different native rates (e.g. 800 Hz for SHEFA) are
   decimated to match. Cables below 400 Hz are upsampled via linear
   interpolation (rare in practice).

2. Robust Z-score normalization — Each channel is independently
   standardized using the robust formula:

       z = (x - median) / (1.4826 × MAD)

   where MAD = median(|x - median|). The constant 1.4826 makes the
   denominator consistent with the standard deviation for Gaussian data.
   Using median/MAD instead of mean/std makes the normalization resistant
   to outliers (anomalies don't inflate the scale).

   Statistics (per-channel median and MAD) are computed ONCE from the
   calibration window and reused for all subsequent detection windows.
   This ensures the model sees a consistent data distribution and that
   anomalies remain visible (they won't be normalized away).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.signal import decimate, resample_poly

log = logging.getLogger(__name__)

# Constant to make MAD consistent with std for Gaussian distributions
_MAD_SCALE = 1.4826


@dataclass
class NormStats:
    """Per-channel normalization statistics computed from calibration data."""

    median: np.ndarray  # shape (n_channels,)
    mad: np.ndarray  # shape (n_channels,) — median absolute deviation


def resample_temporal(data: np.ndarray, fs_native: float, fs_target: float) -> np.ndarray:
    """Resample data along the time axis to match the target sample rate.

    Uses anti-aliased FIR decimation (scipy.signal.decimate) for integer
    downsampling factors, or polyphase resampling for non-integer ratios.
    The anti-aliasing low-pass filter is critical: without it, high-frequency
    content above the new Nyquist frequency would fold back (alias) into the
    passband, creating artificial noise that confuses the autoencoder.

    Args:
        data: Input array shape (n_samples, n_channels).
        fs_native: Original sample rate in Hz.
        fs_target: Desired sample rate in Hz.

    Returns:
        Resampled array with adjusted number of time samples.
    """
    if abs(fs_native - fs_target) < 0.1:
        return data

    ratio = fs_native / fs_target

    if ratio > 1 and abs(ratio - round(ratio)) < 0.01:
        # Integer decimation (most common case: 800→400, 200→400 won't hit this)
        factor = int(round(ratio))
        log.info("Decimating %d× (%.0f Hz → %.0f Hz)", factor, fs_native, fs_target)
        return decimate(data, factor, axis=0, ftype="fir").astype(np.float32)

    # Non-integer ratio: polyphase resampling
    # Find simplest integer up/down ratio
    from fractions import Fraction

    frac = Fraction(fs_target / fs_native).limit_denominator(100)
    up, down = frac.numerator, frac.denominator
    log.info("Resampling %d/%d (%.0f Hz → %.0f Hz)", up, down, fs_native, fs_target)
    return resample_poly(data, up, down, axis=0).astype(np.float32)


def compute_normalization_stats(data: np.ndarray) -> NormStats:
    """Compute per-channel median and MAD from a calibration window.

    These statistics are computed once and reused for all subsequent
    windows to ensure consistent normalization.

    Args:
        data: Calibration data, shape (n_samples, n_channels). Should
              already be resampled to target_fs.

    Returns:
        NormStats containing per-channel median and MAD arrays.
    """
    median = np.median(data, axis=0)
    mad = np.median(np.abs(data - median[np.newaxis, :]), axis=0)

    # Floor MAD to prevent division by zero on dead channels
    mad = np.maximum(mad, 1e-8)

    log.info(
        "Normalization stats: median range [%.4f, %.4f], MAD range [%.6f, %.6f]",
        median.min(), median.max(), mad.min(), mad.max(),
    )
    return NormStats(median=median, mad=mad)


def normalize(data: np.ndarray, stats: NormStats) -> np.ndarray:
    """Apply robust Z-score normalization using precomputed statistics.

    Formula: z = (x - median) / (1.4826 × MAD)

    Args:
        data: Input array shape (n_samples, n_channels).
        stats: Precomputed NormStats from calibration window.

    Returns:
        Normalized array (same shape), dtype float32.
    """
    scale = _MAD_SCALE * stats.mad
    return ((data - stats.median[np.newaxis, :]) / scale[np.newaxis, :]).astype(np.float32)
