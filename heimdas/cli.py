"""HeimDAS command-line interface and pipeline orchestrator.

This is the main entry point for HeimDAS. It provides a single CLI command
that runs the full anomaly detection pipeline end-to-end:

    python -m heimdas <DATA_DIR> [OPTIONS]

The pipeline flow:
    1. Discover and sort HDF5 files by timestamp.
    2. Read cable metadata (sample rate, channel spacing dx).
    3. Load the calibration window (first N minutes of data).
    4. Resample to 400 Hz (model training rate) with anti-aliasing.
    5. Compute per-channel normalization statistics (median/MAD).
    6. Run autoencoder inference → reconstruction residuals.
    7. Fit GPD threshold from calibration residuals → initial τ.
    8. For each subsequent hour of data:
       a. Load and stitch files into continuous segment.
       b. Resample and normalize using calibration statistics.
       c. Run autoencoder inference → residuals.
       d. Detect events using current τ.
       e. Export detection segments as JSON.
       f. Render 2-panel PNG (Raw DAS Waterfall + HeimDAS Detection).
       g. Refit τ from this hour's residuals (used for NEXT hour).
    9. Render threshold evolution staircase plot.
   10. Log summary of all hours processed.

Each run creates a timestamped output folder containing all artefacts.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .calibration import compute_threshold
from .config import PipelineConfig
from .detection import count_display_events
from .loader import CableMetadata, discover_files, group_files_by_hour, load_files, read_metadata
from .model import infer_residual, load_model
from .preprocess import compute_normalization_stats, normalize, resample_temporal
from .visualize import render_hour, render_threshold_history

log = logging.getLogger(__name__)


def app() -> None:
    """Parse CLI arguments and run the pipeline."""
    parser = argparse.ArgumentParser(
        prog="heimdas",
        description="HeimDAS — Unsupervised anomaly detection for DAS.",
    )
    parser.add_argument(
        "data_dir", type=Path, help="Directory containing HDF5 DAS files.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output"), help="Base output directory.",
    )
    parser.add_argument(
        "--calibration-minutes", type=int, default=60, help="Calibration minutes.",
    )
    parser.add_argument(
        "--rolling-window-hours", type=int, default=1, help="Refit every N hours.",
    )
    parser.add_argument(
        "--cable-name", type=str, default=None, help="Cable name for titles.",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Torch device.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Inference batch size.",
    )
    parser.add_argument(
        "--q", type=float, default=None,
        help="GPD false-alarm probability (overrides --sensitivity).",
    )
    parser.add_argument(
        "--q-init", type=float, default=0.85, help="GPD initial quantile.",
    )
    parser.add_argument(
        "--sensitivity", type=str, default="normal",
        choices=["low", "normal", "high", "max"],
        help="Detection sensitivity level: low (fewer false alarms), normal, high, max.",
    )
    parser.add_argument(
        "--cca-stride", type=int, default=90, help="Temporal max-pool stride.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")

    args = parser.parse_args()

    # Sensitivity maps to GPD false-alarm probability q.
    # Lower q = higher threshold = fewer false positives.
    # Higher q = lower threshold = more sensitive but more false alarms.
    sensitivity_map = {
        "low": 1e-4,    # Very conservative, almost no false alarms
        "normal": 1e-3,  # Balanced default
        "high": 5e-3,   # More sensitive, some false positives expected
        "max": 1e-2,    # Maximum sensitivity, many false positives
    }
    q = args.q if args.q is not None else sensitivity_map[args.sensitivity]

    run(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        calibration_minutes=args.calibration_minutes,
        rolling_window_hours=args.rolling_window_hours,
        cable_name=args.cable_name,
        device=args.device,
        batch_size=args.batch_size,
        q=q,
        q_init=args.q_init,
        cca_stride=args.cca_stride,
        verbose=args.verbose,
    )


def run(
    data_dir: Path,
    output_dir: Path,
    calibration_minutes: int,
    rolling_window_hours: int,
    cable_name: str | None,
    device: str,
    batch_size: int,
    q: float,
    q_init: float,
    cca_stride: int,
    verbose: bool,
) -> None:
    """Run the full HeimDAS anomaly detection pipeline."""
    # Configure logging
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    t_start_wall = time.perf_counter()

    # Build config
    config = PipelineConfig(
        target_fs=400.0,
        calibration_minutes=calibration_minutes,
        rolling_window_hours=rolling_window_hours,
        output_dir=output_dir,
        batch_size=batch_size,
        q=q,
        q_init=q_init,
        cca_stride=cca_stride,
    )

    log.info("=" * 70)
    log.info("HeimDAS Anomaly Detection Pipeline")
    log.info("=" * 70)
    log.info("Data directory: %s", data_dir)
    log.info("Calibration window: %d minutes", calibration_minutes)
    log.info("Device: %s", device)

    # ── Step 1: Discover files ──
    files = discover_files(data_dir)

    # ── Step 2: Read cable metadata ──
    meta = read_metadata(files[0])
    if cable_name is None:
        cable_name = _infer_cable_name(data_dir, meta)
    log.info(
        "Cable: %s (fs=%.0f Hz, dx=%.3f m, %d channels)",
        cable_name, meta.fs, meta.dx, meta.n_channels,
    )

    # ── Create timestamped run folder ──
    run_timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{run_timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log.info("Run output directory: %s", run_dir)

    # ── Step 3: Split into calibration and detection files ──
    meta_spf = None
    with __import__("h5py").File(files[0], "r") as _f:
        meta_spf = _f["data"].shape[0]
    seconds_per_file = meta_spf / meta.fs
    n_calib_files = max(1, int(round(config.calibration_minutes * 60 / seconds_per_file)))
    n_calib_files = min(n_calib_files, len(files))

    calib_files = files[:n_calib_files]
    detection_files = files[n_calib_files:]

    if not detection_files:
        log.warning(
            "Data shorter than calibration window. "
            "Using all data for both calibration and detection."
        )
        detection_files = files

    # Group detection files by hour
    detection_groups = group_files_by_hour(detection_files, meta.fs)
    log.info(
        "Calibration: %d files (%.0f s), Detection: %d files (%d hours)",
        len(calib_files), len(calib_files) * seconds_per_file,
        len(detection_files), len(detection_groups),
    )

    # ── Step 4: Load and preprocess calibration data (chunked) ──
    log.info("Loading calibration data (%d files)...", len(calib_files))
    calib_chunk_size = max(1, int(30 / seconds_per_file))  # ~30s chunks
    norm_stats = None

    log.info("Loading pretrained model...")
    model = load_model(device)

    # Reservoir sampling: keep only enough rows for GPD fit (avoids OOM).
    # compute_threshold subsamples to max_calibration_samples/n_channels rows anyway.
    max_reservoir_rows = config.max_calibration_samples // meta.n_channels + 100
    reservoir: np.ndarray | None = None
    reservoir_total_rows = 0  # total rows seen (for fair sampling)

    for ci in range(0, len(calib_files), calib_chunk_size):
        chunk_files = calib_files[ci : ci + calib_chunk_size]
        chunk_raw, _ = load_files(chunk_files, expected_channels=meta.n_channels)

        # Resample to target rate (decimate uses anti-aliasing FIR filter)
        chunk_resampled = resample_temporal(chunk_raw, meta.fs, config.target_fs)
        del chunk_raw

        # Compute normalization stats from first chunk only
        if norm_stats is None:
            norm_stats = compute_normalization_stats(chunk_resampled)

        # Normalize and infer
        chunk_norm = normalize(chunk_resampled, norm_stats)
        del chunk_resampled
        chunk_res = infer_residual(model, chunk_norm, device, config.batch_size)
        del chunk_norm

        # Reservoir update: keep a bounded random sample of residual rows
        reservoir_total_rows += chunk_res.shape[0]
        if reservoir is None:
            reservoir = chunk_res
        else:
            reservoir = np.concatenate([reservoir, chunk_res], axis=0)

        # Downsample reservoir if it exceeds budget
        if reservoir.shape[0] > max_reservoir_rows * 2:
            rng = np.random.default_rng(42)
            idx = rng.choice(reservoir.shape[0], size=max_reservoir_rows, replace=False)
            reservoir = reservoir[np.sort(idx)]

    # Final trim
    if reservoir.shape[0] > max_reservoir_rows:
        rng = np.random.default_rng(42)
        idx = rng.choice(reservoir.shape[0], size=max_reservoir_rows, replace=False)
        reservoir = reservoir[np.sort(idx)]
    log.info(
        "Calibration reservoir: %d/%d rows retained",
        reservoir.shape[0], reservoir_total_rows,
    )

    # ── Step 5: Fit GPD threshold ──
    cal_result = compute_threshold(reservoir, config)
    current_tau = cal_result.tau
    log.info("Initial threshold: τ = %.4f (return period T=%d)", current_tau, int(1 / config.q))
    del reservoir

    # ── Step 6: Process each detection hour ──
    # Distance axis: channel_index × dx gives physical distance in metres
    distances_km = np.arange(meta.n_channels) * meta.dx / 1000.0

    # Track threshold evolution for staircase plot (timestamp, tau)
    tau_history: list[tuple[float, float]] = [(meta.t0_unix, current_tau)]

    # Process files in small chunks to avoid OOM
    chunk_size = max(1, int(30 / seconds_per_file))  # ~30s per chunk

    for hour_idx, file_group in enumerate(detection_groups):
        log.info("─" * 50)
        log.info(
            "Processing hour %d/%d (%d files)",
            hour_idx + 1, len(detection_groups), len(file_group),
        )

        # Process hour in chunks — compress residuals on-the-fly to avoid OOM.
        # Each chunk: infer → median_filter → max_pool(stride) → append pooled.
        # This reduces memory by ~90× (cca_stride).
        from scipy.ndimage import median_filter as _medfilt

        pooled_chunks: list[np.ndarray] = []
        raw_chunks: list[np.ndarray] = []
        reservoir_for_refit: list[np.ndarray] = []
        t0_hour: float | None = None
        total_raw_samples = 0
        stride = config.cca_stride

        for ci in range(0, len(file_group), chunk_size):
            chunk_files = file_group[ci : ci + chunk_size]
            chunk_raw, chunk_t0 = load_files(chunk_files, expected_channels=meta.n_channels)
            total_raw_samples += chunk_raw.shape[0]
            if t0_hour is None:
                t0_hour = chunk_t0

            # Keep subsampled raw for visualization (uniform across all chunks)
            step = max(1, chunk_raw.shape[0] // 200)
            raw_chunks.append(chunk_raw[::step])

            # Resample + normalize + infer
            chunk_resampled = resample_temporal(
                chunk_raw, meta.fs, config.target_fs,
            )
            del chunk_raw
            chunk_norm = normalize(chunk_resampled, norm_stats)
            del chunk_resampled
            chunk_res = infer_residual(
                model, chunk_norm, device, config.batch_size,
            )
            del chunk_norm

            # Keep a small sample for threshold refit (avoid storing full residual)
            if chunk_res.shape[0] > 200:
                rng = np.random.default_rng(ci)
                idx = rng.choice(chunk_res.shape[0], size=200, replace=False)
                reservoir_for_refit.append(chunk_res[np.sort(idx)])
            else:
                reservoir_for_refit.append(chunk_res)

            # Compress: median filter → temporal max-pool → append
            res_filt = _medfilt(chunk_res, size=config.median_kernel)
            del chunk_res
            n_full_blocks = res_filt.shape[0] // stride
            if n_full_blocks > 0:
                trimmed = res_filt[: n_full_blocks * stride]
                pooled = trimmed.reshape(n_full_blocks, stride, -1).max(axis=1)
                pooled_chunks.append(pooled)
            del res_filt

        # Concatenate compressed detection grid (~90x smaller than raw residual)
        if pooled_chunks:
            pooled_grid = np.concatenate(pooled_chunks, axis=0)
        else:
            pooled_grid = np.zeros((0, meta.n_channels), dtype=np.float32)
        del pooled_chunks
        duration_s = total_raw_samples / meta.fs

        # Build raw waterfall from all chunks (subsampled)
        raw_sample = np.concatenate(raw_chunks, axis=0)
        del raw_chunks

        # Detect events on pre-compressed grid (skip median+pool inside detect)
        from .detection import detect_events_from_pooled

        labeled = detect_events_from_pooled(pooled_grid, current_tau, config, dx=meta.dx)
        n_display = count_display_events(labeled, config)
        log.info("  Detected %d display events", n_display)
        del pooled_grid

        # Export detection segments as JSON
        segments = _extract_segments(labeled, distances_km, t0_hour, duration_s, config)
        json_path = run_dir / f"detections_hour_{hour_idx + 1:03d}.json"
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as jf:
            json.dump(segments, jf, indent=2)
        log.info("  Exported %d segments to %s", len(segments), json_path.name)

        # Render visualization (use subsampled raw for display)
        out_path = run_dir / f"hour_{hour_idx + 1:03d}.png"
        render_hour(
            raw_data=raw_sample,
            labeled=labeled,
            distances_km=distances_km,
            t0_unix=t0_hour,
            duration_s=duration_s,
            output_path=out_path,
            cable_name=cable_name,
            config=config,
        )
        del raw_sample, labeled

        # Refit threshold using reservoir sample (not full residual)
        refit_data = np.concatenate(reservoir_for_refit, axis=0)
        del reservoir_for_refit
        current_tau = _refit_threshold(refit_data, current_tau, config)
        tau_history.append((t0_hour + duration_s, current_tau))
        del refit_data

    # ── Step 7: Render threshold evolution ──
    if len(tau_history) > 1:
        render_threshold_history(
            tau_history,
            output_path=run_dir / "threshold_history.png",
            cable_name=cable_name,
        )

    # ── Summary ──
    elapsed = time.perf_counter() - t_start_wall
    log.info("=" * 70)
    log.info("HeimDAS complete: %d hours processed in %.1f s", len(detection_groups), elapsed)
    log.info("Output: %s", run_dir)
    log.info("=" * 70)


# ─── Internal Helpers ─────────────────────────────────────────────────────────


def _infer_cable_name(data_dir: Path, meta: CableMetadata) -> str:
    """Infer a cable name from directory path or metadata.

    Args:
        data_dir: Path to data directory.
        meta: Cable metadata.

    Returns:
        Human-readable cable name string.
    """
    name = data_dir.stem
    if name in ("train", "test", "raw"):
        name = data_dir.parent.stem
    return name.replace("_", " ").title()


def _refit_threshold(
    residual: np.ndarray, current_tau: float, config: PipelineConfig
) -> float:
    """Refit GPD threshold from the latest hour's residuals.

    Args:
        residual: This hour's autoencoder residuals.
        current_tau: Current threshold value.
        config: Pipeline configuration.

    Returns:
        Updated threshold τ.
    """
    try:
        new_cal = compute_threshold(residual, config)
        drift = new_cal.tau - current_tau
        log.info(
            "  Threshold refit: τ %.4f → %.4f (Δ=%.4f, %.1f%%)",
            current_tau, new_cal.tau, drift,
            100 * drift / max(current_tau, 1e-8),
        )
        return new_cal.tau
    except RuntimeError as e:
        log.warning("  Threshold refit failed (%s), keeping τ=%.4f", e, current_tau)
        return current_tau


def _extract_segments(
    labeled: np.ndarray,
    distances_km: np.ndarray,
    t0_unix: float,
    duration_s: float,
    config: PipelineConfig,
) -> list[dict]:
    """Extract detection segments as serializable dictionaries.

    Each segment includes bounding box in physical units (time, distance),
    pixel area, and whether it passes the display filter.

    Args:
        labeled: Integer-labeled event array, shape (T_ds, C).
        distances_km: Distance axis in km.
        t0_unix: Start UNIX timestamp.
        duration_s: Total duration in seconds.
        config: Pipeline config.

    Returns:
        List of segment dictionaries.
    """
    from scipy.ndimage import find_objects

    T_ds, C = labeled.shape
    max_id = int(labeled.max())
    if max_id == 0:
        return []

    slices = find_objects(labeled)
    segments = []
    for eid, sl in enumerate(slices, 1):
        if sl is None:
            continue
        area = int((labeled[sl[0], sl[1]] == eid).sum())
        spatial_px = sl[1].stop - sl[1].start

        # Convert pixel coordinates to physical units
        t_start_frac = sl[0].start / T_ds
        t_end_frac = sl[0].stop / T_ds
        t_start_s = t0_unix + t_start_frac * duration_s
        t_end_s = t0_unix + t_end_frac * duration_s

        x_start_km = float(distances_km[sl[1].start])
        x_end_km = float(distances_km[min(sl[1].stop - 1, C - 1)])

        is_display = (
            area >= config.min_display_area
            and spatial_px >= config.min_display_spatial_px
        )

        segments.append({
            "event_id": eid,
            "area_px": area,
            "spatial_extent_px": spatial_px,
            "is_display_event": is_display,
            "time_start_utc": datetime.fromtimestamp(t_start_s, tz=timezone.utc).isoformat(),
            "time_end_utc": datetime.fromtimestamp(t_end_s, tz=timezone.utc).isoformat(),
            "distance_start_km": round(x_start_km, 3),
            "distance_end_km": round(x_end_km, 3),
            "duration_s": round(t_end_s - t_start_s, 2),
            "spatial_extent_km": round(x_end_km - x_start_km, 3),
        })

    return segments
