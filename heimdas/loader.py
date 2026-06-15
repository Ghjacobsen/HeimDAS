"""HDF5 data discovery, sorting, and loading for HeimDAS.

This module handles the first stage of the pipeline: finding all HDF5 files
in a user-supplied directory, sorting them chronologically (by filename
timestamp), reading cable metadata (sample rate, channel spacing, channel
count), and loading contiguous time windows by concatenating consecutive
files.

DAS interrogators typically write one HDF5 file per acquisition burst
(e.g. 10 seconds). HeimDAS stitches these into continuous hour-long
segments for processing.

Expected HDF5 structure (per file):
    /data        — int16 or float32 array, shape (n_samples, n_channels)
    /header/time — UNIX timestamp (float) of first sample
    /header/dx   — channel spacing in metres
    /header/dt   — sample interval in seconds (1/fs)

Files are sorted lexicographically by stem, which works for the standard
DAS naming convention (e.g. dphi_HHMMSS.hdf5 or HHMMSS.hdf5).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class CableMetadata:
    """Physical properties of the DAS cable, inferred from file headers."""

    fs: float  # Native sample rate (Hz)
    dx: float  # Effective channel spacing (metres), accounts for channel stride
    n_channels: int  # Number of spatial channels
    t0_unix: float  # UNIX timestamp of first sample in first file


def discover_files(data_dir: Path) -> list[Path]:
    """Find and chronologically sort all HDF5 files in a directory.

    Args:
        data_dir: Path to directory containing .hdf5 files.

    Returns:
        Sorted list of HDF5 file paths.

    Raises:
        FileNotFoundError: If no HDF5 files are found.
    """
    files = sorted(data_dir.glob("*.hdf5"))
    if not files:
        # Try subdirectories (some datasets have train/test splits)
        files = sorted(data_dir.rglob("*.hdf5"))
    if not files:
        raise FileNotFoundError(f"No .hdf5 files found in {data_dir}")
    log.info("Discovered %d HDF5 files in %s", len(files), data_dir)
    return files


def read_metadata(file_path: Path) -> CableMetadata:
    """Extract cable metadata from a single HDF5 file header.

    Args:
        file_path: Path to any HDF5 file from the dataset.

    Returns:
        CableMetadata with sample rate, spacing, channel count, start time.
    """
    with h5py.File(file_path, "r") as f:
        data_shape = f["data"].shape
        dt = float(f["header"]["dt"][()])
        dx = float(f["header"]["dx"][()])
        t0 = float(f["header"]["time"][()])

        # Determine effective channel spacing from header/channels array.
        # Some DAS systems subsample spatially (e.g. every 4th channel),
        # so the true spacing is stride * dx.
        channel_stride = 1
        if "channels" in f["header"]:
            ch = f["header"]["channels"][()]
            if len(ch) > 1:
                channel_stride = int(ch[1] - ch[0])

    fs = 1.0 / dt
    effective_dx = dx * channel_stride
    n_channels = data_shape[1]

    log.info(
        "Cable metadata: fs=%.0f Hz, dx=%.3f m (stride=%d, raw_dx=%.3f), channels=%d, t0=%s",
        fs, effective_dx, channel_stride, dx, n_channels, t0,
    )
    return CableMetadata(fs=fs, dx=effective_dx, n_channels=n_channels, t0_unix=t0)


def load_files(
    file_paths: list[Path], expected_channels: int | None = None,
) -> tuple[np.ndarray, float]:
    """Load and concatenate a list of HDF5 files into a single time-series array.

    Files are concatenated along the time axis (axis=0). The raw integer
    data is converted to float32. If a 'dataScale' attribute exists in the
    header, it is applied. Files with mismatched channel count are skipped.

    Args:
        file_paths: Ordered list of HDF5 files to load.
        expected_channels: If provided, only load files with this channel count.
            This ensures consistency across chunked loading.

    Returns:
        Tuple of (data array shape (N, C), t0_unix of first file).
    """
    chunks: list[np.ndarray] = []
    t0_unix: float | None = None

    for fp in file_paths:
        try:
            with h5py.File(fp, "r") as f:
                raw = f["data"][:].astype(np.float32)
                # Skip files with different channel count
                if expected_channels is None:
                    expected_channels = raw.shape[1]
                elif raw.shape[1] != expected_channels:
                    log.debug(
                        "Skipping %s: %d channels (expected %d)",
                        fp.name, raw.shape[1], expected_channels,
                    )
                    continue
                if t0_unix is None:
                    t0_unix = float(f["header"]["time"][()])
                # Apply data scale if present
                if "dataScale" in f["header"]:
                    scale = float(f["header"]["dataScale"][()])
                    raw *= scale
            chunks.append(raw)
        except (OSError, KeyError) as e:
            log.warning("Skipping corrupt file %s: %s", fp.name, e)
            continue

    if not chunks:
        raise RuntimeError("No valid HDF5 files could be loaded")

    data = np.concatenate(chunks, axis=0)
    del chunks
    log.info("Loaded %d samples × %d channels", *data.shape)
    return data, t0_unix  # type: ignore[return-value]


def group_files_by_hour(
    file_paths: list[Path],
    fs: float,
    samples_per_file: int | None = None,
) -> list[list[Path]]:
    """Partition files into hourly groups based on cumulative duration.

    If sample count per file is unknown, it is read from the first file.
    Files are grouped such that each group spans approximately 1 hour of
    data.

    Args:
        file_paths: Chronologically sorted list of all HDF5 files.
        fs: Native sample rate (Hz).
        samples_per_file: Number of time samples per file (if uniform).

    Returns:
        List of file groups, each group covering ~1 hour.
    """
    if samples_per_file is None:
        with h5py.File(file_paths[0], "r") as f:
            samples_per_file = f["data"].shape[0]

    seconds_per_file = samples_per_file / fs
    files_per_hour = max(1, int(round(3600.0 / seconds_per_file)))

    groups: list[list[Path]] = []
    for i in range(0, len(file_paths), files_per_hour):
        groups.append(file_paths[i : i + files_per_hour])

    log.info(
        "Grouped %d files into %d hourly segments (~%d files/hour, %.1f s/file)",
        len(file_paths), len(groups), files_per_hour, seconds_per_file,
    )
    return groups
