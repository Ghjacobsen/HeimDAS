"""Two-panel visualization for HeimDAS.

This module renders the standard HeimDAS output: a two-panel figure showing
the raw DAS waterfall (Panel A) and the semantic detection map (Panel B).

Panel A — "Raw DAS Waterfall":
    Displays the raw (un-normalized) DAS data as a time-vs-distance heatmap
    using the jet colormap. The color scale is clipped at the 98th percentile
    to handle outliers gracefully.

Panel B — "HeimDAS Detection":
    Displays the labeled event map. Events passing the display filter
    (min_display_area AND min_display_spatial_px) are rendered with distinct
    colours from the Tab20 palette. Smaller events are shown in gray.
    Background (no detection) is black.

Figure title format:
    "{Cable Name} — {start_datetime UTC} to {end_datetime UTC}"

Axes:
    X-axis: Distance along cable (km), computed as channel_index × dx
    Y-axis: Time (UTC, HH:MM labels)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scienceplots  # noqa: E402, F401
from scipy.ndimage import find_objects  # noqa: E402

from .config import PipelineConfig  # noqa: E402

log = logging.getLogger(__name__)

# Apply SciencePlots style globally for all HeimDAS figures
plt.style.use(["science", "nature"])


def render_hour(
    raw_data: np.ndarray,
    labeled: np.ndarray,
    distances_km: np.ndarray,
    t0_unix: float,
    duration_s: float,
    output_path: Path,
    cable_name: str,
    config: PipelineConfig,
) -> None:
    """Render a two-panel PNG for one hour of DAS data.

    Args:
        raw_data: Raw (un-normalized) DAS data, shape (T, C).
        labeled: Detection labels from detect_events(), shape (T_ds, C).
        distances_km: Distance axis in km (channel_index × dx / 1000).
        t0_unix: UNIX timestamp of the first sample.
        duration_s: Total duration in seconds.
        output_path: Where to save the PNG.
        cable_name: Cable identifier for the figure title.
        config: Pipeline configuration (for display thresholds).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Time range for title
    t_start = datetime.fromtimestamp(t0_unix, tz=timezone.utc)
    t_end = datetime.fromtimestamp(t0_unix + duration_s, tz=timezone.utc)
    title = (
        f"{cable_name} — {t_start.strftime('%Y-%m-%d %H:%M:%S')}"
        f" to {t_end.strftime('%H:%M:%S')} UTC"
    )

    # Axis extents: [x_min, x_max, y_bottom, y_top]
    t_max_min = duration_s / 60.0
    extent = [distances_km[0], distances_km[-1], t_max_min, 0]

    # Subsample raw data for display (target ~2000 pixels in each dim)
    raw_display = _subsample(raw_data, target=2000)

    # Build RGB detection map
    rgb_map = _build_rgb(labeled, config)

    # Create figure — wide format for DAS data
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(8, 6))
    fig.suptitle(title, fontweight="bold")

    # Panel A: Raw DAS Waterfall
    vmax = float(np.percentile(np.abs(raw_display), 98))
    ax_a.imshow(
        np.abs(raw_display),
        aspect="auto",
        cmap="jet",
        interpolation="bilinear",
        extent=extent,
        vmin=0,
        vmax=vmax,
    )
    ax_a.set_title("Raw DAS Waterfall")
    ax_a.set_xlabel("Distance (km)")
    ax_a.set_ylabel("Time (min)")
    _add_time_ticks(ax_a, t0_unix, duration_s)

    # Panel B: HeimDAS Detection
    ax_b.imshow(
        rgb_map,
        aspect="auto",
        interpolation="nearest",
        extent=extent,
    )
    ax_b.set_title("HeimDAS Detection")
    ax_b.set_xlabel("Distance (km)")
    ax_b.set_ylabel("Time (min)")
    _add_time_ticks(ax_b, t0_unix, duration_s)

    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved visualization: %s", output_path)


def render_threshold_history(
    tau_history: list[tuple[int, float]],
    output_path: Path,
    cable_name: str,
) -> None:
    """Render a staircase plot of threshold τ over processing hours.

    Args:
        tau_history: List of (hour_index, tau_value) tuples.
        output_path: Where to save the PNG.
        cable_name: Cable name for the figure title.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    hours = [h for h, _ in tau_history]
    taus = [t for _, t in tau_history]

    fig, ax = plt.subplots(figsize=(6, 3.5))

    ax.step(hours, taus, where="post", linewidth=2)
    ax.scatter(hours, taus, s=30, zorder=5)

    ax.set_xlabel("Hour")
    ax.set_ylabel(r"Threshold $\tau$")
    ax.set_title(f"{cable_name} — Threshold Evolution")
    ax.grid(True, alpha=0.3, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved threshold plot: %s", output_path)


# ─── Internal Helpers ─────────────────────────────────────────────────────────


def _subsample(data: np.ndarray, target: int = 2000) -> np.ndarray:
    """Subsample array to approximately target pixels in each dimension.

    Args:
        data: 2D array (T, C).
        target: Target size for each dimension.

    Returns:
        Subsampled array.
    """
    T, C = data.shape
    t_step = max(1, T // target)
    c_step = max(1, C // target)
    return data[::t_step, ::c_step]


def _build_rgb(labeled: np.ndarray, config: PipelineConfig) -> np.ndarray:
    """Build RGB array from labeled event map.

    Display-eligible events get distinct Tab20 colours. Small events
    are rendered in gray. Background is black.

    Args:
        labeled: Integer-labeled event array.
        config: Pipeline config for display thresholds.

    Returns:
        RGB array shape (H, W, 3), float32 in [0, 1].
    """
    tab20 = plt.cm.tab20(np.linspace(0, 1, 20))[:, :3]

    max_id = int(labeled.max())
    if max_id == 0:
        return np.zeros((*labeled.shape, 3), dtype=np.float32)

    # Determine which events pass display filter
    slices = find_objects(labeled)
    display_ids: set[int] = set()
    for eid, sl in enumerate(slices, 1):
        if sl is None:
            continue
        area = int((labeled[sl[0], sl[1]] == eid).sum())
        spatial_px = sl[1].stop - sl[1].start
        if area >= config.min_display_area and spatial_px >= config.min_display_spatial_px:
            display_ids.add(eid)

    # Build colour LUT
    lut = np.zeros((max_id + 1, 3), dtype=np.float32)
    for eid in range(1, max_id + 1):
        if eid in display_ids:
            lut[eid] = tab20[eid % 20]
        else:
            lut[eid] = [0.3, 0.3, 0.3]  # gray for small events

    return lut[labeled]


def _add_time_ticks(ax: plt.Axes, t0_unix: float, duration_s: float) -> None:
    """Replace y-axis minute labels with UTC HH:MM timestamps.

    Args:
        ax: Matplotlib axes with time (min) on Y-axis.
        t0_unix: Start time as UNIX timestamp.
        duration_s: Total duration in seconds.
    """
    n_ticks = min(7, max(3, int(duration_s / 600)))  # ~1 tick per 10 min
    tick_mins = np.linspace(0, duration_s / 60.0, n_ticks)
    labels = []
    for m in tick_mins:
        t_utc = datetime.fromtimestamp(t0_unix + m * 60, tz=timezone.utc)
        labels.append(t_utc.strftime("%H:%M"))
    ax.set_yticks(tick_mins)
    ax.set_yticklabels(labels)
