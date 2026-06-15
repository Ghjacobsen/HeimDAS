"""Event detection and segmentation for HeimDAS.

This module implements the full detection pipeline that transforms
autoencoder residuals into a labeled event map. The pipeline consists
of six sequential stages:

1. Median filtering — Suppresses single-pixel noise spikes.
2. Temporal max-pooling — Compresses time dimension by cca_stride
   (e.g. 90 samples → 1 pixel), making CCA computationally tractable.
3. Hysteresis thresholding — Two-level threshold: "seed" pixels must
   exceed τ (strict), but they expand to include all connected pixels
   above τ × hysteresis_ratio (extend). This captures the weak tails
   of strong events without generating false alarms from noise.
4. Morphological filtering — Binary opening removes thin filaments;
   binary closing fills internal gaps within event blobs.
5. Connected Component Analysis (CCA) — Labels contiguous regions.
   Components smaller than k_min pixels are discarded.
6. Proximity merging — Nearby components (within gap_t seconds AND
   gap_x metres) are merged via Union-Find. This reconnects events
   that were fragmented by brief signal dropouts or narrow quiet gaps.

After merging, a display filter determines which events are rendered
in colour vs. gray on the visualization (purely cosmetic).
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.ndimage import binary_closing, binary_opening, find_objects, label, median_filter

from .config import PipelineConfig

log = logging.getLogger(__name__)


def detect_events(
    residual: np.ndarray,
    tau: float,
    config: PipelineConfig,
    dx: float = 1.0,
) -> np.ndarray:
    """Run the full detection pipeline on autoencoder residuals.

    Args:
        residual: Reconstruction error array, shape (T, C), dtype float32.
        tau: Detection threshold from GPD calibration.
        config: Pipeline configuration parameters.
        dx: Channel spacing in metres (for gap_x conversion).

    Returns:
        Labeled array of shape (T_compressed, C) with integer event IDs.
        Background = 0, each event has a unique positive integer.
    """
    T, C = residual.shape
    stride = config.cca_stride
    n_t_ds = T // stride

    log.info("Detection: (%d, %d) → (%d, %d) grid, τ=%.3f", T, C, n_t_ds, C, tau)

    # Stage 1: Median filter
    res_filt = median_filter(residual, size=config.median_kernel)

    # Stage 2: Temporal max-pooling + thresholding
    tau_extend = tau * config.hysteresis_ratio
    mask_high = np.zeros((n_t_ds, C), dtype=bool)
    mask_low = np.zeros((n_t_ds, C), dtype=bool)

    for i in range(n_t_ds):
        chunk = res_filt[i * stride : (i + 1) * stride, :]
        chunk_max = chunk.max(axis=0)
        mask_high[i, :] = chunk_max > tau
        mask_low[i, :] = chunk_max > tau_extend

    log.info(
        "  Seeds (>τ): %.3f%%, Extended (>τ×%.1f): %.3f%%",
        mask_high.mean() * 100, config.hysteresis_ratio, mask_low.mean() * 100,
    )

    # Stage 3: Hysteresis thresholding
    mask_hyst = _hysteresis(mask_high, mask_low)
    log.info("  After hysteresis: %.3f%%", mask_hyst.mean() * 100)

    # Stage 4: Morphological filtering
    s_open = np.ones(config.open_kernel, dtype=bool)
    s_close = np.ones(config.close_kernel, dtype=bool)
    mask_morph = binary_closing(binary_opening(mask_hyst, structure=s_open), structure=s_close)
    log.info("  After morphology: %.3f%%", mask_morph.mean() * 100)

    # Stage 5: CCA + area filter
    labeled_raw, n_raw = label(mask_morph)
    log.info("  Raw components: %d", n_raw)

    labeled_kept, n_kept = _area_filter(labeled_raw, config.k_min)
    log.info("  After area filter (k≥%d): %d", config.k_min, n_kept)

    # Stage 6: Proximity merging
    gap_t_px = max(1, int(round(config.gap_t_seconds * (config.target_fs / stride))))
    gap_x_px = max(1, int(round(config.gap_x_metres / dx)))
    labeled_merged = _merge_components(labeled_kept, gap_t_px, gap_x_px)
    n_merged = int(labeled_merged.max())
    log.info("  After merge (gap_t=%dpx, gap_x=%dpx): %d events", gap_t_px, gap_x_px, n_merged)

    return labeled_merged


def detect_events_from_pooled(
    pooled: np.ndarray,
    tau: float,
    config: PipelineConfig,
    dx: float = 1.0,
) -> np.ndarray:
    """Run detection on pre-compressed (already median-filtered + max-pooled) data.

    This is the memory-efficient variant: median filter and temporal max-pooling
    are performed per-chunk during inference to avoid materializing the full
    residual array. This function receives the already-pooled grid and applies
    hysteresis thresholding, morphology, CCA, area filtering, and proximity merging.

    Args:
        pooled: Max-pooled residual grid, shape (T_pooled, C), dtype float32.
        tau: Detection threshold from GPD calibration.
        config: Pipeline configuration parameters.
        dx: Channel spacing in metres (for gap_x conversion).

    Returns:
        Labeled array of shape (T_pooled, C) with integer event IDs.
    """
    n_t_ds, C = pooled.shape
    stride = config.cca_stride

    log.info("Detection (pre-pooled): (%d, %d) grid, τ=%.3f", n_t_ds, C, tau)

    # Hysteresis thresholding on pooled max values
    tau_extend = tau * config.hysteresis_ratio
    mask_high = pooled > tau
    mask_low = pooled > tau_extend

    log.info(
        "  Seeds (>τ): %.3f%%, Extended (>τ×%.1f): %.3f%%",
        mask_high.mean() * 100, config.hysteresis_ratio, mask_low.mean() * 100,
    )

    # Hysteresis
    mask_hyst = _hysteresis(mask_high, mask_low)
    log.info("  After hysteresis: %.3f%%", mask_hyst.mean() * 100)

    # Morphological filtering
    s_open = np.ones(config.open_kernel, dtype=bool)
    s_close = np.ones(config.close_kernel, dtype=bool)
    mask_morph = binary_closing(binary_opening(mask_hyst, structure=s_open), structure=s_close)
    log.info("  After morphology: %.3f%%", mask_morph.mean() * 100)

    # CCA + area filter
    labeled_raw, n_raw = label(mask_morph)
    log.info("  Raw components: %d", n_raw)

    labeled_kept, n_kept = _area_filter(labeled_raw, config.k_min)
    log.info("  After area filter (k≥%d): %d", config.k_min, n_kept)

    # Proximity merging
    gap_t_px = max(1, int(round(config.gap_t_seconds * (config.target_fs / stride))))
    gap_x_px = max(1, int(round(config.gap_x_metres / dx)))
    labeled_merged = _merge_components(labeled_kept, gap_t_px, gap_x_px)
    n_merged = int(labeled_merged.max())
    log.info("  After merge (gap_t=%dpx, gap_x=%dpx): %d events", gap_t_px, gap_x_px, n_merged)

    return labeled_merged


def count_display_events(labeled: np.ndarray, config: PipelineConfig) -> int:
    """Count events passing the display filter.

    Args:
        labeled: Labeled event array from detect_events().
        config: Pipeline configuration.

    Returns:
        Number of events exceeding both area and spatial thresholds.
    """
    slices = find_objects(labeled)
    count = 0
    for eid, sl in enumerate(slices, 1):
        if sl is None:
            continue
        area = int((labeled[sl[0], sl[1]] == eid).sum())
        spatial_px = sl[1].stop - sl[1].start
        if area >= config.min_display_area and spatial_px >= config.min_display_spatial_px:
            count += 1
    return count


# ─── Internal Helpers ─────────────────────────────────────────────────────────


def _hysteresis(mask_high: np.ndarray, mask_low: np.ndarray) -> np.ndarray:
    """Keep components in mask_low that overlap mask_high (seed-based expansion).

    Args:
        mask_high: Binary mask of strict-threshold seeds.
        mask_low: Binary mask of extend-threshold candidates.

    Returns:
        Binary mask combining seed regions with their connected extensions.
    """
    labeled_low, n = label(mask_low)
    if n == 0:
        return mask_low

    # Find which low-threshold components contain at least one seed
    seed_ids = set(np.unique(labeled_low[mask_high]))
    seed_ids.discard(0)

    if not seed_ids:
        return np.zeros_like(mask_low, dtype=bool)

    return np.isin(labeled_low, list(seed_ids))


def _area_filter(labeled: np.ndarray, k_min: int) -> tuple[np.ndarray, int]:
    """Remove components smaller than k_min pixels.

    Args:
        labeled: CCA-labeled array.
        k_min: Minimum area threshold.

    Returns:
        Tuple of (relabeled array, number of kept components).
    """
    slices = find_objects(labeled)
    keep_ids = []
    for eid, sl in enumerate(slices, 1):
        if sl is None:
            continue
        area = int((labeled[sl[0], sl[1]] == eid).sum())
        if area >= k_min:
            keep_ids.append(eid)

    if not keep_ids:
        return np.zeros_like(labeled, dtype=np.int32), 0

    keep_mask = np.isin(labeled, keep_ids)
    relabeled, n = label(keep_mask)
    return relabeled.astype(np.int32), n


def _merge_components(labeled: np.ndarray, gap_t: int, gap_x: int) -> np.ndarray:
    """Merge nearby components via Union-Find on bounding-box proximity.

    Two components are merged if their bounding boxes are within gap_t
    pixels temporally AND gap_x pixels spatially.

    Args:
        labeled: CCA-labeled array (int, background=0).
        gap_t: Maximum temporal gap (pixels) for merging.
        gap_x: Maximum spatial gap (pixels) for merging.

    Returns:
        Relabeled array with merged components numbered contiguously.
    """
    n = int(labeled.max())
    if n <= 1:
        return labeled

    slices = find_objects(labeled)

    # Union-Find data structure
    parent = list(range(n + 1))
    rank = [0] * (n + 1)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri == rj:
            return
        if rank[ri] < rank[rj]:
            ri, rj = rj, ri
        parent[rj] = ri
        if rank[ri] == rank[rj]:
            rank[ri] += 1

    # Extract bounding boxes
    boxes: list[tuple[int, int, int, int] | None] = []
    for eid in range(1, n + 1):
        sl = slices[eid - 1]
        if sl is None:
            boxes.append(None)
        else:
            boxes.append((sl[0].start, sl[0].stop, sl[1].start, sl[1].stop))

    # Sort by temporal start for early termination
    sorted_ids = sorted(range(1, n + 1), key=lambda i: boxes[i - 1][0] if boxes[i - 1] else 1e9)

    for idx_i in range(len(sorted_ids)):
        i = sorted_ids[idx_i]
        bi = boxes[i - 1]
        if bi is None:
            continue
        t0_i, t1_i, x0_i, x1_i = bi

        for idx_j in range(idx_i + 1, len(sorted_ids)):
            j = sorted_ids[idx_j]
            bj = boxes[j - 1]
            if bj is None:
                continue
            t0_j, t1_j, x0_j, x1_j = bj

            # Early termination: j starts too far after i ends
            if t0_j > t1_i + gap_t:
                break

            # Gap computation (distance between bounding boxes, not overlap)
            t_gap = max(0, max(t0_i, t0_j) - min(t1_i, t1_j))
            x_gap = max(0, max(x0_i, x0_j) - min(x1_i, x1_j))

            if t_gap <= gap_t and x_gap <= gap_x:
                union(i, j)

    # Relabel via LUT
    lut = np.zeros(n + 1, dtype=np.int32)
    root_map: dict[int, int] = {}
    counter = 0
    for eid in range(1, n + 1):
        if boxes[eid - 1] is None:
            continue
        root = find(eid)
        if root not in root_map:
            counter += 1
            root_map[root] = counter
        lut[eid] = root_map[root]

    return lut[labeled]
