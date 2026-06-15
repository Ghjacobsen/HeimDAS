"""GPD threshold calibration for HeimDAS.

This module implements the Peaks-Over-Threshold (POT) method with the
Grimshaw Maximum Likelihood Estimator to derive a statistically principled
detection threshold from autoencoder residuals.

The algorithm (Siffer et al., KDD 2017):

1. Set an initial threshold u₀ at quantile q_init of the residual pool.
2. Extract exceedances Y = {x_i - u₀ : x_i > u₀} (the "tail").
3. Fit a Generalised Pareto Distribution (GPD) to Y via Grimshaw MLE.
4. Compute the operational threshold:

       τ = u₀ + (σ/γ) × ((q × n / N_t)^{-γ} - 1)

   where γ, σ are GPD shape/scale, n = total samples, N_t = peak count,
   and q is the desired false-alarm probability.

Before fitting, the calibration block undergoes robustness filtering:
- Edge channels are excluded (configurable margin).
- Dead/broken channels (extreme median residual) are excluded.
- Values above clip_sigma × MAD are clipped to prevent glitches from
  distorting the tail fit.

The Grimshaw procedure restricts root-finding to two bounded intervals
(Siffer Proposition 1), with scipy.stats.genpareto as a robust fallback.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .config import PipelineConfig

log = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Stores GPD calibration output for reproducibility and logging."""

    tau: float  # Operational threshold z_q
    gamma: float  # GPD shape parameter ξ
    sigma: float  # GPD scale parameter σ
    t: float  # Initial threshold u₀
    n_exceedances: int  # Number of peaks above u₀
    n_total: int  # Total samples used
    q: float  # Target false-alarm probability
    log_likelihood: float  # MLE quality metric


# ─── Grimshaw MLE Implementation ─────────────────────────────────────────────


def _gpd_return_level(
    gamma: float, sigma: float, t: float, n_t: int, n_total: int, q: float
) -> float:
    """Compute operational threshold z_q satisfying P(X > z_q) ≈ q.

    Args:
        gamma: GPD shape parameter.
        sigma: GPD scale parameter.
        t: Initial empirical threshold.
        n_t: Number of exceedances.
        n_total: Total observations.
        q: Target tail probability.

    Returns:
        Threshold value z_q.
    """
    if n_t <= 0 or n_total <= 0:
        return float("nan")
    rate = q * n_total / n_t
    if abs(gamma) < 1e-8:
        return t + sigma * (-np.log(rate))
    return t + (sigma / gamma) * (rate ** (-gamma) - 1.0)


def _gpd_log_likelihood(gamma: float, sigma: float, y: np.ndarray) -> float:
    """Log-likelihood of GPD(γ, σ) for exceedances y.

    Args:
        gamma: Shape parameter.
        sigma: Scale parameter.
        y: Exceedance values (must be > 0).

    Returns:
        Log-likelihood value (or -inf if invalid).
    """
    if sigma <= 0:
        return -np.inf
    n = y.size
    if abs(gamma) < 1e-10:
        return float(-n * np.log(sigma) - y.sum() / sigma)
    z = 1.0 + gamma / sigma * y
    if np.any(z <= 0):
        return -np.inf
    return float(-n * np.log(sigma) - (1.0 + 1.0 / gamma) * np.log(z).sum())


def _grimshaw_solve_interval(
    lo: float, hi: float, y: np.ndarray, n_starts: int = 10
) -> list[float]:
    """Find roots of u(x)·v(x) = 1 in [lo, hi] via L-BFGS-B.

    Args:
        lo: Lower bound of search interval.
        hi: Upper bound of search interval.
        y: Exceedance values.
        n_starts: Number of starting points for multi-start optimization.

    Returns:
        List of candidate root values.
    """
    if hi <= lo:
        return []

    def obj(xv: np.ndarray) -> float:
        x = xv[0]
        a = 1.0 + x * y
        if np.any(a <= 0):
            return 1e10
        u = np.mean(1.0 / a)
        v = 1.0 + np.mean(np.log(a))
        return (u * v - 1.0) ** 2

    roots: list[float] = []
    for x0 in np.linspace(lo, hi, n_starts):
        try:
            res = minimize(
                obj,
                x0=np.array([x0]),
                method="L-BFGS-B",
                bounds=[(lo, hi)],
                options={"maxiter": 100, "ftol": 1e-12},
            )
        except Exception:
            continue
        if res.success and obj(res.x) < 1e-6:
            roots.append(float(res.x[0]))

    return sorted(set(round(r, 8) for r in roots))


def _grimshaw_estimate(y: np.ndarray, epsilon: float = 1e-8) -> tuple[float, float, float]:
    """Estimate GPD parameters (γ, σ) via Grimshaw procedure with Siffer intervals.

    Implements the root-finding on two bounded intervals from Siffer (2017)
    Proposition 1, with scipy.stats.genpareto as a robust fallback. Returns
    the candidate with highest log-likelihood.

    Args:
        y: Exceedance values (positive, non-zero).
        epsilon: Small offset to avoid singularities.

    Returns:
        Tuple (gamma, sigma, log_likelihood) for best-fit GPD.
    """
    y = np.asarray(y, dtype=float)
    y = y[y > 0]

    if y.size < 10:
        # Too few peaks — fall back to method-of-moments
        m = float(y.mean()) if y.size else 1.0
        v = float(y.var(ddof=1)) if y.size > 1 else m * m
        sigma_hat = 0.5 * m * (m * m / max(v, 1e-12) + 1.0)
        gamma_hat = 0.5 * (m * m / max(v, 1e-12) - 1.0)
        return gamma_hat, max(sigma_hat, 1e-8), _gpd_log_likelihood(gamma_hat, sigma_hat, y)

    y_min = float(y.min())
    y_max = float(y.max())
    y_bar = float(y.mean())

    # Interval (i): negative-gamma region
    lo_i = -1.0 / y_max + epsilon
    hi_i = -epsilon

    # Interval (ii): positive region (Siffer Prop. 1)
    lo_ii = (2.0 * (y_bar - y_min)) / (y_bar * y_min) if y_bar * y_min > 0 else None
    hi_ii = (2.0 * (y_bar - y_min)) / (y_min**2) if y_min > 0 else None

    candidates: list[tuple[float, float, float]] = []

    # Exponential limit (gamma = 0)
    sigma_exp = y_bar
    candidates.append((0.0, sigma_exp, _gpd_log_likelihood(0.0, sigma_exp, y)))

    # scipy fallback (robust for heavy tails)
    try:
        from scipy.stats import genpareto

        c_scipy, _, sigma_scipy = genpareto.fit(y, floc=0)
        if sigma_scipy > 0:
            ll_scipy = _gpd_log_likelihood(float(c_scipy), float(sigma_scipy), y)
            if np.isfinite(ll_scipy):
                candidates.append((float(c_scipy), float(sigma_scipy), ll_scipy))
    except Exception:
        pass

    # Grimshaw root search on both intervals
    for lo, hi in [(lo_i, hi_i), (lo_ii, hi_ii)]:
        if lo is None or hi is None:
            continue
        for x_star in _grimshaw_solve_interval(lo, hi, y):
            a = 1.0 + x_star * y
            if np.any(a <= 0):
                continue
            v_val = 1.0 + np.mean(np.log(a))
            gamma_star = v_val - 1.0
            if x_star == 0.0:
                continue
            sigma_star = gamma_star / x_star
            if sigma_star <= 0:
                continue
            ll = _gpd_log_likelihood(gamma_star, sigma_star, y)
            if np.isfinite(ll):
                candidates.append((gamma_star, sigma_star, ll))

    # Pick highest log-likelihood
    candidates.sort(key=lambda c: c[2], reverse=True)
    gamma, sigma, ll = candidates[0]

    # Clamp gamma to sensible range
    gamma = float(np.clip(gamma, -0.5, 1.5))
    return gamma, max(sigma, 1e-8), ll


# ─── Public Calibration Function ─────────────────────────────────────────────


def compute_threshold(residual: np.ndarray, config: PipelineConfig) -> CalibrationResult:
    """Calibrate GPD threshold from autoencoder residuals.

    Applies the full robustness pipeline: edge crop → dead-channel
    exclusion → extreme-value clipping → GPD fit → threshold computation.

    Args:
        residual: Reconstruction error array, shape (T, C).
        config: Pipeline configuration with calibration parameters.

    Returns:
        CalibrationResult with threshold τ and all GPD fit parameters.
    """
    n_rows, n_ch = residual.shape
    log.info(
        "Calibrating GPD (q=%.1e, q_init=%.2f) on (%d, %d) residuals",
        config.q, config.q_init, n_rows, n_ch,
    )

    # Subsample if too large
    max_rows = config.max_calibration_samples // n_ch
    if n_rows > max_rows:
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(n_rows, size=max_rows, replace=False))
        calib = residual[idx, :].astype(np.float32)
        log.info("  Subsampled %d/%d rows for GPD fit", max_rows, n_rows)
    else:
        calib = residual.astype(np.float32)

    # Edge crop
    margin = config.edge_crop
    if margin > 0 and n_ch > 2 * margin:
        calib = calib[:, margin : n_ch - margin]
        log.info("  Excluded %d edge channels each side → %d channels", margin, calib.shape[1])

    # Dead channel exclusion
    ch_medians = np.median(calib[: min(500, calib.shape[0]), :], axis=0)
    global_median = np.median(ch_medians)
    outlier_thresh = global_median * config.calibration_outlier_factor
    healthy = ch_medians <= outlier_thresh
    n_excluded = int((~healthy).sum())
    if n_excluded > 0:
        calib = calib[:, healthy]
        log.info(
            "  Excluded %d dead channels (median > %.1f× global)",
            n_excluded, config.calibration_outlier_factor,
        )

    # Clip extremes
    flat_sample = calib[: min(1000, calib.shape[0]), :].ravel()
    clip_med = np.median(flat_sample)
    clip_mad = np.median(np.abs(flat_sample - clip_med))
    clip_val = clip_med + config.calibration_clip_sigma * max(clip_mad, 1e-6)
    n_clipped = int((calib > clip_val).sum())
    if n_clipped > 0:
        calib = np.clip(calib, None, clip_val)
        log.info("  Clipped %d values above %.2f", n_clipped, clip_val)

    # Flatten and fit GPD
    stream = calib.ravel()
    stream = stream[np.isfinite(stream)]

    if stream.size < 100:
        raise RuntimeError("Insufficient valid samples for GPD calibration")

    t = float(np.quantile(stream, config.q_init))
    exceedances = stream[stream > t] - t
    n_t = int(exceedances.size)
    n_total = int(stream.size)

    if n_t < 10:
        raise RuntimeError(
            f"Too few exceedances ({n_t}) above u₀={t:.4f}. "
            "Try lowering q_init or using more calibration data."
        )

    gamma, sigma, ll = _grimshaw_estimate(exceedances)
    tau = _gpd_return_level(gamma, sigma, t, n_t, n_total, config.q)

    log.info(
        "  GPD fit: γ=%.4f, σ=%.4f, u₀=%.4f → τ=%.4f (T=%d)",
        gamma, sigma, t, tau, int(1 / config.q),
    )

    return CalibrationResult(
        tau=tau,
        gamma=gamma,
        sigma=sigma,
        t=t,
        n_exceedances=n_t,
        n_total=n_total,
        q=config.q,
        log_likelihood=ll,
    )
