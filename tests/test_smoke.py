"""Smoke tests for HeimDAS package."""

from __future__ import annotations

import numpy as np
import torch

from heimdas import __version__
from heimdas.config import PipelineConfig
from heimdas.model import ConvAutoencoder, load_model, infer_residual
from heimdas.calibration import compute_threshold
from heimdas.detection import detect_events


def test_version():
    """Package version is a non-empty string."""
    assert isinstance(__version__, str)
    assert len(__version__) > 0


def test_model_loads():
    """Pretrained weights load without error."""
    model = load_model(device="cpu")
    assert isinstance(model, ConvAutoencoder)
    assert not model.training


def test_forward_pass():
    """Model produces output matching input shape."""
    model = load_model(device="cpu")
    x = torch.randn(1, 1, 128, 128)
    with torch.no_grad():
        y = model(x)
    assert y.shape == x.shape


def test_infer_residual():
    """Residual has same shape as input and is non-negative."""
    model = load_model(device="cpu")
    data = np.random.randn(256, 256).astype(np.float32)
    residual = infer_residual(model, data, device="cpu", batch_size=16)
    assert residual.shape == data.shape
    assert residual.dtype == np.float32
    assert (residual >= 0).all()


def test_calibration():
    """GPD threshold fitting returns a positive threshold."""
    rng = np.random.default_rng(42)
    # Simulate a (T, C) residual array with exponential-like values
    residuals = rng.exponential(scale=1.0, size=(1000, 100)).astype(np.float32)
    config = PipelineConfig(q=1e-3, q_init=0.85, edge_crop=0)
    result = compute_threshold(residuals, config)
    assert result.tau > 0
    assert np.isfinite(result.tau)


def test_detection_empty():
    """Detection on a zero grid returns no events."""
    grid = np.zeros((900, 100), dtype=np.float32)
    config = PipelineConfig(cca_stride=90)
    labels = detect_events(grid, tau=1.0, config=config, dx=1.0)
    assert labels.max() == 0


def test_detection_finds_blob():
    """Detection finds a clear synthetic event."""
    grid = np.zeros((900, 200), dtype=np.float32)
    # Insert a bright blob well above threshold
    grid[100:400, 80:120] = 10.0
    config = PipelineConfig(cca_stride=10, k_min=5)
    labels = detect_events(grid, tau=1.0, config=config, dx=1.0)
    assert labels.max() >= 1
