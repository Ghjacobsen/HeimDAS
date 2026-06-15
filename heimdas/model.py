"""Convolutional Autoencoder model and inference for HeimDAS.

This module defines the ConvAutoencoder architecture and provides helper
functions to load pretrained weights and run patch-based inference on
full DAS data arrays.

Architecture:
    Encoder:  3 × Conv2d(stride=2) with ReLU — compresses 128×128 → 16×16
    Bottleneck: Flatten → Linear(16384 → 128) → ReLU → Linear(128 → 16384) → ReLU
    Decoder:  3 × ConvTranspose2d(stride=2) with ReLU (final layer: linear output)

The linear (no activation) final layer is essential because inputs are
Z-scored and can take negative values.

Residual computation:
    residual = |input - reconstruction|   (L1 / absolute error)

Inference is performed patch-by-patch on a non-overlapping 128×128 grid.
Patches at the edges are handled by zero-padding the input to the nearest
multiple of 128.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

log = logging.getLogger(__name__)

# Path to bundled pretrained weights (relative to this file)
_WEIGHTS_DIR = Path(__file__).parent / "weights"
_WEIGHTS_PATH = _WEIGHTS_DIR / "best_cae.pth"

# Model hyperparameters (fixed — match pretrained weights)
_LATENT_DIM = 224
_PATCH_SIZE = 128


class ConvAutoencoder(nn.Module):
    """Convolutional Autoencoder for DAS anomaly detection.

    Compresses 128×128 DAS patches into a 128-dimensional latent vector
    and reconstructs them. Anomalies produce high reconstruction error
    because the model has only learned to reconstruct normal patterns.
    """

    def __init__(self, latent_dim: int = _LATENT_DIM, patch_size: int = _PATCH_SIZE):
        """Initialize the autoencoder.

        Args:
            latent_dim: Bottleneck dimensionality (default 128).
            patch_size: Spatial side length of input patches (default 128).
        """
        super().__init__()
        self.patch_size = patch_size

        # Encoder: 3 conv layers, each halves spatial dims
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Flatten dim: 64 channels × (patch_size/8)²
        self.flatten_dim = 64 * (patch_size // 8) * (patch_size // 8)

        # Bottleneck: compress to latent_dim and back
        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, self.flatten_dim),
            nn.ReLU(),
        )

        # Decoder input reshaping
        self.decoder_input = nn.Unflatten(1, (64, patch_size // 8, patch_size // 8))

        # Decoder: 3 transposed conv layers (mirror of encoder)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # Final layer: LINEAR output (no activation) — inputs can be negative
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: encode → bottleneck → decode.

        Args:
            x: Input tensor shape (batch, 1, 128, 128).

        Returns:
            Reconstruction tensor, same shape as input.
        """
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder_input(x)
        x = self.decoder(x)
        return x


def load_model(device: torch.device | str = "cpu") -> ConvAutoencoder:
    """Load the pretrained autoencoder with bundled weights.

    Args:
        device: Torch device to load model onto ('cpu', 'cuda', etc.).

    Returns:
        Pretrained ConvAutoencoder in eval mode.

    Raises:
        FileNotFoundError: If bundled weights are not found.
    """
    if not _WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"Pretrained weights not found at {_WEIGHTS_PATH}. "
            "Ensure best_cae.pth is in heimdas/weights/."
        )

    model = ConvAutoencoder(latent_dim=_LATENT_DIM, patch_size=_PATCH_SIZE)
    state = torch.load(_WEIGHTS_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval().to(device)
    log.info("Model loaded (latent_dim=%d, device=%s)", _LATENT_DIM, device)
    return model


def infer_residual(
    model: ConvAutoencoder,
    data: np.ndarray,
    device: torch.device | str = "cpu",
    batch_size: int = 64,
) -> np.ndarray:
    """Run patch-based autoencoder inference and compute L1 residual.

    The input array is zero-padded to the nearest multiple of patch_size
    in both dimensions, then processed as non-overlapping 128×128 patches.
    The residual |input - reconstruction| is returned (trimmed to original
    shape).

    Args:
        model: Pretrained ConvAutoencoder in eval mode.
        data: Normalized input array, shape (n_time, n_channels), float32.
        device: Torch device for inference.
        batch_size: Number of patches per forward pass.

    Returns:
        Residual array, same shape as input, dtype float32.
    """
    p = _PATCH_SIZE
    T, C = data.shape

    # Pad to nearest multiple of patch_size
    pad_t = (p - T % p) % p
    pad_c = (p - C % p) % p
    if pad_t > 0 or pad_c > 0:
        data = np.pad(data, ((0, pad_t), (0, pad_c)), mode="constant", constant_values=0.0)

    T_pad, C_pad = data.shape
    residual = np.zeros((T_pad, C_pad), dtype=np.float32)

    # Generate patch coordinates
    coords = [
        (t, c)
        for t in range(0, T_pad, p)
        for c in range(0, C_pad, p)
    ]

    # Process in batches
    device = torch.device(device) if isinstance(device, str) else device
    n_batches = (len(coords) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(coords), batch_size), total=n_batches, desc="Inference"):
        batch_coords = coords[i : i + batch_size]
        patches = np.stack([data[t : t + p, c : c + p] for t, c in batch_coords])
        patches = patches[:, np.newaxis, :, :]  # (B, 1, 128, 128)

        x = torch.from_numpy(patches).to(device)
        with torch.no_grad():
            recon = model(x)
        diff = torch.abs(x - recon).cpu().numpy()[:, 0]

        for (t, c), d in zip(batch_coords, diff):
            residual[t : t + p, c : c + p] = d

    # Trim padding
    return residual[:T, :C]
