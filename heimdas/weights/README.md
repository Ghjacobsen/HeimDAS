# HeimDAS Model Weights

This directory contains pretrained weights for the HeimDAS convolutional
autoencoder.

## Provenance

- **Model**: ConvAutoencoder (3-layer encoder/decoder, latent_dim=128)
- **Training data**: Storebælt DAS cable, Config B (7,250 channels, 400 Hz)
- **Normalization**: Robust Z-score (per-channel median/MAD)
- **Training**: Optuna hyperparameter search → 100 epochs, L1 loss
- **Source**: `DAS_Project/models/sweeps/sweep_41_zscore_global/best_cae.pth`

## File

- `best_cae.pth` — PyTorch state dict (ConvAutoencoder, ~4.2 MB)
