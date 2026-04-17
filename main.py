import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

import numpy as np
import torch

import config
from dataset import get_dataloaders
from models import AutoEncoder, ResnetFeatures
from train import train, plot_learning_curves
from evaluate import (
    compute_threshold,
    predict,
    plot_roc_and_confusion,
    visualize_heatmaps,
)


def main():
    # ------------------------------------------------------------------ #
    # 1. Data
    # ------------------------------------------------------------------ #
    train_loader, val_loader, calib_loader = get_dataloaders()

    # ------------------------------------------------------------------ #
    # 2. Models
    # ------------------------------------------------------------------ #
    model = AutoEncoder(
        in_channels=config.IN_CHANNELS,
        latent_dim=config.LATENT_DIM,
        is_bn=config.IS_BN,
    ).to(config.DEVICE)

    feat_extractor = ResnetFeatures(finetune_layers=config.FINETUNE_LAYERS).to(config.DEVICE)

    # ------------------------------------------------------------------ #
    # 3. Train
    # ------------------------------------------------------------------ #
    train_losses, val_losses = train(model, feat_extractor, train_loader, val_loader)
    plot_learning_curves(train_losses, val_losses)

    # ------------------------------------------------------------------ #
    # 4. Load best checkpoint
    # ------------------------------------------------------------------ #
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
    model.eval()
    if config.FINETUNE_LAYERS and Path(config.BACKBONE_SAVE_PATH).exists():
        feat_extractor.load_state_dict(
            torch.load(config.BACKBONE_SAVE_PATH, map_location=config.DEVICE)
        )
    feat_extractor.eval()

    # ------------------------------------------------------------------ #
    # 5. Calibrate threshold on training data
    # ------------------------------------------------------------------ #
    threshold, recon_errors = compute_threshold(model, feat_extractor, calib_loader)
    print(f'Anomaly threshold (mean + 3σ): {threshold:.6f}')

    # ------------------------------------------------------------------ #
    # 6. Evaluate on test set
    # ------------------------------------------------------------------ #
    y_true, _, y_score = predict(model, feat_extractor, threshold)
    plot_roc_and_confusion(y_true, y_score, deployed_threshold=threshold)
    np.save('threshold.npy', np.array(threshold))
    print(f'Threshold saved to threshold.npy')

    # ------------------------------------------------------------------ #
    # 7. Visualize heatmaps for abnormal images
    # ------------------------------------------------------------------ #
    visualize_heatmaps(model, feat_extractor, threshold, recon_errors)


if __name__ == '__main__':
    main()
