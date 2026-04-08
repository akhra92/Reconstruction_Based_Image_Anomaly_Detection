import warnings
warnings.filterwarnings('ignore')

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
    train_loader, val_loader = get_dataloaders()

    # ------------------------------------------------------------------ #
    # 2. Models
    # ------------------------------------------------------------------ #
    model = AutoEncoder(
        in_channels=config.IN_CHANNELS,
        latent_dim=config.LATENT_DIM,
        is_bn=config.IS_BN,
    ).to(config.DEVICE)

    feat_extractor = ResnetFeatures().to(config.DEVICE)

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

    # ------------------------------------------------------------------ #
    # 5. Calibrate threshold on training data
    # ------------------------------------------------------------------ #
    threshold, recon_errors = compute_threshold(model, feat_extractor, train_loader)
    print(f'Anomaly threshold (mean + 3σ): {threshold:.6f}')

    # ------------------------------------------------------------------ #
    # 6. Evaluate on test set
    # ------------------------------------------------------------------ #
    y_true, y_pred, y_score = predict(model, feat_extractor, threshold)
    best_threshold = plot_roc_and_confusion(y_true, y_score)

    # ------------------------------------------------------------------ #
    # 7. Visualize heatmaps for abnormal images
    # ------------------------------------------------------------------ #
    visualize_heatmaps(model, feat_extractor, best_threshold, recon_errors)


if __name__ == '__main__':
    main()
