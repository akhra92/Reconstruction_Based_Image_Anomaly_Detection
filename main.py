import warnings
warnings.filterwarnings('ignore')

from config import device, IN_CHANNELS, LATENT_DIM, CHECKPOINT_PATH
from dataset import get_dataloaders
from models import AutoEncoder, ResnetFeatures
from train import train, load_model
from evaluate import (visualize_single_abnormal, compute_reconstruction_error,
                      compute_best_threshold, predict_test_images,
                      plot_score_histogram, plot_roc_and_confusion,
                      visualize_abnormal_heatmaps)


def main():
    # Train the model (saves checkpoint to CHECKPOINT_PATH)
    model, feat_extractor, train_loader, val_loader = train()

    # Load the trained model
    model = load_model(model, CHECKPOINT_PATH)
    print(model)

    # Visualize the abnormal image result (single image)
    visualize_single_abnormal(model, feat_extractor)

    # Calculate reconstruction error on training data
    RECON_ERROR = compute_reconstruction_error(model, feat_extractor, train_loader)

    # Compute the initial best threshold (mean + 3*std)
    best_threshold, heat_map_max, heat_map_min = compute_best_threshold(RECON_ERROR)

    # Predict on test images
    y_true, y_pred, y_score = predict_test_images(model, feat_extractor, best_threshold)

    # Visualize the predicted anomaly score histogram with the best threshold
    plot_score_histogram(y_score, best_threshold)

    # ROC curve, F1-based best threshold and confusion matrix
    best_threshold = plot_roc_and_confusion(y_true, y_score)

    # Visualize abnormal test images with heatmaps
    visualize_abnormal_heatmaps(model, feat_extractor, best_threshold, heat_map_min, heat_map_max)


if __name__ == '__main__':
    main()
