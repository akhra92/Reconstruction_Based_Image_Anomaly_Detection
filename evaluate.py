from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay, f1_score,
)

import config
from dataset import get_transform


# ---------------------------------------------------------------------------
# Anomaly scoring
# ---------------------------------------------------------------------------

def decision_function(segm_map: torch.Tensor) -> torch.Tensor:
    """Return the mean of the top-10 pixel values per map as the anomaly score."""
    scores = []
    for m in segm_map:
        flat = m.reshape(-1)
        sorted_vals, _ = torch.sort(flat, descending=True)
        scores.append(sorted_vals[:10].mean())
    return torch.stack(scores)


# ---------------------------------------------------------------------------
# Threshold calibration
# ---------------------------------------------------------------------------

def compute_threshold(model, feat_extractor, train_loader):
    """Estimate anomaly threshold as mean + 3*std over training reconstruction errors."""
    model.eval()
    feat_extractor.eval()
    all_scores = []

    with torch.no_grad():
        for data, _ in train_loader:
            features = feat_extractor(data.to(config.DEVICE)).squeeze()
            recon = model(features)
            segm_map = ((features - recon) ** 2).mean(dim=1)[:, 3:-3, 3:-3]
            all_scores.append(decision_function(segm_map))

    recon_errors = torch.cat(all_scores).cpu().numpy()
    threshold = float(np.mean(recon_errors) + 3 * np.std(recon_errors))

    plt.hist(recon_errors, bins=50)
    plt.axvline(x=threshold, color='r', label=f'Threshold = {threshold:.4f}')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Count')
    plt.title('Training Reconstruction Error Distribution')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return threshold, recon_errors


# ---------------------------------------------------------------------------
# Inference on test set
# ---------------------------------------------------------------------------

def predict(model, feat_extractor, threshold: float):
    """Run inference on the test set and return ground-truth labels, predictions and scores."""
    model.eval()
    feat_extractor.eval()
    transform = get_transform()
    test_path = Path(config.TEST_DATA_PATH)

    y_true, y_pred, y_score = [], [], []

    with torch.no_grad():
        for path in test_path.glob('*/*.*'):
            fault_type = path.parts[-2]
            image = transform(Image.open(path)).unsqueeze(0).to(config.DEVICE)
            features = feat_extractor(image)
            recon = model(features)
            segm_map = ((features - recon) ** 2).mean(dim=1)[:, 3:-3, 3:-3]
            score = decision_function(segm_map)

            y_true.append(0 if fault_type == 'good' else 1)
            y_pred.append(int(score.item() >= threshold))
            y_score.append(score.item())

    return np.array(y_true), np.array(y_pred), np.array(y_score)


# ---------------------------------------------------------------------------
# Metrics & plots
# ---------------------------------------------------------------------------

def plot_roc_and_confusion(y_true, y_score):
    """Plot ROC curve and confusion matrix; return the F1-optimal threshold."""
    auc = roc_auc_score(y_true, y_score)
    print(f'AUC-ROC: {auc:.4f}')

    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

    f1_scores = [f1_score(y_true, y_score >= t) for t in thresholds]
    best_threshold = float(thresholds[np.argmax(f1_scores)])
    print(f'Best F1 threshold: {best_threshold:.6f}')

    cm = confusion_matrix(y_true, (y_score >= best_threshold).astype(int))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Abnormal'])
    disp.plot()
    plt.tight_layout()
    plt.show()

    return best_threshold


def visualize_heatmaps(model, feat_extractor, best_threshold: float, recon_errors: np.ndarray):
    """Overlay reconstruction-error heatmaps on abnormal test images."""
    model.eval()
    feat_extractor.eval()
    transform = get_transform()
    test_path = Path(config.TEST_DATA_PATH)
    heat_map_min = float(np.min(recon_errors))
    heat_map_max = float(np.max(recon_errors))

    with torch.no_grad():
        for path in test_path.glob('*/*.*'):
            fault_type = path.parts[-2]
            if fault_type != 'bad':
                continue

            image = transform(Image.open(path)).unsqueeze(0).to(config.DEVICE)
            features = feat_extractor(image)
            recon = model(features)
            segm_map = ((features - recon) ** 2).mean(dim=1)
            score = decision_function(segm_map)

            heat_map = cv2.resize(segm_map.squeeze().cpu().numpy(), (128, 128))

            plt.figure(figsize=(15, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(image.squeeze().permute(1, 2, 0).cpu().numpy())
            plt.title('Original Abnormal Image')

            plt.subplot(1, 2, 2)
            plt.imshow(heat_map, cmap='jet', vmin=heat_map_min, vmax=heat_map_max * 10)
            plt.title(f'Heatmap  |  Score ratio: {score[0].item() / best_threshold:.4f}')

            plt.tight_layout()
            plt.show()
