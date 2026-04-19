import warnings
warnings.filterwarnings('ignore')

import os
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             ConfusionMatrixDisplay, f1_score)

from config import device, TEST_DIR, SAMPLE_TEST_IMAGE, ASSETS_DIR
from dataset import transform


def visualize_single_abnormal(model, feat_extractor, image_path=SAMPLE_TEST_IMAGE):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        features = feat_extractor(image.to(device))
        recon = model(features)

    recon_error = ((features - recon) ** 2).mean(axis=(1)).unsqueeze(0)

    segm_map = torch.nn.functional.interpolate(     # Upscale by bi-linaer interpolation to match the original input resolution
        recon_error,
        size=(224, 224),
        mode='bilinear'
    )

    plt.figure()
    plt.imshow(segm_map.squeeze().cpu().numpy(), cmap='jet')
    plt.savefig(os.path.join(ASSETS_DIR, 'single_abnormal_segmap.png'), bbox_inches='tight')
    plt.show()
    plt.close()


def decision_function(segm_map):

    mean_top_10_values = []

    for map in segm_map:
        flattened_tensor = map.reshape(-1)

        sorted_tensor, _ = torch.sort(flattened_tensor, descending=True)

        mean_top_10_value = sorted_tensor[:10].mean()

        mean_top_10_values.append(mean_top_10_value)

    return torch.stack(mean_top_10_values)


def compute_reconstruction_error(model, feat_extractor, train_loader):
    model.eval()

    RECON_ERROR = []
    for data, _ in train_loader:

        with torch.no_grad():
            features = feat_extractor(data.to(device)).squeeze()
            recon = model(features)

        segm_map = ((features - recon) ** 2).mean(axis=(1))[:, 3:-3, 3:-3]
        anomaly_score = decision_function(segm_map)

        RECON_ERROR.append(anomaly_score)

    RECON_ERROR = torch.cat(RECON_ERROR).cpu().numpy()
    return RECON_ERROR


def compute_best_threshold(RECON_ERROR):
    best_threshold = np.mean(RECON_ERROR) + 3 * np.std(RECON_ERROR)

    heat_map_max, heat_map_min = np.max(RECON_ERROR), np.min(RECON_ERROR)

    plt.figure()
    plt.hist(RECON_ERROR, bins=50)
    plt.vlines(x=best_threshold, ymin=0, ymax=30, color='r')
    plt.savefig(os.path.join(ASSETS_DIR, 'recon_error_hist.png'), bbox_inches='tight')
    plt.show()
    plt.close()

    return best_threshold, heat_map_max, heat_map_min


def predict_test_images(model, feat_extractor, best_threshold, test_dir=TEST_DIR):
    y_true = []
    y_pred = []
    y_score = []

    model.eval()
    feat_extractor.eval()

    test_path = Path(test_dir)

    for path in test_path.glob('*/*.*'):
        fault_type = path.parts[-2]
        test_image = transform(Image.open(path)).to(device).unsqueeze(0)

        with torch.no_grad():
            features = feat_extractor(test_image)
            recon = model(features)

        segm_map = ((features - recon) ** 2).mean(axis=(1))[:, 3:-3, 3:-3]
        y_score_image = decision_function(segm_map=segm_map)

        y_pred_image = 1 * (y_score_image >= best_threshold)

        y_true_image = 0 if fault_type == 'good' else 1

        y_true.append(y_true_image)
        y_pred.append(y_pred_image.cpu().numpy())
        y_score.append(y_score_image.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_score = np.array(y_score)

    return y_true, y_pred, y_score


def plot_score_histogram(y_score, best_threshold):
    plt.figure()
    plt.hist(y_score, bins=50)
    plt.vlines(x=best_threshold, ymin=0, ymax=30, color='r')
    plt.savefig(os.path.join(ASSETS_DIR, 'score_hist.png'), bbox_inches='tight')
    plt.show()
    plt.close()


def plot_roc_and_confusion(y_true, y_score):
    auc_roc_score = roc_auc_score(y_true, y_score)
    print("AUC-ROC Score:", auc_roc_score)

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc_roc_score)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(ASSETS_DIR, 'roc_curve.png'), bbox_inches='tight')
    plt.show()
    plt.close()

    f1_scores = [f1_score(y_true, y_score >= threshold) for threshold in thresholds]

    best_threshold = thresholds[np.argmax(f1_scores)]  # finding best threshold based on f1 scores

    print(f'best_threshold = {best_threshold}')

    cm = confusion_matrix(y_true, (y_score >= best_threshold).astype(int), labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Abnormal'])
    disp.plot()
    plt.savefig(os.path.join(ASSETS_DIR, 'confusion_matrix.png'), bbox_inches='tight')
    plt.show()
    plt.close()

    return best_threshold


def visualize_abnormal_heatmaps(model, feat_extractor, best_threshold, heat_map_min, heat_map_max, test_dir=TEST_DIR):
    model.eval()
    feat_extractor.eval()

    test_path = Path(test_dir)

    for path in test_path.glob('*/*.*'):
        fault_type = path.parts[-2]
        test_image = transform(Image.open(path)).to(device).unsqueeze(0)

        with torch.no_grad():
            features = feat_extractor(test_image)
            recon = model(features)

        segm_map = ((features - recon) ** 2).mean(axis=(1))
        y_score_image = decision_function(segm_map=segm_map)

        y_pred_image = 1 * (y_score_image >= best_threshold)
        class_label = ['Normal', 'Abnormal']

        if fault_type in ['bad']:

            plt.figure(figsize=(15, 5))

            plt.subplot(1, 2, 1)
            plt.imshow(test_image.squeeze().permute(1, 2, 0).cpu().numpy())
            plt.title('Original Abnormal Image')

            plt.subplot(1, 2, 2)
            heat_map = segm_map.squeeze().cpu().numpy()
            heat_map = heat_map
            heat_map = cv2.resize(heat_map, (128, 128))
            plt.imshow(heat_map, cmap='jet', vmin=heat_map_min, vmax=heat_map_max * 10)
            plt.title(f'Generated Heatmap with Anomaly Score: {y_score_image[0].cpu().numpy() / best_threshold:0.4f}')

            plt.savefig(os.path.join(ASSETS_DIR, f'heatmap_{path.stem}.png'), bbox_inches='tight')
            plt.show()
            plt.close()
