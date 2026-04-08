# AI-Based Anomaly Detection

Unsupervised anomaly detection using a **ResNet50 feature extractor** and a **1×1 Conv AutoEncoder**. The model is trained exclusively on normal images and detects anomalies at inference time by measuring patch-level reconstruction error.

---

## How It Works

1. **Feature extraction** — A frozen pretrained ResNet50 extracts intermediate feature maps from `layer2` and `layer3`, which are pooled and concatenated into a single patch tensor.
2. **AutoEncoder** — A lightweight 1×1 convolutional autoencoder learns to reconstruct normal patch features. Anomalous regions produce high reconstruction error.
3. **Anomaly scoring** — The mean of the top-10 reconstruction error values per image is used as the anomaly score.
4. **Thresholding** — The decision threshold is set to `mean + 3×std` of training scores, then refined using the F1-optimal threshold on the test set.

---

## Project Structure

```
Anomaly_Detection/
├── main.py          # Entry point — runs the full pipeline
├── config.py        # All hyperparameters and paths
├── dataset.py       # Transform and DataLoader factory
├── models.py        # ResnetFeatures and AutoEncoder
├── train.py         # Training loop with early stopping
├── evaluate.py      # Threshold calibration, inference, metrics, heatmaps
└── requirements.txt
```

---

## Dataset Structure

Place your data under a `dataset/` folder in the project root:

```
dataset/
├── train/
│   └── good/          # Normal images only
└── test/
    ├── good/          # Normal test images
    └── bad/           # Anomalous test images
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

```bash
python main.py
```

The pipeline will:
1. Train the autoencoder on `dataset/train/`
2. Save the best checkpoint to `AE_ResNet50.pth`
3. Calibrate the anomaly threshold on training reconstruction errors
4. Evaluate on `dataset/test/` and report AUC-ROC + confusion matrix
5. Display heatmaps highlighting anomalous regions in bad images

---

## Generated Plots

All plots are saved to the `plots/` directory after each run.

### Learning Curves
Training and validation reconstruction loss per epoch.

![Learning Curves](plots/learning_curves.png)

### Threshold Distribution
Histogram of training anomaly scores with the decision threshold (`mean + 3σ`) marked in red.

![Threshold Distribution](plots/threshold_distribution.png)

### ROC Curve
Receiver Operating Characteristic curve with AUC score on the test set.

![ROC Curve](plots/roc_curve.png)

### Confusion Matrix
Predicted vs. actual labels at the F1-optimal threshold.

![Confusion Matrix](plots/confusion_matrix.png)

### Anomaly Heatmaps
Per-image side-by-side view of the original abnormal image and its reconstruction error heatmap. One file is saved per bad test image: `plots/heatmap_<filename>.png`.

![Heatmap Example](plots/heatmap_example.png)

---

## Configuration

All settings are in [config.py](config.py):

| Parameter | Default | Description |
|---|---|---|
| `IMAGE_SIZE` | 224 | Input image size |
| `BATCH_SIZE` | 4 | Training batch size |
| `VAL_SPLIT` | 0.2 | Fraction of training data used for validation |
| `IN_CHANNELS` | 1536 | Concatenated feature map channels (layer2 + layer3) |
| `LATENT_DIM` | 100 | AutoEncoder bottleneck dimension |
| `NUM_EPOCHS` | 50 | Maximum training epochs |
| `LEARNING_RATE` | 0.001 | Adam optimizer learning rate |
| `EARLY_STOP_PATIENCE` | 5 | Epochs without improvement before stopping |
| `MODEL_SAVE_PATH` | `AE_ResNet50.pth` | Checkpoint file path |
