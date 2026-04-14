# AI-Based Anomaly Detection

Unsupervised anomaly detection using a **ResNet50 feature extractor** and a **1×1 Conv AutoEncoder**. The model is trained exclusively on normal images and detects anomalies at inference time by measuring patch-level reconstruction error.

---

## How It Works

1. **Feature extraction** — A pretrained ResNet50 extracts intermediate feature maps from `layer2` and `layer3`, pooled and concatenated into a single patch tensor. Selected layers can optionally be fine-tuned on domain-specific data.
2. **AutoEncoder** — A lightweight 1×1 convolutional autoencoder learns to reconstruct normal patch features. Anomalous regions produce high reconstruction error.
3. **Anomaly scoring** — The mean of the top-K reconstruction error values per image is used as the anomaly score.
4. **Thresholding** — The decision threshold is set to `mean + N×std` of training scores (no test-set leakage). The F1-optimal threshold from the test set is reported as an oracle reference only.

---

## Project Structure

```
Anomaly_Detection/
├── main.py                   # CLI entry point — trains and evaluates the model
├── app.py                    # Streamlit web app for interactive inference
├── api.py                    # FastAPI REST endpoint for system integration
├── config.py                 # All hyperparameters and paths
├── dataset.py                # Transform and DataLoader factory
├── models.py                 # ResnetFeatures and AutoEncoder
├── train.py                  # Training loop with early stopping
├── evaluate.py               # Threshold calibration, inference, metrics, heatmaps
├── requirements.txt          # Python dependencies
├── packages.txt              # System dependencies for Streamlit Cloud
└── .streamlit/
    └── config.toml           # Streamlit theme and upload size config
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

### CLI (training + evaluation)

```bash
python main.py
```

The pipeline will:
1. Train the autoencoder on `dataset/train/` with data augmentation
2. Save the best checkpoint to `AE_ResNet50.pth`
3. Calibrate the anomaly threshold on training reconstruction errors (`mean + N×std`)
4. Evaluate on `dataset/test/` and report AUC-ROC + confusion matrix
5. Save the threshold to `threshold.npy`
6. Display heatmaps highlighting anomalous regions in bad images

### Streamlit App

```bash
streamlit run app.py
```

Upload images in the browser to get anomaly scores, heatmaps, and Normal/Abnormal predictions interactively.

You can also try a live demo of the project on Streamlit Cloud: [Demo Link](https://share.streamlit.io)

### REST API

```bash
python api.py
# or: uvicorn api:app --reload
```

Interactive API docs are available at `http://localhost:8000/docs`.

#### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Service status and current threshold |
| `POST` | `/predict` | Single image → score, prediction, optional heatmap |
| `POST` | `/predict/batch` | Multiple images → list of results |

#### Example

```bash
# Single image
curl -X POST "http://localhost:8000/predict?heatmap=true" \
  -F "file=@image.jpg"

# Batch
curl -X POST "http://localhost:8000/predict/batch" \
  -F "files=@img1.jpg" -F "files=@img2.png"
```

Response:
```json
{
  "filename": "image.jpg",
  "score": 0.003842,
  "threshold": 0.002951,
  "prediction": "Abnormal",
  "score_ratio": 1.3019,
  "heatmap_b64": "<base64-encoded PNG>"
}
```

---

## Generated Plots

All plots are saved to the `plots/` directory after each run.

### Learning Curves
Training and validation reconstruction loss per epoch.

![Learning Curves](plots/learning_curves.png)

### Threshold Distribution
Histogram of training anomaly scores with the decision threshold (`mean + N×std`) marked in red.

![Threshold Distribution](plots/threshold_distribution.png)

### ROC Curve
Receiver Operating Characteristic curve with AUC score on the test set.

![ROC Curve](plots/roc_curve.png)

### Confusion Matrix
Predicted vs. actual labels at the deployed (training-based) threshold.

![Confusion Matrix](plots/confusion_matrix.png)

### Anomaly Heatmaps
Per-image side-by-side view of the original abnormal image and its reconstruction error heatmap. One file is saved per bad test image: `plots/heatmap_<filename>.png`.

![Heatmap Example](plots/heatmap_example.png)

---

## Configuration

All settings are in [config.py](config.py):

### Data

| Parameter | Default | Description |
|---|---|---|
| `IMAGE_SIZE` | 224 | Input image size |
| `BATCH_SIZE` | 4 | Training batch size |
| `VAL_SPLIT` | 0.2 | Fraction of training data used for validation |

### Data Augmentation

| Parameter | Default | Description |
|---|---|---|
| `AUG_ROTATION_DEGREES` | 15 | Max rotation angle |
| `AUG_BRIGHTNESS` | 0.2 | ColorJitter brightness factor |
| `AUG_CONTRAST` | 0.2 | ColorJitter contrast factor |
| `AUG_SATURATION` | 0.1 | ColorJitter saturation factor |
| `AUG_ERASING_PROB` | 0.1 | RandomErasing probability |
| `AUG_ERASING_SCALE` | (0.02, 0.1) | RandomErasing patch size range |

### Model

| Parameter | Default | Description |
|---|---|---|
| `IN_CHANNELS` | 1536 | Concatenated feature channels (layer2 + layer3) |
| `LATENT_DIM` | 100 | AutoEncoder bottleneck dimension |
| `FINETUNE_LAYERS` | `['layer3']` | ResNet layers to unfreeze; set to `[]` to disable |
| `FINETUNE_LR` | 1e-4 | Learning rate for backbone fine-tuning |

### Training

| Parameter | Default | Description |
|---|---|---|
| `NUM_EPOCHS` | 50 | Maximum training epochs |
| `LEARNING_RATE` | 0.001 | Adam learning rate for the AutoEncoder |
| `EARLY_STOP_PATIENCE` | 5 | Epochs without improvement before stopping |
| `MODEL_SAVE_PATH` | `AE_ResNet50.pth` | Checkpoint file path |

### Anomaly Scoring

| Parameter | Default | Description |
|---|---|---|
| `TOP_K_PIXELS` | 10 | Top-K error pixels used in the anomaly score |
| `BORDER_CROP` | 3 | Edge pixels to crop from the segmentation map |
| `THRESHOLD_SIGMA` | 3.0 | Std multiplier for the decision threshold |
| `HEATMAP_VMAX_SCALE` | 10.0 | Heatmap color scale factor |
| `HEATMAP_SIZE` | 128 | Output heatmap resolution |