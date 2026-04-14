import base64
import io
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

import config
from dataset import get_val_transform
from evaluate import decision_function
from models import AutoEncoder, ResnetFeatures

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models once at startup
    if not Path(config.MODEL_SAVE_PATH).exists():
        raise RuntimeError(
            f'Model checkpoint not found at {config.MODEL_SAVE_PATH!r}. '
            'Run `python main.py` first.'
        )

    feat_extractor = ResnetFeatures(finetune_layers=config.FINETUNE_LAYERS).to(config.DEVICE)
    model = AutoEncoder(
        in_channels=config.IN_CHANNELS,
        latent_dim=config.LATENT_DIM,
        is_bn=config.IS_BN,
    ).to(config.DEVICE)
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
    model.eval()
    feat_extractor.eval()

    threshold = (
        float(np.load('threshold.npy'))
        if Path('threshold.npy').exists()
        else config.THRESHOLD_SIGMA  # fallback — should not happen in practice
    )

    _state['model'] = model
    _state['feat_extractor'] = feat_extractor
    _state['threshold'] = threshold
    _state['transform'] = get_val_transform()

    yield

    _state.clear()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title='Anomaly Detection API',
    description='REST API for image anomaly detection using ResNet50 + AutoEncoder.',
    version='1.0.0',
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class PredictionResponse(BaseModel):
    filename: str
    score: float
    threshold: float
    prediction: str          # "Normal" | "Abnormal"
    score_ratio: float       # score / threshold
    heatmap_b64: str | None  # base64-encoded PNG heatmap (if requested)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_inference(image: Image.Image, include_heatmap: bool) -> dict:
    transform = _state['transform']
    model = _state['model']
    feat_extractor = _state['feat_extractor']
    threshold = _state['threshold']

    tensor = transform(image.convert('RGB')).unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():
        features = feat_extractor(tensor)
        recon = model(features)

    segm_map = ((features - recon) ** 2).mean(dim=1)
    score = decision_function(segm_map).item()
    prediction = 'Abnormal' if score >= threshold else 'Normal'

    heatmap_b64 = None
    if include_heatmap:
        sz = config.HEATMAP_SIZE
        heat = cv2.resize(segm_map.squeeze().cpu().numpy(), (sz, sz))
        heat_norm = ((heat - heat.min()) / (heat.max() - heat.min() + 1e-8) * 255).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)
        _, buf = cv2.imencode('.png', heat_color)
        heatmap_b64 = base64.b64encode(buf.tobytes()).decode('utf-8')

    return {
        'score': score,
        'threshold': threshold,
        'prediction': prediction,
        'score_ratio': score / threshold if threshold > 0 else float('inf'),
        'heatmap_b64': heatmap_b64,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get('/health', tags=['System'])
def health():
    """Returns service status and current threshold."""
    return {
        'status': 'ok',
        'device': config.DEVICE,
        'threshold': _state.get('threshold'),
    }


@app.post('/predict', response_model=PredictionResponse, tags=['Inference'])
async def predict(
    file: UploadFile = File(..., description='Image file (PNG, JPG, BMP, TIFF)'),
    heatmap: bool = False,
):
    """Run anomaly detection on a single uploaded image.

    Returns the anomaly score, binary prediction, and optionally a
    base64-encoded heatmap PNG.
    """
    allowed = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status_code=415, detail=f'Unsupported file type: {suffix}')

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f'Could not decode image: {exc}')

    result = _run_inference(image, include_heatmap=heatmap)
    return PredictionResponse(filename=file.filename, **result)


@app.post('/predict/batch', tags=['Inference'])
async def predict_batch(
    files: list[UploadFile] = File(..., description='Multiple image files'),
    heatmap: bool = False,
):
    """Run anomaly detection on multiple uploaded images."""
    results = []
    allowed = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}

    for file in files:
        suffix = Path(file.filename).suffix.lower()
        if suffix not in allowed:
            results.append({'filename': file.filename, 'error': f'Unsupported file type: {suffix}'})
            continue

        contents = await file.read()
        try:
            image = Image.open(io.BytesIO(contents))
        except Exception as exc:
            results.append({'filename': file.filename, 'error': str(exc)})
            continue

        result = _run_inference(image, include_heatmap=heatmap)
        results.append({'filename': file.filename, **result})

    return JSONResponse(content=results)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    uvicorn.run('api:app', host='0.0.0.0', port=8000, reload=False)