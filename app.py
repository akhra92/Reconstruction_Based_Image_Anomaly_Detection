import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from PIL import Image

import config
from dataset import get_val_transform
from evaluate import decision_function
from models import AutoEncoder, ResnetFeatures

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title='Anomaly Detection',
    page_icon='🔍',
    layout='wide',
)

THRESHOLD_PATH = 'threshold.npy'

# ---------------------------------------------------------------------------
# Cached model loader
# ---------------------------------------------------------------------------

@st.cache_resource
def load_models():
    feat_extractor = ResnetFeatures().to(config.DEVICE)
    model = AutoEncoder(
        in_channels=config.IN_CHANNELS,
        latent_dim=config.LATENT_DIM,
        is_bn=config.IS_BN,
    ).to(config.DEVICE)
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
    model.eval()
    feat_extractor.eval()
    return model, feat_extractor


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model, feat_extractor, image: Image.Image, threshold: float):
    transform = get_val_transform()
    tensor = transform(image.convert('RGB')).unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():
        features = feat_extractor(tensor)
        recon = model(features)

    segm_map = ((features - recon) ** 2).mean(dim=1)
    score = decision_function(segm_map).item()
    prediction = 'Abnormal' if score >= threshold else 'Normal'
    heat_map = cv2.resize(segm_map.squeeze().cpu().numpy(), (224, 224))
    return score, prediction, heat_map


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title('Settings')

    model_exists = Path(config.MODEL_SAVE_PATH).exists()
    st.markdown(f'**Model:** {"✅ Loaded" if model_exists else "❌ Not found"}')
    st.markdown(f'**Device:** `{config.DEVICE.upper()}`')

    st.divider()

    default_threshold = (
        float(np.load(THRESHOLD_PATH))
        if Path(THRESHOLD_PATH).exists()
        else 0.05
    )
    threshold = st.number_input(
        'Anomaly Threshold',
        min_value=0.0,
        max_value=10.0,
        value=default_threshold,
        format='%.6f',
        help='Images with a score above this value are classified as Abnormal.',
    )

    if Path(THRESHOLD_PATH).exists():
        st.caption(f'Loaded from `{THRESHOLD_PATH}`')
    else:
        st.caption('No saved threshold found. Set manually above.')

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

st.title('AI-Based Anomaly Detection')
st.markdown('Upload one or more images to detect anomalies using the trained AutoEncoder.')

if not model_exists:
    st.error(
        f'Model checkpoint not found at `{config.MODEL_SAVE_PATH}`. '
        'Train the model first by running `python main.py`.'
    )
    st.stop()

model, feat_extractor = load_models()

uploaded_files = st.file_uploader(
    'Upload image(s)',
    type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info('Upload images using the file uploader above.')
    st.stop()

for uploaded_file in uploaded_files:
    image = Image.open(uploaded_file)
    score, prediction, heat_map = run_inference(model, feat_extractor, image, threshold)
    is_abnormal = prediction == 'Abnormal'

    st.divider()
    st.subheader(uploaded_file.name)

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.image(image, caption='Input Image', use_container_width=True)

    with col2:
        fig, ax = plt.subplots()
        im = ax.imshow(heat_map, cmap='jet')
        ax.set_title('Reconstruction Error Heatmap')
        ax.axis('off')
        fig.colorbar(im, ax=ax)
        st.pyplot(fig)
        plt.close(fig)

    with col3:
        st.metric('Anomaly Score', f'{score:.6f}')
        st.metric('Threshold', f'{threshold:.6f}')
        ratio = score / threshold if threshold > 0 else float('inf')
        st.metric('Score / Threshold', f'{ratio:.4f}')
        if is_abnormal:
            st.error('Abnormal')
        else:
            st.success('Normal')
