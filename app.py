import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

from config import device, IN_CHANNELS, LATENT_DIM, CHECKPOINT_PATH
from dataset import transform
from evaluate import decision_function
from models import AutoEncoder, ResnetFeatures


DEFAULT_IMAGE_PATH = 'sample_images/default.png'
DEFAULT_THRESHOLD = 1.0


@st.cache_resource(show_spinner='Loading model...')
def load_models():
    model = AutoEncoder(in_channels=IN_CHANNELS, latent_dim=LATENT_DIM).to(device)
    state = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    feat_extractor = ResnetFeatures().to(device)
    feat_extractor.eval()
    return model, feat_extractor


def predict(image: Image.Image, model, feat_extractor):
    tensor = transform(image.convert('RGB')).unsqueeze(0).to(device)
    with torch.no_grad():
        features = feat_extractor(tensor)
        recon = model(features)

    segm_map = ((features - recon) ** 2).mean(axis=1)
    score = decision_function(segm_map[:, 3:-3, 3:-3]).item()
    heat_map = segm_map.squeeze().cpu().numpy()
    return score, heat_map, tensor


def render_visuals(image_tensor, heat_map):
    image = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    image = (np.clip(image, 0, 1) * 255).astype(np.uint8)

    heat_resized = cv2.resize(heat_map, (image.shape[1], image.shape[0]))
    span = heat_resized.max() - heat_resized.min()
    heat_norm = (heat_resized - heat_resized.min()) / (span + 1e-8)
    heat_color = cv2.applyColorMap((heat_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(image, 0.5, heat_color, 0.5, 0)
    return image, heat_color, overlay


def main():
    st.set_page_config(
        page_title='Anomaly Detection',
        page_icon='🔍',
        layout='wide',
    )

    st.title('Reconstruction-based Anomaly Detection')
    st.caption(
        'ResNet50 feature maps reconstructed by a 1×1 conv AutoEncoder — '
        'per-pixel reconstruction error drives the anomaly score.'
    )

    model, feat_extractor = load_models()

    with st.sidebar:
        st.header('Settings')
        uploaded = st.file_uploader('Upload an image', type=['png', 'jpg', 'jpeg'])
        threshold = st.slider(
            'Anomaly threshold',
            min_value=0.0,
            max_value=5.0,
            value=float(DEFAULT_THRESHOLD),
            step=0.01,
            help='Image-level score above this value is flagged as abnormal.',
        )
        st.markdown('---')
        st.markdown(f'**Device:** `{device}`')

    if uploaded is not None:
        image = Image.open(uploaded)
        source_caption = f'Uploaded: {uploaded.name}'
    else:
        image = Image.open(DEFAULT_IMAGE_PATH)
        source_caption = 'Default sample (no image uploaded)'
        st.info('No image uploaded — running on the bundled default sample.')

    with st.spinner('Running inference...'):
        score, heat_map, tensor = predict(image, model, feat_extractor)
    original, heat_color, overlay = render_visuals(tensor, heat_map)

    label = 'Abnormal' if score >= threshold else 'Normal'
    color = '#d62728' if label == 'Abnormal' else '#2ca02c'

    c1, c2, c3 = st.columns(3)
    c1.image(original, caption=source_caption, use_container_width=True)
    c2.image(heat_color, caption='Anomaly Heatmap', use_container_width=True)
    c3.image(overlay, caption='Heatmap Overlay', use_container_width=True)

    st.markdown(
        f"### Prediction: <span style='color:{color}'>{label}</span>",
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3)
    m1.metric('Anomaly score', f'{score:.4f}')
    m2.metric('Threshold', f'{threshold:.4f}')
    m3.metric('Score / threshold', f'{score / threshold:.3f}' if threshold > 0 else 'n/a')


if __name__ == '__main__':
    main()
