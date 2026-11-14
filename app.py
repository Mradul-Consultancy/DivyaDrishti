"""
DivyaDrishti (दिव्यदृष्टि) AI - Interactive Deepfake Detection Studio
Streamlit application for image & video deepfake detection and analysis.
"""

import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import streamlit as st

# Setup path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import config
from models.detector import get_model
from preprocessing.face_extractor import FaceExtractor
from preprocessing.transforms import get_inference_transforms

st.set_page_config(
    page_title="DivyaDrishti AI — Deepfake Detection",
    page_icon="🛡️",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .verdict-box {
        padding: 1.2rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
        font-weight: 600;
    }
    .verdict-fake {
        background-color: #FEE2E2;
        color: #991B1B;
        border: 1px solid #F87171;
    }
    .verdict-real {
        background-color: #DCFCE7;
        color: #166534;
        border: 1px solid #4ADE80;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_detector_model(model_name: str, checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(checkpoint_path):
        return None, device
    try:
        model = get_model(model_name=model_name, num_classes=2, pretrained=False)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Unpack state_dict if checkpoint contains metadata
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Strip 'module.' prefix if trained with DataParallel
        clean_state_dict = {}
        for k, v in state_dict.items():
            key = k.replace("module.", "")
            clean_state_dict[key] = v

        model.load_state_dict(clean_state_dict)
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        return None, device


@st.cache_resource
def get_face_extractor():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return FaceExtractor(margin_ratio=config.FACE_MARGIN_RATIO, device=device)


def main():
    st.markdown('<div class="main-header">🛡️ DivyaDrishti (दिव्यदृष्टि) AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Advanced Deepfake Image & Video Verification Platform</div>', unsafe_allow_html=True)

    # Sidebar: Model Selection
    st.sidebar.title("⚙️ Model Configuration")
    
    checkpoints_dir = config.CHECKPOINT_DIR

    model_options = {
        "EfficientNet-B3 (Recommended - AUC 0.925)": {
            "name": "efficientnet_b3.ra2_in1k",
            "file": "efficientnet_b3.ra2_in1k_best.pth"
        },
        "ConvNeXt-Tiny (High Robustness)": {
            "name": "convnext_tiny.fb_in22k_ft_in1k",
            "file": "convnext_tiny.fb_in22k_ft_in1k_best.pth"
        },
        "ResNet-18 (Lightweight Baseline)": {
            "name": "resnet18",
            "file": "resnet18_best.pth"
        }
    }

    selected_model_label = st.sidebar.selectbox("Choose Detection Backbone", list(model_options.keys()))
    model_cfg = model_options[selected_model_label]
    ckpt_path = str(checkpoints_dir / model_cfg["file"])

    if not os.path.exists(ckpt_path):
        st.sidebar.warning(f"⚠️ Checkpoint `{model_cfg['file']}` not found in `models/checkpoints/`.")
        st.sidebar.info("Download pre-trained weights from GitHub Releases v1.0.0.")
    else:
        st.sidebar.success(f"✅ Loaded checkpoint: `{model_cfg['file']}`")

    threshold = st.sidebar.slider("Classification Threshold (Fake %)", min_value=10, max_value=90, value=50, step=5) / 100.0

    # Initialize modules
    extractor = get_face_extractor()
    transform = get_inference_transforms(image_size=config.IMAGE_SIZE)
    model, device = load_detector_model(model_cfg["name"], ckpt_path)

    # Tabs
    tab1, tab2 = st.tabs(["📸 Image Detection", "🎥 Video Inspection"])

    # Tab 1: Image
    with tab1:
        st.subheader("Upload an Image for Analysis")
        uploaded_image = st.file_uploader("Select JPG, PNG, or WEBP image", type=["jpg", "jpeg", "png", "webp"])

        if uploaded_image:
            col1, col2 = st.columns([1, 1])
            with col1:
                img_pil = Image.open(uploaded_image).convert("RGB")
                st.image(img_pil, caption="Source Image", use_container_width=True)

            with col2:
                if model is None:
                    st.error("Please ensure the selected model checkpoint exists.")
                else:
                    with st.spinner("Detecting facial bounding box and evaluating forgery indicators..."):
                        # Extract face
                        face_crop = extractor.extract_face(img_pil)
                        st.image(face_crop, caption="Cropped Facial Area", width=180)

                        # Inference
                        tensor = transform(face_crop).unsqueeze(0).to(device)
                        with torch.no_grad():
                            outputs = model(tensor)
                            probs = torch.softmax(outputs, dim=1).squeeze(0)
                            prob_fake = float(probs[1].item())

                        is_fake = prob_fake >= threshold

                        st.write("---")
                        st.write("### Detection Verdict")
                        
                        prob_pct = prob_fake * 100.0
                        if is_fake:
                            st.markdown(
                                f'<div class="verdict-box verdict-fake">'
                                f'🚨 <b>VERDICT: SYNTHETIC / DEEPFAKE DETECTED</b><br>'
                                f'Fake Probability: {prob_pct:.2f}% (Threshold: {threshold*100:.0f}%)'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                f'<div class="verdict-box verdict-real">'
                                f'✅ <b>VERDICT: AUTHENTIC / REAL MEDIA</b><br>'
                                f'Real Probability: {100.0 - prob_pct:.2f}% (Fake Probability: {prob_pct:.2f}%)'
                                f'</div>',
                                unsafe_allow_html=True
                            )

                        st.progress(float(prob_fake))

    # Tab 2: Video
    with tab2:
        st.subheader("Upload a Video for Frame-by-Frame Deepfake Inspection")
        uploaded_video = st.file_uploader("Select MP4, AVI, or MOV video", type=["mp4", "avi", "mov"])

        if uploaded_video:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_video.read())
            video_path = tfile.name

            col_v1, col_v2 = st.columns([1, 1])
            with col_v1:
                st.video(video_path)

            with col_v2:
                if model is None:
                    st.error("Checkpoint required for inference.")
                else:
                    if st.button("🚀 Analyze Video Frames"):
                        with st.spinner("Sampling video frames and running neural verification..."):
                            cap = cv2.VideoCapture(video_path)
                            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            step = max(1, total_frames // 15)  # Sample ~15 frames
                            
                            scores = []
                            current = 0

                            while cap.isOpened() and len(scores) < 15:
                                ret, frame = cap.read()
                                if not ret:
                                    break
                                if current % step == 0:
                                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                    frame_pil = Image.fromarray(frame_rgb)
                                    face_crop = extractor.extract_face(frame_pil)
                                    
                                    tensor = transform(face_crop).unsqueeze(0).to(device)
                                    with torch.no_grad():
                                        outputs = model(tensor)
                                        probs = torch.softmax(outputs, dim=1).squeeze(0)
                                        scores.append(float(probs[1].item()))
                                current += 1
                            cap.release()

                            if scores:
                                avg_fake = float(np.mean(scores))
                                max_fake = float(np.max(scores))
                                is_fake_video = avg_fake >= threshold

                                st.write("### Video Assessment Results")
                                if is_fake_video:
                                    st.markdown(
                                        f'<div class="verdict-box verdict-fake">'
                                        f'🚨 <b>DEEPFAKE DETECTED IN VIDEO</b><br>'
                                        f'Average Fake Probability: {avg_fake*100:.2f}% | Peak Frame Fake Score: {max_fake*100:.2f}%'
                                        f'</div>',
                                        unsafe_allow_html=True
                                    )
                                else:
                                    st.markdown(
                                        f'<div class="verdict-box verdict-real">'
                                        f'✅ <b>VIDEO ASSESSED AS AUTHENTIC</b><br>'
                                        f'Average Fake Score: {avg_fake*100:.2f}%'
                                        f'</div>',
                                        unsafe_allow_html=True
                                    )

                                st.line_chart({"Fake Probability": scores})


if __name__ == "__main__":
    main()
