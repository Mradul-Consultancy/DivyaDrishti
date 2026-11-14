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
from evaluation.evaluate import get_eval_transform
from evaluation.calibration import load_threshold
from evaluation.gradcam import generate_gradcam_for_model

# Standard ImageFolder alphabetical sorting: 'fake' = 0, 'real' = 1
FAKE_CLASS_IDX = config.FAKE_LABEL_IDX  # 0
REAL_CLASS_IDX = config.REAL_LABEL_IDX  # 1

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
        return None, None, device
    try:
        model = get_model(model_name=model_name, num_classes=2, pretrained=False)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(clean_state_dict)
        model.to(device)
        model.eval()

        transform = get_eval_transform(model, model_name)
        return model, transform, device
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        return None, None, device


@st.cache_resource
def get_face_extractor():
    return FaceExtractor(margin_ratio=0.2)


def main():
    st.markdown('<div class="main-header">🛡️ DivyaDrishti (दिव्यदृष्टि) AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Advanced Deepfake Image & Video Verification Platform</div>', unsafe_allow_html=True)

    # Sidebar: Model Selection
    st.sidebar.title("⚙️ Model Configuration")
    
    checkpoints_dir = config.CHECKPOINT_DIR

    model_options = {
        "ConvNeXt-Tiny (CelebDF Tuned - Best Discrimination)": {
            "name": "convnext_tiny.fb_in22k_ft_in1k",
            "file": "celebdf/convnext_tiny.fb_in22k_ft_in1k_best.pth",
            "domain": "celebdf",
        },
        "ConvNeXt-Tiny (General Purpose Backbone)": {
            "name": "convnext_tiny.fb_in22k_ft_in1k",
            "file": "convnext_tiny.fb_in22k_ft_in1k_best.pth",
            "domain": "celebdf",
        },
        "EfficientNet-B3 (In-Domain Checkpoint)": {
            "name": "efficientnet_b3.ra2_in1k",
            "file": "efficientnet_b3.ra2_in1k_best.pth",
            "domain": "celebdf",
        },
        "ResNet-18 (Lightweight Baseline)": {
            "name": "resnet18",
            "file": "resnet18_best.pth",
            "domain": "celebdf",
        }
    }

    selected_model_label = st.sidebar.selectbox("Choose Detection Backbone", list(model_options.keys()))
    model_cfg = model_options[selected_model_label]
    ckpt_path = str(checkpoints_dir / model_cfg["file"])

    if not os.path.exists(ckpt_path):
        st.sidebar.warning(f"⚠️ Checkpoint `{model_cfg['file']}` not found.")
        st.sidebar.info("Download pre-trained weights from GitHub Releases v1.0.0.")
    else:
        st.sidebar.success(f"✅ Loaded: `{Path(model_cfg['file']).name}`")

    # The threshold defaults to the value calibrated on this domain's
    # VALIDATION split. At 0.5 the detectors called roughly half of all
    # authentic faces fake on the imbalanced Celeb-DF v2 split, so the
    # default matters a great deal.
    domain = model_cfg.get("domain")
    calibrated = load_threshold(domain, model_cfg["name"]) if domain else 0.5
    threshold = st.sidebar.slider(
        "Classification Threshold (Fake %)",
        min_value=10, max_value=99,
        value=int(round(calibrated * 100)), step=1,
    ) / 100.0
    if domain:
        st.sidebar.caption(
            f"Calibrated on {domain} validation data: {calibrated:.3f} "
            "(balanced operating point)."
        )
    show_heatmap = st.sidebar.checkbox(
        "Show Grad-CAM explainability overlay", value=False,
        help="Highlights the regions that drove the score - evidence a "
             "reviewer needs before acting on a verdict.",
    )

    # Quick Samples Loader in Sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧪 Quick Test Samples")
    sample_choice = st.sidebar.radio(
        "Load Pre-packaged Test Sample",
        ["None (Custom Upload)", "Real Sample (CelebDF Real Face)", "Fake Sample (CelebDF Synthetic Face)"]
    )

    # Initialize modules
    extractor = get_face_extractor()
    model, transform, device = load_detector_model(model_cfg["name"], ckpt_path)

    # Tabs
    tab1, tab2 = st.tabs(["📸 Image Detection", "🎥 Video Inspection"])

    # Tab 1: Image
    with tab1:
        st.subheader("Upload an Image for Analysis")
        uploaded_image = st.file_uploader("Select JPG, PNG, or WEBP image", type=["jpg", "jpeg", "png", "webp"])

        # Determine active image source
        img_pil = None
        source_desc = ""

        if uploaded_image is not None:
            img_pil = Image.open(uploaded_image).convert("RGB")
            source_desc = "Uploaded Custom Image"
        elif sample_choice == "Real Sample (CelebDF Real Face)":
            real_sample_path = "data/processed/celebdf/test/real/00170_f00000_face0.jpg"
            if os.path.exists(real_sample_path):
                img_pil = Image.open(real_sample_path).convert("RGB")
                source_desc = "CelebDF Authentic Subject (Expected: REAL)"
        elif sample_choice == "Fake Sample (CelebDF Synthetic Face)":
            fake_sample_path = "data/processed/celebdf/test/fake/id0_id16_0000_f00000_face0.jpg"
            if os.path.exists(fake_sample_path):
                img_pil = Image.open(fake_sample_path).convert("RGB")
                source_desc = "CelebDF Manipulated Face (Expected: FAKE)"

        if img_pil is not None:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(img_pil, caption=source_desc, use_container_width=True)

            with col2:
                if model is None or transform is None:
                    st.error("Please ensure the selected model checkpoint exists.")
                else:
                    with st.spinner("Detecting facial bounding box and evaluating forgery indicators..."):
                        # Score EVERY detected face; a frame with several
                        # people is no longer judged on an arbitrary one.
                        faces = extractor.extract_faces(img_pil)
                        if not faces:
                            st.warning(
                                "No face was detected with sufficient confidence, so no "
                                "verdict was produced. The detectors are trained on face "
                                "crops only, so scoring this frame would be meaningless."
                            )
                            st.stop()
                        face_crop = faces[0].image
                        caption = "Analyzed Facial Region"
                        if len(faces) > 1:
                            caption += f" (of {len(faces)} faces detected)"
                        st.image(face_crop, caption=caption, width=180)

                        tensor = transform(face_crop).unsqueeze(0).to(device)
                        with torch.no_grad():
                            outputs = model(tensor)
                            probs = torch.softmax(outputs, dim=1).squeeze(0)
                            prob_fake = float(probs[FAKE_CLASS_IDX].item())
                            prob_real = float(probs[REAL_CLASS_IDX].item())

                        is_fake = prob_fake >= threshold

                        # Grad-CAM: a score on its own is not actionable
                        # evidence. The overlay shows which pixels drove it.
                        if show_heatmap:
                            try:
                                overlay, _cam, _cls, _conf = generate_gradcam_for_model(
                                    model, model_cfg["name"], face_crop,
                                    target_class=FAKE_CLASS_IDX, device=device,
                                )
                                st.image(overlay, caption="Grad-CAM (FAKE class) - regions driving the score", width=180)
                            except Exception as e:
                                st.warning(f"Grad-CAM unavailable: {e}")

                        st.write("---")
                        st.write("### Detection Verdict")
                        
                        prob_pct = prob_fake * 100.0
                        if is_fake:
                            st.markdown(
                                f'<div class="verdict-box verdict-fake">'
                                f'🚨 <b>VERDICT: SYNTHETIC / DEEPFAKE DETECTED</b><br>'
                                f'Fake Probability: {prob_pct:.2f}% (Threshold: {threshold*100:.0f}%)<br>'
                                f'Authentic Probability: {prob_real*100:.2f}%'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                            st.caption(
                                "Threshold calibrated on validation data. Note the "
                                "measured false-positive rate on authentic faces "
                                "(see reports/calibration_summary.json) before "
                                "treating a verdict as evidence."
                            )
                        else:
                            st.markdown(
                                f'<div class="verdict-box verdict-real">'
                                f'✅ <b>VERDICT: AUTHENTIC / REAL MEDIA</b><br>'
                                f'Authentic Probability: {prob_real*100:.2f}%<br>'
                                f'Fake Probability: {prob_pct:.2f}% (Threshold: {threshold*100:.0f}%)'
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
                if model is None or transform is None:
                    st.error("Checkpoint required for inference.")
                else:
                    if st.button("🚀 Analyze Video Frames"):
                        with st.spinner("Sampling video frames and running neural verification..."):
                            cap = cv2.VideoCapture(video_path)
                            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            step = max(1, total_frames // 15)
                            
                            scores = []
                            current = 0

                            while cap.isOpened() and len(scores) < 15:
                                ret, frame = cap.read()
                                if not ret:
                                    break
                                if current % step == 0:
                                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                    frame_pil = Image.fromarray(frame_rgb)
                                    frame_faces = extractor.extract_faces(frame_pil)
                                    if not frame_faces:
                                        current += 1
                                        continue
                                    face_crop = frame_faces[0].image
                                    
                                    tensor = transform(face_crop).unsqueeze(0).to(device)
                                    with torch.no_grad():
                                        outputs = model(tensor)
                                        probs = torch.softmax(outputs, dim=1).squeeze(0)
                                        scores.append(float(probs[FAKE_CLASS_IDX].item()))
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
