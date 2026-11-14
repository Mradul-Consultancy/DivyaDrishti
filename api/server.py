"""
Production-facing HTTP API for deepfake detection.

Endpoints
---------
GET  /health                        service and model status
GET  /v1/models                     available checkpoints and their calibration
POST /v1/detect/image               single-image detection
POST /v1/detect/video               video analysis (synchronous, with an
                                    optional webhook for the async pattern)
GET  /v1/audit/recent               recent decision records (metadata only)
POST /v1/audit/purge                clears the in-memory decision buffer

Design notes
------------
* **No biometric content is persisted.** Uploads are read into memory,
  processed, and dropped. The audit trail keeps a SHA-256 digest of the
  input bytes, never the bytes themselves.
* **The threshold is calibrated, not assumed.** Each checkpoint carries a
  threshold measured on its validation split; a caller may override it
  explicitly, and the response always states which was used.
* **A frame with no detectable face gets no verdict.** The detectors are
  trained on face crops, so scoring a background would be meaningless.
"""

import base64
import io
import os
import sys
import time
import uuid
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from api import audit
from api.c2pa import inspect_bytes
from api.schemas import (
    DetectionResponse,
    FaceDetail,
    HealthResponse,
    TimelineHotspot,
    VideoJobRequest,
    VideoJobResponse,
    VideoResult,
)
from evaluation.calibration import load_threshold
from evaluation.evaluate import get_eval_transform
from evaluation.gradcam import generate_gradcam_for_model
from evaluation.spectral_combiner import combine, load_combiner, spectral_vector_for
from models.detector import get_model
from preprocessing.face_extractor import FaceExtractor, FaceNotFoundError

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "FastAPI is required for the API server: pip install fastapi uvicorn "
        f"({exc})"
    )

API_VERSION = "1.0.0"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

app = FastAPI(
    title="Deepfake Detection API",
    version=API_VERSION,
    description=(
        "Deepfake image and video detection with calibrated operating points, "
        "Grad-CAM explainability and C2PA provenance inspection."
    ),
)

# --------------------------------------------------------------------------
# Model registry
# --------------------------------------------------------------------------

MODELS = {
    "convnext_tiny.fb_in22k_ft_in1k": {
        "label": "ConvNeXt-Tiny",
        "file": "celebdf/convnext_tiny.fb_in22k_ft_in1k_best.pth",
        "domain": "celebdf",
    },
    "efficientnet_b3.ra2_in1k": {
        "label": "EfficientNet-B3",
        "file": "celebdf/efficientnet_b3.ra2_in1k_best.pth",
        "domain": "celebdf",
    },
    "resnet18": {
        "label": "ResNet-18",
        "file": "celebdf/resnet18_best.pth",
        "domain": "celebdf",
    },
}


class LoadedModel:
    """A model plus the pieces needed to score with it."""

    def __init__(self, name: str, cfg: dict):
        self.name = name
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_model(model_name=name, num_classes=config.NUM_CLASSES, pretrained=False)
        ckpt_path = config.CHECKPOINT_DIR / cfg["file"]
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
        state = {k.replace("module.", ""): v for k, v in state.items()}
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        # Transform must match training.
        self.transform = get_eval_transform(self.model, name)

    @property
    def threshold(self) -> float:
        return load_threshold(self.cfg["domain"], self.name)

    def score(self, face_img: Image.Image) -> tuple:
        tensor = self.transform(face_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1).squeeze(0)
        return float(probs[config.FAKE_LABEL_IDX].item()), float(probs[config.REAL_LABEL_IDX].item())


_loaded: dict = {}
_extractor = None


def get_loaded(name: str) -> LoadedModel:
    if name not in MODELS:
        raise HTTPException(status_code=404, detail=f"unknown model '{name}'")
    if name not in _loaded:
        cfg = MODELS[name]
        ckpt_path = config.CHECKPOINT_DIR / cfg["file"]
        if not ckpt_path.exists():
            raise HTTPException(
                status_code=503,
                detail=f"checkpoint missing: {cfg['file']}",
            )
        _loaded[name] = LoadedModel(name, cfg)
    return _loaded[name]


def get_extractor() -> FaceExtractor:
    global _extractor
    if _extractor is None:
        _extractor = FaceExtractor(margin_ratio=0.2, min_confidence=0.90)
    return _extractor


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health():
    available = [
        name for name, cfg in MODELS.items()
        if (config.CHECKPOINT_DIR / cfg["file"]).exists()
    ]
    calibrated = sorted({
        cfg["domain"] for name, cfg in MODELS.items()
        if (config.CHECKPOINT_DIR / cfg["domain"] / f"threshold_{name}.json").exists()
    })
    return HealthResponse(
        status="ok",
        models_available=available,
        calibrated_domains=calibrated,
        version=API_VERSION,
    )


@app.get("/v1/models")
def list_models():
    """Lists the checkpoints, their calibration and the measured trade-off."""
    out = []
    for name, cfg in MODELS.items():
        entry = {
            "name": name,
            "label": cfg["label"],
            "domain": cfg["domain"],
            "available": (config.CHECKPOINT_DIR / cfg["file"]).exists(),
            "threshold": load_threshold(cfg["domain"], name),
        }
        cal_path = config.CHECKPOINT_DIR / cfg["domain"] / f"threshold_{name}.json"
        if cal_path.exists():
            try:
                entry["calibration"] = json.loads(cal_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        out.append(entry)
    return {"models": out}


@app.post("/v1/detect/image", response_model=DetectionResponse)
async def detect_image(
    file: UploadFile = File(...),
    model: str = Form("convnext_tiny.fb_in22k_ft_in1k"),
    threshold: Optional[float] = Form(None),
    explain: bool = Form(False),
    use_spectral: bool = Form(False),
):
    """
    Scores an image.

    Every detected face is scored; the verdict comes from the most suspicious
    one. A frame with no acceptable face returns 422 rather than a verdict,
    because the detectors are trained on face crops only.
    """
    t0 = time.perf_counter()
    request_id = str(uuid.uuid4())
    data = await file.read()
    digest = audit.sha256_bytes(data)
    c2pa = inspect_bytes(data)

    try:
        pil_img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=415, detail="input is not a decodable image")

    loaded = get_loaded(model)
    if threshold is not None:
        thr = float(threshold)
        thr_source = "request"
    else:
        thr = loaded.threshold
        thr_source = "calibrated" if (config.CHECKPOINT_DIR / loaded.cfg["domain"] /
                                      f"threshold_{model}.json").exists() else "default"

    extractor = get_extractor()
    faces = extractor.extract_faces(pil_img)
    if not faces:
        audit.record(
            request_id=request_id, model=model, domain=loaded.cfg["domain"],
            threshold=thr, threshold_source=thr_source, verdict="no_verdict",
            fake_probability=0.0, face_count=0, input_sha256=digest,
            processing_time_ms=(time.perf_counter() - t0) * 1000,
            media_type="image",
            notes=["no face detected above the confidence threshold"],
        )
        raise HTTPException(
            status_code=422,
            detail=(
                "no face detected above the confidence threshold; the detectors "
                "are trained on face crops, so no verdict was produced"
            ),
        )

    details: List[FaceDetail] = []
    best = None
    for det in faces:
        fake_p, real_p = loaded.score(det.image)
        details.append(FaceDetail(
            box=list(det.box),
            detection_confidence=round(det.confidence, 4),
            fake_probability=round(fake_p, 6),
            real_probability=round(real_p, 6),
        ))
        if best is None or fake_p > best[0]:
            best = (fake_p, real_p, det)

    fake_p, real_p, best_det = best

    # Optional frequency-domain branch. The combiner is a logistic
    # regression fitted on the domain's validation split; it is only shipped
    # for domains where it was measured to help (see
    # reports/spectral_ablation_*.json).
    spectral_note = None
    if use_spectral:
        combiner = load_combiner(loaded.cfg["domain"])
        if combiner is None:
            spectral_note = "no spectral combiner fitted for this domain; neural score used"
        else:
            try:
                vec = spectral_vector_for(best_det.image)
                combined = combine(fake_p, vec, combiner)
                spectral_note = f"spectral-combined P(fake)={combined:.4f} (neural {fake_p:.4f})"
                fake_p = combined
                real_p = 1.0 - combined
            except Exception as exc:
                spectral_note = f"spectral branch unavailable: {exc}"

    verdict = "fake" if fake_p >= thr else "real"

    gradcam_b64 = None
    if explain:
        try:
            overlay, _cam, _cls, _conf = generate_gradcam_for_model(
                loaded.model, model, best_det.image,
                target_class=config.FAKE_LABEL_IDX, device=loaded.device,
            )
            buf = io.BytesIO()
            overlay.save(buf, format="PNG")
            gradcam_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception as exc:
            gradcam_b64 = None

    elapsed = (time.perf_counter() - t0) * 1000
    audit.record(
        request_id=request_id, model=model, domain=loaded.cfg["domain"],
        threshold=thr, threshold_source=thr_source, verdict=verdict,
        fake_probability=fake_p, face_count=len(faces), input_sha256=digest,
        processing_time_ms=elapsed, media_type="image",
        notes=(["c2pa manifest present, signature not verified"] if c2pa.present else []),
    )

    notes = []
    if spectral_note:
        notes.append(spectral_note)
    if c2pa.present:
        notes.append("c2pa manifest present, signature not verified")

    return DetectionResponse(
        verdict=verdict,
        fake_probability=round(fake_p, 6),
        real_probability=round(real_p, 6),
        confidence=round(max(fake_p, real_p), 6),
        threshold=thr,
        threshold_source=thr_source,
        face_count=len(faces),
        faces=details,
        gradcam_base64=gradcam_b64,
        processing_time_ms=round(elapsed, 3),
        model=model,
        input_sha256=digest,
        c2pa_present=c2pa.present,
        note="; ".join(notes) if notes else None,
    )


@app.post("/v1/detect/video", response_model=VideoResult)
async def detect_video(
    file: UploadFile = File(...),
    model: str = Form("convnext_tiny.fb_in22k_ft_in1k"),
    threshold: Optional[float] = Form(None),
    explain: bool = Form(False),
    max_frames: int = Form(60),
    frame_step: int = Form(4),
):
    """
    Scores a video by sampling frames.

    Beyond the mean score it reports a **temporal consistency** figure: deepfakes
    that flicker between frames produce unstable per-frame scores, and that
    instability is itself a forgery cue that a single-frame detector misses.
    """
    t0 = time.perf_counter()
    request_id = str(uuid.uuid4())
    data = await file.read()
    digest = audit.sha256_bytes(data)
    c2pa = inspect_bytes(data)

    loaded = get_loaded(model)
    thr = float(threshold) if threshold is not None else loaded.threshold

    try:
        import cv2
        cap = cv2.VideoCapture(io.BytesIO(data))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"video decoding unavailable: {exc}")

    extractor = get_extractor()
    scores: List[float] = []
    frame_index = 0
    try:
        while cap.isOpened() and len(scores) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % frame_step == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = extractor.extract_faces(Image.fromarray(rgb))
                if faces:
                    fake_p, _real_p = loaded.score(faces[0].image)
                    scores.append(fake_p)
            frame_index += 1
    finally:
        cap.release()

    if not scores:
        raise HTTPException(
            status_code=422,
            detail="no frame yielded an acceptable face; no verdict was produced",
        )

    arr = np.asarray(scores, dtype=float)
    mean_p = float(arr.mean())
    max_p = float(arr.max())
    # Consistency: 1 - normalised spread. A stable video scores close to 1.
    spread = float(arr.std() / max(arr.mean(), 1e-6))
    consistency = float(max(0.0, min(1.0, 1.0 - spread)))

    order = np.argsort(-arr)
    hotspots = [
        TimelineHotspot(
            frame_index=int(i * frame_step),
            timestamp_seconds=round(float(i * frame_step) / 30.0, 3),  # assume 30 fps
            fake_probability=round(float(arr[i]), 6),
        )
        for i in order[:10]
    ]

    verdict = "fake" if max_p >= thr else "real"
    elapsed = (time.perf_counter() - t0) * 1000
    audit.record(
        request_id=request_id, model=model, domain=loaded.cfg["domain"],
        threshold=thr, threshold_source="request" if threshold is not None else "calibrated",
        verdict=verdict, fake_probability=max_p, face_count=len(scores),
        input_sha256=digest, processing_time_ms=elapsed, media_type="video",
        notes=(["c2pa manifest present, signature not verified"] if c2pa.present else []),
    )

    return VideoResult(
        job_id=request_id,
        verdict=verdict,
        mean_fake_probability=round(mean_p, 6),
        max_fake_probability=round(max_p, 6),
        temporal_consistency=round(consistency, 6),
        frames_analyzed=len(scores),
        hotspots=hotspots,
        processing_time_ms=round(elapsed, 3),
        input_sha256=digest,
        model=model,
    )


@app.get("/v1/audit/recent")
def audit_recent(limit: int = 50):
    """Recent decision records - metadata only, never image content."""
    return {"records": audit.recent(limit=limit)}


@app.post("/v1/audit/purge")
def audit_purge():
    """Clears the in-memory decision buffer (the on-disk log is untouched)."""
    return {"cleared_records": audit.wipe_in_memory()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("DF_API_PORT", "8000")))
