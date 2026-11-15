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
import json
import io
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

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
    VideoResult,
)
from evaluation.calibration import load_threshold
from evaluation.evaluate import get_eval_transform
from evaluation.gradcam import generate_gradcam_for_model
from evaluation.spectral_combiner import combine, load_combiner, spectral_vector_for
from models.detector import get_model
from preprocessing.face_extractor import FaceExtractor, FaceNotFoundError

try:
    from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "FastAPI is required for the API server: pip install fastapi uvicorn "
        f"({exc})"
    )

API_VERSION = "1.0.0"

# A caller-supplied threshold is clamped to this range: outside it a single
# parameter would decide every verdict.
MIN_THRESHOLD = 0.05
MAX_THRESHOLD = 0.99

# Maximum upload size. Larger bodies are rejected rather than buffered.
MAX_UPLOAD_BYTES = int(os.environ.get("DF_MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))

# Shared secret for the audit endpoints and for purging the decision log.
# Set DF_API_KEY to require it; when unset the service runs open, which is
# only appropriate behind an authenticating gateway.
API_KEY = os.environ.get("DF_API_KEY") or None
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _warn_if_open() -> None:
    """Loud, once-per-boot notice when the service has no API key configured."""
    if API_KEY is None:
        logging.getLogger(__name__).warning(
            "DF_API_KEY is NOT set: every /v1 endpoint is open to anyone with "
            "network access. Deploy behind an authenticating gateway "
            "(see deploy/nginx-divyadrishhti.conf) or set DF_API_KEY."
        )


@asynccontextmanager
async def lifespan(_: FastAPI):
    _warn_if_open()
    yield


app = FastAPI(
    title="Deepfake Detection API",
    version=API_VERSION,
    lifespan=lifespan,
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
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
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


async def _read_capped(file: UploadFile) -> bytes:
    """
    Reads an upload, refusing anything over MAX_UPLOAD_BYTES.

    Reading the whole body into memory without a cap lets one request
    exhaust the worker's RAM.
    """
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"upload exceeds the {MAX_UPLOAD_BYTES // (1024*1024)} MB limit",
        )
    return data


def require_api_key(x_api_key: Optional[str] = Header(None)) -> None:
    """
    Shared-secret check for every /v1 endpoint.

    When DF_API_KEY is unset the service runs open, which is only appropriate
    behind an authenticating gateway (the deployment configs under deploy/
    show one). As soon as DF_API_KEY is configured, EVERY /v1 endpoint -
    detection included - demands the key; only /health stays open so that
    load balancers can still run liveness probes. The audit endpoints are
    additionally dangerous (they expose and can purge the decision log), so
    they must never be deployed without the key or a gateway.
    """
    if API_KEY is None:
        return
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid or missing API key")


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
def list_models(x_api_key: Optional[str] = Header(None)):
    require_api_key(x_api_key)
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
    x_api_key: Optional[str] = Header(None),
):
    """
    Scores an image.

    Every detected face is scored; the verdict comes from the most suspicious
    one. A frame with no acceptable face returns 422 rather than a verdict,
    because the detectors are trained on face crops only.
    """
    require_api_key(x_api_key)
    t0 = time.perf_counter()
    request_id = str(uuid.uuid4())
    data = await _read_capped(file)
    digest = audit.sha256_bytes(data)
    c2pa = inspect_bytes(data)

    try:
        pil_img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=415, detail="input is not a decodable image")

    # Face detection and inference are CPU/GPU-bound. Running them inline in
    # an `async def` endpoint blocks the event loop, so every other request
    # waits for this one. They are pushed onto a worker thread instead.
    from starlette.concurrency import run_in_threadpool

    loaded = get_loaded(model)
    if threshold is not None:
        thr = float(threshold)
        # A caller must not be able to force a verdict: threshold=0 would
        # call every image fake, threshold=1 would call every image real.
        if not (MIN_THRESHOLD <= thr <= MAX_THRESHOLD):
            raise HTTPException(
                status_code=422,
                detail=f"threshold must be within [{MIN_THRESHOLD}, {MAX_THRESHOLD}]",
            )
        thr_source = "request"
    else:
        thr = loaded.threshold
        thr_source = "calibrated" if (config.CHECKPOINT_DIR / loaded.cfg["domain"] /
                                      f"threshold_{model}.json").exists() else "default"

    extractor = get_extractor()
    faces = await run_in_threadpool(extractor.extract_faces, pil_img)
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
    scored_faces = await run_in_threadpool(
        lambda: [(det, *loaded.score(det.image)) for det in faces]
    )
    for det, fake_p, real_p in scored_faces:
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
                # The combined score has its own distribution, so the neural
                # threshold does not apply to it.
                combined_threshold = combiner.get("chosen_threshold")
                if combined_threshold:
                    thr = float(combined_threshold)
                    thr_source = "calibrated (spectral-combined)"
            except Exception as exc:
                spectral_note = f"spectral branch unavailable: {exc}"

    verdict = "fake" if fake_p >= thr else "real"

    gradcam_b64 = None
    if explain:
        try:
            overlay, _cam, _cls, _conf = await run_in_threadpool(
                generate_gradcam_for_model,
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
    max_frames: int = Form(60, ge=1, le=600),
    frame_step: int = Form(4, ge=1, le=60),
    x_api_key: Optional[str] = Header(None),
):
    """
    Scores a video by sampling frames.

    Beyond the mean score it reports a **temporal consistency** figure: deepfakes
    that flicker between frames produce unstable per-frame scores, and that
    instability is itself a forgery cue that a single-frame detector misses.
    """
    require_api_key(x_api_key)
    t0 = time.perf_counter()
    request_id = str(uuid.uuid4())
    data = await _read_capped(file)
    digest = audit.sha256_bytes(data)
    c2pa = inspect_bytes(data)

    loaded = get_loaded(model)
    thr = float(threshold) if threshold is not None else loaded.threshold

    # OpenCV's VideoCapture only accepts a path, so the upload is spooled to a
    # temporary file. It is deleted in the finally block, and never persisted.
    import tempfile
    import cv2

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        cap = cv2.VideoCapture(tmp_path)
    except Exception as exc:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=f"video decoding unavailable: {exc}")

    extractor = get_extractor()
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0

    scores: List[float] = []
    # (source frame index, probability) - frames with no face are dropped, so
    # the index cannot be recovered from the position in `scores`.
    scored_frames: List[tuple] = []
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
                    best = max(
                        (loaded.score(d.image) for d in faces),
                        key=lambda s: s[0],
                    )
                    scores.append(best[0])
                    scored_frames.append((frame_index, best[0]))
            frame_index += 1
    finally:
        cap.release()
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

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
            frame_index=int(scored_frames[int(i)][0]),
            timestamp_seconds=round(scored_frames[int(i)][0] / fps, 3),
            fake_probability=round(float(scored_frames[int(i)][1]), 6),
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
def audit_recent(limit: int = 50, x_api_key: Optional[str] = Header(None)):
    """Recent decision records - metadata only, never image content."""
    require_api_key(x_api_key)
    return {"records": audit.recent(limit=limit)}


@app.post("/v1/audit/purge")
def audit_purge(x_api_key: Optional[str] = Header(None)):
    """
    Clears the in-memory decision buffer (the on-disk log is untouched).

    The purge itself is recorded, so the audit trail cannot be erased
    without leaving a trace.
    """
    require_api_key(x_api_key)
    cleared = audit.wipe_in_memory()
    audit.record(
        request_id="purge", model="-", domain=None, threshold=0.0,
        threshold_source="admin", verdict="audit_purged", fake_probability=0.0,
        face_count=0, input_sha256="-" * 64, processing_time_ms=0.0,
        media_type="admin", notes=[f"cleared {cleared} in-memory records"],
    )
    return {"cleared_records": cleared}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("DF_API_PORT", "8000")))
