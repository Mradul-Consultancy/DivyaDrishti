"""
Pydantic request/response schemas for the detection API.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class FaceDetail(BaseModel):
    """One detected face and the score assigned to it."""

    box: List[int] = Field(..., description="Bounding box [x1, y1, x2, y2] in source pixels")
    detection_confidence: float = Field(..., ge=0.0, le=1.0,
                                        description="Confidence of the face detector, not the classifier")
    fake_probability: float = Field(..., ge=0.0, le=1.0)
    real_probability: float = Field(..., ge=0.0, le=1.0)


class DetectionResponse(BaseModel):
    """Result of a single-image detection."""

    verdict: str = Field(..., description="'fake' or 'real'")
    fake_probability: float = Field(..., ge=0.0, le=1.0)
    real_probability: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    threshold: float = Field(..., description="Decision threshold that was applied")
    threshold_source: str = Field(..., description="'calibrated', 'request' or 'default'")
    face_count: int = Field(..., ge=0, description="Faces detected in the frame")
    faces: List[FaceDetail] = Field(default_factory=list)
    gradcam_base64: Optional[str] = Field(
        None, description="PNG heatmap overlay, base64-encoded (requires explain=true)"
    )
    processing_time_ms: float = Field(..., description="Server-side processing time")
    model: str = Field(..., description="Backbone that produced the score")
    input_sha256: str = Field(..., description="SHA-256 of the input bytes - for audit, not the image")
    c2pa_present: Optional[bool] = Field(
        None, description="Whether the media carries a C2PA/Content-Credentials manifest"
    )
    note: Optional[str] = Field(
        None,
        description="Caveats attached to this result, e.g. no face detected so no verdict",
    )


class TimelineHotspot(BaseModel):
    """A stretch of the video where the forgery signal peaks."""

    frame_index: int
    timestamp_seconds: float
    fake_probability: float


class VideoResult(BaseModel):
    """Completed video analysis."""

    job_id: str
    verdict: str
    mean_fake_probability: float
    max_fake_probability: float
    temporal_consistency: float = Field(
        ..., ge=0.0, le=1.0,
        description="1.0 means the per-frame scores were stable; low values indicate "
                    "frame-to-frame flicker, a temporal forgery cue",
    )
    frames_analyzed: int
    hotspots: List[TimelineHotspot]
    processing_time_ms: float
    input_sha256: str
    model: str


class HealthResponse(BaseModel):
    status: str
    models_available: List[str]
    calibrated_domains: List[str]
    version: str
