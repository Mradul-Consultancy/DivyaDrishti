"""
Face extraction and alignment utility for raw image inputs.

Detects facial regions and crops them with a contextual margin, because
deepfake artifacts concentrate at the hair, jaw and ear boundaries.

Detection priority:
  1. MTCNN (facenet-pytorch) - the detector the training pipeline used.
  2. OpenCV Haar cascade - only when the installed OpenCV still provides
     ``CascadeClassifier`` *and* the cascade XML is present. OpenCV 5
     removed both, so this path is frequently unavailable.
  3. No detector - the caller is told no face was found.

Two behaviours were changed after an audit, because both silently corrupted
downstream scores:

* **All faces are returned, with confidence.** A frame containing several
  people (an interview, a news broadcast) previously had whichever face the
  detector happened to return first evaluated - an arbitrary choice.
* **A missing face is an explicit condition, not a silent centre crop.**
  The models are trained exclusively on tight face crops; feeding a whole
  1080p frame produces an arbitrary probability. Callers now receive
  ``face_count == 0`` and decide what to do, instead of getting a cropped
  background.
"""

import os
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    from facenet_pytorch import MTCNN
    MTCNN_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    MTCNN_AVAILABLE = False


class FaceNotFoundError(RuntimeError):
    """Raised when no face passes the confidence threshold."""


class FaceDetection(NamedTuple):
    """A single detected face and the crop taken from it."""

    image: Image.Image                       # cropped face, RGB
    box: Tuple[int, int, int, int]          # x1, y1, x2, y2 in source pixels
    confidence: float                       # detector confidence in [0, 1]

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.box
        return max(0, x2 - x1) * max(0, y2 - y1)


class FaceExtractor:
    """Detects and crops faces from images with an optional margin."""

    def __init__(
        self,
        margin_ratio: float = 0.2,
        detector: str = "auto",
        min_confidence: float = 0.90,
        allow_center_fallback: bool = False,
        device: Optional[str] = None,
    ):
        """
        Args:
            margin_ratio: Extra boundary padding around the detected face box
                          (e.g. 0.2 = 20%). Deepfake artifacts often appear at
                          hair and jaw boundaries.
            detector: "auto" (MTCNN, then Haar, then none), "mtcnn", "haar"
                      or "none".
            min_confidence: Minimum detector confidence for a face to count.
            allow_center_fallback: When True, a frame with no accepted face
                          falls back to a centre crop. Off by default: the
                          detectors are trained on face crops only, so a
                          centre crop of a full frame yields an arbitrary
                          score rather than a meaningful one.
            device: Torch device string for MTCNN ("cuda" / "cpu").
        """
        self.margin_ratio = margin_ratio
        self.min_confidence = min_confidence
        self.allow_center_fallback = allow_center_fallback
        self.device = device
        self.backend = "none"

        self.face_cascade = None
        self.mtcnn = None

        if detector in ("auto", "mtcnn") and MTCNN_AVAILABLE:
            try:
                import torch
                self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
                self.mtcnn = MTCNN(
                    image_size=224,
                    margin=0,          # proportional margin applied here instead
                    keep_all=True,
                    post_process=False,
                    device=self.device,
                )
                self.backend = "mtcnn"
            except Exception:
                self.mtcnn = None

        if self.backend == "none" and detector in ("auto", "haar"):
            self.face_cascade = self._build_haar_cascade()
            if self.face_cascade is not None:
                self.backend = "haar"

    @staticmethod
    def _build_haar_cascade():
        """
        Returns a Haar cascade, or None.

        OpenCV 5 dropped ``CascadeClassifier`` from the main module and no
        longer ships the cascade XML files, so this returns None there - which
        used to mean the extractor silently centre-cropped every image.
        """
        if not OPENCV_AVAILABLE:
            return None
        if not hasattr(cv2, "CascadeClassifier"):
            return None
        try:
            path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if not os.path.exists(path):
                return None
            return cv2.CascadeClassifier(path)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _to_pil(img_input) -> Image.Image:
        if isinstance(img_input, (str, Path)):
            return Image.open(img_input).convert("RGB")
        if isinstance(img_input, np.ndarray):
            arr = img_input
            # Heuristic: OpenCV arrays are BGR, PIL-backed arrays are RGB.
            return Image.fromarray(arr)
        if isinstance(img_input, Image.Image):
            return img_input.convert("RGB")
        raise ValueError(f"Unsupported image input type: {type(img_input)}")

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------
    def extract_faces(self, img_input) -> List[FaceDetection]:
        """
        Returns every face that passes the confidence threshold.

        Each entry carries the cropped image, the source bounding box and the
        detector confidence. An empty list means no acceptable face was found.
        """
        pil_img = self._to_pil(img_input)
        detections: List[FaceDetection] = []

        if self.mtcnn is not None:
            detections = self._detect_mtcnn(pil_img)
        elif self.face_cascade is not None:
            detections = self._detect_haar(pil_img)

        detections.sort(key=lambda d: d.area, reverse=True)
        return detections

    def _detect_mtcnn(self, pil_img: Image.Image) -> List[FaceDetection]:
        try:
            boxes, probs = self.mtcnn.detect(pil_img)
        except Exception:
            return []
        if boxes is None or len(boxes) == 0:
            return []
        out = []
        for box, prob in zip(boxes, probs):
            if prob is None or prob < self.min_confidence:
                continue
            out.append(self._crop(pil_img, box, float(prob)))
        return out

    def _detect_haar(self, pil_img: Image.Image) -> List[FaceDetection]:
        # Haar has no confidence score, so every detection is treated as
        # maximally confident; the confidence gate therefore cannot reject
        # a Haar detection. Prefer the "mtcnn" detector where it matters.
        cv_img = np.array(pil_img)
        try:
            gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )
        except Exception:
            return []
        return [self._crop(pil_img, box, 1.0) for box in faces]

    def _crop(self, pil_img: Image.Image, box, confidence: float) -> FaceDetection:
        x, y, w, h = [int(v) for v in box]
        margin_x = int(w * self.margin_ratio)
        margin_y = int(h * self.margin_ratio)
        img_w, img_h = pil_img.size
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(img_w, x + w + margin_x)
        y2 = min(img_h, y + h + margin_y)
        crop = pil_img.crop((x1, y1, x2, y2))
        return FaceDetection(image=crop, box=(x1, y1, x2, y2), confidence=confidence)

    # ------------------------------------------------------------------
    # Single-face convenience API (backwards compatible)
    # ------------------------------------------------------------------
    def extract_face(
        self,
        img_input,
        require_face: bool = True,
    ) -> Image.Image:
        """
        Extracts the largest detected face.

        With several people in the frame the largest face is chosen - which is
        a heuristic, not a decision about which person to judge. Callers that
        must not guess should use :meth:`extract_faces` and handle the list.

        Raises:
            FaceNotFoundError: no face passed the confidence threshold and
                ``require_face`` is True.
        """
        faces = self.extract_faces(img_input)
        if faces:
            return faces[0].image

        if self.allow_center_fallback:
            return self._fallback_crop(self._to_pil(img_input))

        if require_face:
            raise FaceNotFoundError(
                "no face detected above the confidence threshold "
                f"({self.min_confidence:.2f}); the model is trained on face "
                "crops, so scoring a full frame would be meaningless"
            )
        return self._fallback_crop(self._to_pil(img_input))

    def _fallback_crop(self, pil_img: Image.Image) -> Image.Image:
        """Centre crop to square. Only used when explicitly allowed."""
        w, h = pil_img.size
        min_dim = min(w, h)
        left = (w - min_dim) // 2
        top = (h - min_dim) // 2
        return pil_img.crop((left, top, left + min_dim, top + min_dim))

    def process_directory(self, input_dir: Path, output_dir: Path, target_size=(224, 224)):
        """
        Processes all images in input_dir and saves cropped faces into output_dir.

        Images with no acceptable face are reported and skipped rather than
        written out as a centre crop.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        image_files = [p for p in input_dir.rglob("*") if p.suffix.lower() in valid_exts]

        print(f"Extracting faces from {len(image_files)} images in {input_dir}...")
        print(f"Detector backend: {self.backend}")
        processed = 0
        skipped = 0
        for img_path in image_files:
            rel_path = img_path.relative_to(input_dir)
            out_path = output_dir / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                face = self.extract_face(img_path)
                face = face.resize(target_size, Image.Resampling.BILINEAR)
                face.save(out_path.with_suffix(".png"))
                processed += 1
            except FaceNotFoundError:
                skipped += 1
            except Exception as e:
                print(f"Skipping {img_path}: {e}")

        print(f"Successfully saved {processed} cropped faces to {output_dir}")
        if skipped:
            print(f"Skipped {skipped} image(s): no face above the confidence threshold")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Crop and extract faces from images")
    parser.add_argument("--input", type=str, required=True, help="Input directory of raw images")
    parser.add_argument("--output", type=str, required=True, help="Output directory for cropped faces")
    parser.add_argument("--margin", type=float, default=0.2, help="Contextual margin ratio (default: 0.2)")
    parser.add_argument("--detector", type=str, default="auto",
                        choices=["auto", "mtcnn", "haar", "none"], help="Face detector backend")
    parser.add_argument("--min-confidence", type=float, default=0.90,
                        help="Minimum detector confidence (default: 0.90)")
    args = parser.parse_args()

    ex = FaceExtractor(
        margin_ratio=args.margin,
        detector=args.detector,
        min_confidence=args.min_confidence,
    )
    ex.process_directory(Path(args.input), Path(args.output))
