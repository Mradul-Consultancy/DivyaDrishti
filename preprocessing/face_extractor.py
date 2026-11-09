"""
Face extraction and alignment utility for raw image inputs.
Extracts facial regions using OpenCV Haar Cascades with contextual margin,
falling back gracefully to center crop if no frontal face is detected.
"""

import os
from pathlib import Path
from PIL import Image
import numpy as np

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


class FaceExtractor:
    """Detects and crops faces from images with an optional margin."""

    def __init__(self, margin_ratio: float = 0.2):
        """
        Args:
            margin_ratio: Extra boundary padding around the detected face box (e.g. 0.2 = 20%).
                          Deepfake artifacts often appear at hair and jaw boundaries.
        """
        self.margin_ratio = margin_ratio
        self.face_cascade = None
        
        if OPENCV_AVAILABLE:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if os.path.exists(cascade_path):
                self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def extract_face(self, img_input: Image.Image | np.ndarray | str | Path) -> Image.Image:
        """
        Extracts the most prominent face from an input image.
        
        Args:
            img_input: PIL Image, numpy array (RGB/BGR), or file path.
            
        Returns:
            PIL.Image containing the cropped face, or center-cropped image if no face found.
        """
        # Load into PIL Image
        if isinstance(img_input, (str, Path)):
            pil_img = Image.open(img_input).convert("RGB")
        elif isinstance(img_input, np.ndarray):
            pil_img = Image.fromarray(img_input)
        elif isinstance(img_input, Image.Image):
            pil_img = img_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(img_input)}")

        if not OPENCV_AVAILABLE or self.face_cascade is None:
            return self._fallback_crop(pil_img)

        # Convert to grayscale for Haar cascade
        cv_img = np.array(pil_img)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        if len(faces) == 0:
            # Fall back to center crop if no face was found
            return self._fallback_crop(pil_img)

        # Pick largest face by area
        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])

        # Apply margin
        margin_x = int(w * self.margin_ratio)
        margin_y = int(h * self.margin_ratio)

        img_w, img_h = pil_img.size
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(img_w, x + w + margin_x)
        y2 = min(img_h, y + h + margin_y)

        return pil_img.crop((x1, y1, x2, y2))

    def _fallback_crop(self, pil_img: Image.Image) -> Image.Image:
        """Center crop to square if no face is detected."""
        w, h = pil_img.size
        min_dim = min(w, h)
        left = (w - min_dim) // 2
        top = (h - min_dim) // 2
        return pil_img.crop((left, top, left + min_dim, top + min_dim))

    def process_directory(self, input_dir: Path, output_dir: Path, target_size=(224, 224)):
        """
        Processes all images in input_dir and saves cropped faces into output_dir.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        image_files = [p for p in input_dir.rglob("*") if p.suffix.lower() in valid_exts]

        print(f"Extracting faces from {len(image_files)} images in {input_dir}...")
        processed = 0
        for img_path in image_files:
            rel_path = img_path.relative_to(input_dir)
            out_path = output_dir / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                face = self.extract_face(img_path)
                face = face.resize(target_size, Image.Resampling.BILINEAR)
                face.save(out_path.with_suffix(".png"))
                processed += 1
            except Exception as e:
                print(f"Skipping {img_path}: {e}")

        print(f"Successfully saved {processed} cropped faces to {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Crop and extract faces from images")
    parser.add_argument("--input", type=str, required=True, help="Input directory of raw images")
    parser.add_argument("--output", type=str, required=True, help="Output directory for cropped faces")
    parser.add_argument("--margin", type=float, default=0.2, help="Padding ratio around face box")
    args = parser.parse_args()

    extractor = FaceExtractor(margin_ratio=args.margin)
    extractor.process_directory(Path(args.input), Path(args.output))

