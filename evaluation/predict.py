"""
Inference script for Deepfake Detection.
Predicts single images or an entire directory, with face cropping and visual output.

Behavioural notes (from an enterprise-readiness audit):

* The preprocessing transform is derived from the **model**, so it matches
  training. timm backbones use their native bicubic pipeline; the generic
  torchvision resize degraded their scores.
* Every detected face is scored, with the detector's confidence, instead of
  silently judging whichever face happened to be returned first.
* A frame with no acceptable face is reported as such rather than scored as
  a background crop - the detectors are trained on face crops only.
* The decision threshold is calibrated per domain
  (models/checkpoints/<domain>/threshold_<model>.json). At the default 0.5
  threshold the detectors called a large share of authentic faces fake on
  the imbalanced Celeb-DF v2 split; the calibrated threshold trades
  detection rate for a much lower false-positive rate.
"""

import sys
import argparse
from pathlib import Path
from typing import List, Optional

from PIL import Image, ImageDraw, ImageFont
import torch

# Add project root to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from models.detector import get_model
from evaluation.evaluate import get_eval_transform
from evaluation.calibration import load_threshold
from preprocessing.face_extractor import FaceExtractor, FaceNotFoundError


def predict_single_image(
    image_path: Path | str,
    model: torch.nn.Module,
    extractor: FaceExtractor,
    transform,
    device: torch.device,
    crop_face: bool = True,
    threshold: float = 0.5,
    model_name: Optional[str] = None,
) -> dict:
    """
    Runs deepfake inference on an image.

    When the frame contains more than one face, every face is scored and the
    verdict is taken from the face with the highest fake probability.

    Returns:
        dict with 'prediction', 'fake_probability', 'real_probability',
        'face_count', 'faces' and the cropped/annotated images.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at {image_path}")

    pil_img = Image.open(image_path).convert("RGB")

    faces: List = []
    if crop_face:
        faces = extractor.extract_faces(pil_img)
        if not faces:
            raise FaceNotFoundError(
                "no face detected above the confidence threshold; the model "
                "is trained on face crops, so scoring this frame would be "
                "meaningless"
            )

    def _score(face_img):
        tensor = transform(face_img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1).squeeze(0)
        return float(probs[config.FAKE_LABEL_IDX].item()), float(probs[config.REAL_LABEL_IDX].item())

    if faces:
        scored = []
        for det in faces:
            fake_prob, real_prob = _score(det.image)
            scored.append((det, fake_prob, real_prob))
        # The verdict is driven by the most suspicious face.
        det, fake_prob, real_prob = max(scored, key=lambda s: s[1])
        face_img = det.image
        face_count = len(faces)
        face_details = [
            {
                "box": list(d.box),
                "confidence": round(d.confidence, 4),
                "fake_probability": round(fp, 6),
                "real_probability": round(rp, 6),
            }
            for d, fp, rp in scored
        ]
    else:
        fake_prob, real_prob = _score(pil_img)
        face_img = pil_img
        face_count = 0
        face_details = []

    pred_class = "fake" if fake_prob >= threshold else "real"
    confidence = max(real_prob, fake_prob)

    return {
        "image_name": image_path.name,
        "prediction": pred_class,
        "confidence": confidence,
        "real_probability": real_prob,
        "fake_probability": fake_prob,
        "threshold": threshold,
        "face_count": face_count,
        "faces": face_details,
        "face_image": face_img,
        "original_image": pil_img
    }


def save_annotated_prediction(result: dict, output_path: Path):
    """Draws prediction banner on top of the image and saves it."""
    img = result["original_image"].copy()
    draw = ImageDraw.Draw(img)

    pred = result["prediction"].upper()
    conf = result["confidence"] * 100
    label = f"[{pred}] Confidence: {conf:.1f}% (Fake Prob: {result['fake_probability']*100:.1f}%)"
    if result.get("face_count", 0) > 1:
        label += f" | Faces: {result['face_count']}"

    bg_color = (220, 20, 60) if pred == "FAKE" else (34, 139, 34)

    # Banner background
    banner_height = max(30, int(img.size[1] * 0.08))
    draw.rectangle([0, 0, img.size[0], banner_height], fill=bg_color)
    draw.text((10, banner_height // 4), label, fill=(255, 255, 255))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def _resolve_domain(checkpoint: Path) -> Optional[str]:
    """Infers the training domain from the checkpoint path, if any."""
    parts = {p.lower() for p in checkpoint.parts}
    for domain in ("celebdf", "dfdc"):
        if domain in parts:
            return domain
    return None


def run_inference(args):
    device = torch.device(args.device)
    print(f"Loading checkpoint {args.checkpoint} onto {device}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_name = checkpoint.get("model_name", args.model)
    model = get_model(model_name, num_classes=config.NUM_CLASSES, pretrained=False).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Transform must match training (native timm pipeline for timm backbones).
    transform = get_eval_transform(model, model_name)

    # Calibrated operating point, when one has been measured for this model.
    domain = _resolve_domain(Path(args.checkpoint))
    if args.threshold is not None:
        threshold = float(args.threshold)
        print(f"Using explicit threshold: {threshold}")
    else:
        threshold = load_threshold(domain, model_name) if domain else 0.5
        if domain:
            print(f"Calibrated threshold for {domain}/{model_name}: {threshold}")
        else:
            print(f"No calibration found for this checkpoint; using {threshold}")

    extractor = FaceExtractor(margin_ratio=args.margin, min_confidence=args.min_confidence)

    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    targets = []

    if args.image:
        targets.append(Path(args.image))
    elif args.input_dir:
        targets = [p for p in Path(args.input_dir).rglob("*") if p.suffix.lower() in valid_exts]
    else:
        print("[Error] Specify either --image <path> or --input-dir <folder>")
        return

    print(f"\nProcessing {len(targets)} image(s)...")
    print("-" * 65)

    no_face = 0
    for img_path in targets:
        try:
            res = predict_single_image(
                image_path=img_path,
                model=model,
                extractor=extractor,
                transform=transform,
                device=device,
                crop_face=not args.no_crop,
                threshold=threshold,
                model_name=model_name,
            )

            pred_tag = res["prediction"].upper()
            status_symbol = "[!]" if pred_tag == "FAKE" else "[+]"
            faces_info = f" | Faces: {res['face_count']}" if res["face_count"] != 1 else ""
            print(
                f"{status_symbol} {img_path.name:<25} -> {pred_tag:<5} | "
                f"Confidence: {res['confidence']*100:5.1f}% | "
                f"(Real: {res['real_probability']*100:4.1f}%, Fake: {res['fake_probability']*100:4.1f}%)"
                f"{faces_info}"
            )

            if args.output_dir:
                out_file = Path(args.output_dir) / f"pred_{img_path.name}"
                save_annotated_prediction(res, out_file)

        except FaceNotFoundError as e:
            no_face += 1
            print(f"[--] {img_path.name:<25} -> NO FACE | {e}")
        except Exception as e:
            print(f"[Error] Failed on {img_path.name}: {e}")

    print("-" * 65)
    if no_face:
        print(f"{no_face} image(s) had no acceptable face and were not scored.")
    if args.output_dir:
        print(f"Annotated predictions saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deepfake Image Detector Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint .pth")
    parser.add_argument("--model", type=str, default="resnet18", help="Model backbone architecture")
    parser.add_argument("--image", type=str, help="Path to single image for inference")
    parser.add_argument("--input-dir", type=str, help="Directory of images for batch inference")
    parser.add_argument("--output-dir", type=str, help="Directory to save visual predictions")
    parser.add_argument("--no-crop", action="store_true", help="Skip face extraction if input is already cropped face")
    parser.add_argument("--device", type=str, default=str(config.DEVICE), help="Device (cuda/cpu)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Decision threshold. Defaults to the calibrated value for this "
                             "checkpoint's domain, else 0.5.")
    parser.add_argument("--margin", type=float, default=0.2,
                        help="Contextual margin around the detected face (default: 0.2)")
    parser.add_argument("--min-confidence", type=float, default=0.90,
                        help="Minimum face-detector confidence (default: 0.90)")

    args = parser.parse_args()
    run_inference(args)
