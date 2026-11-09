"""
Inference script for Deepfake Detection.
Predicts single images or an entire directory, with face cropping and visual output.
"""

import sys
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import torch

# Add project root to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from models.detector import get_model
from preprocessing.face_extractor import FaceExtractor
from preprocessing.transforms import get_inference_transforms


def predict_single_image(
    image_path: Path | str,
    model: torch.nn.Module,
    extractor: FaceExtractor,
    transform,
    device: torch.device,
    crop_face: bool = True
):
    """
    Runs deepfake inference on an image.
    Returns:
        dict: {'prediction': 'real'|'fake', 'fake_probability': float, 'real_probability': float, 'face_image': PIL.Image}
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at {image_path}")

    pil_img = Image.open(image_path).convert("RGB")

    if crop_face:
        face_img = extractor.extract_face(pil_img)
    else:
        face_img = pil_img

    tensor = transform(face_img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).squeeze(0)

    fake_prob = probs[config.FAKE_LABEL_IDX].item()
    real_prob = probs[config.REAL_LABEL_IDX].item()
    pred_class = "fake" if fake_prob >= 0.5 else "real"
    confidence = max(real_prob, fake_prob)

    return {
        "image_name": image_path.name,
        "prediction": pred_class,
        "confidence": confidence,
        "real_probability": real_prob,
        "fake_probability": fake_prob,
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

    bg_color = (220, 20, 60) if pred == "FAKE" else (34, 139, 34)

    # Banner background
    banner_height = max(30, int(img.size[1] * 0.08))
    draw.rectangle([0, 0, img.size[0], banner_height], fill=bg_color)
    draw.text((10, banner_height // 4), label, fill=(255, 255, 255))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def run_inference(args):
    device = torch.device(args.device)
    print(f"Loading checkpoint {args.checkpoint} onto {device}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_name = checkpoint.get("model_name", args.model)
    model = get_model(model_name, num_classes=config.NUM_CLASSES, pretrained=False).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    extractor = FaceExtractor(margin_ratio=0.2)
    transform = get_inference_transforms()

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

    for img_path in targets:
        try:
            res = predict_single_image(
                image_path=img_path,
                model=model,
                extractor=extractor,
                transform=transform,
                device=device,
                crop_face=not args.no_crop
            )

            pred_tag = res["prediction"].upper()
            status_symbol = "[!]" if pred_tag == "FAKE" else "[+]"
            print(
                f"{status_symbol} {img_path.name:<25} -> {pred_tag:<5} | "
                f"Confidence: {res['confidence']*100:5.1f}% | "
                f"(Real: {res['real_probability']*100:4.1f}%, Fake: {res['fake_probability']*100:4.1f}%)"
            )

            if args.output_dir:
                out_file = Path(args.output_dir) / f"pred_{img_path.name}"
                save_annotated_prediction(res, out_file)

        except Exception as e:
            print(f"[Error] Failed on {img_path.name}: {e}")

    print("-" * 65)
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

    args = parser.parse_args()
    run_inference(args)

