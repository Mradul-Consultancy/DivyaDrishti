import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config, torch
from evaluation.predict import predict_single_image
from models.detector import get_model
from preprocessing.face_extractor import FaceExtractor
from preprocessing.transforms import get_inference_transforms

device = torch.device('cpu')
extractor = FaceExtractor(margin_ratio=0.2)
transform = get_inference_transforms(image_size=config.IMAGE_SIZE)

real_sample = 'data/processed/celebdf/test/real/00170_f00000_face0.jpg'
fake_sample = 'data/processed/celebdf/test/fake/id0_id16_0000_f00000_face0.jpg'

for model_name, ckpt_name in [
    ('resnet18', 'resnet18_best.pth'),
    ('convnext_tiny.fb_in22k_ft_in1k', 'convnext_tiny.fb_in22k_ft_in1k_best.pth'),
    ('efficientnet_b3.ra2_in1k', 'efficientnet_b3.ra2_in1k_best.pth')
]:
    model = get_model(model_name, num_classes=2, pretrained=False)
    ckpt = torch.load(config.CHECKPOINT_DIR / ckpt_name, map_location=device)
    sd = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()

    r_res = predict_single_image(real_sample, model, extractor, transform, device)
    f_res = predict_single_image(fake_sample, model, extractor, transform, device)

    print(f"=== {model_name} ===")
    print(f"  REAL sample -> Verdict: {r_res['prediction'].upper()} (P_Fake: {r_res['fake_probability']:.4f}, P_Real: {r_res['real_probability']:.4f})")
    print(f"  FAKE sample -> Verdict: {f_res['prediction'].upper()} (P_Fake: {f_res['fake_probability']:.4f}, P_Real: {f_res['real_probability']:.4f})")
