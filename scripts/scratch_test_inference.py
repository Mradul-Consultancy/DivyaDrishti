import torch, sys, random
from pathlib import Path
from PIL import Image

sys.path.append('e:/deepfake')
import config
from models.detector import get_model
from preprocessing.face_extractor import FaceExtractor
from preprocessing.transforms import get_inference_transforms

# Alphabetical ImageFolder ordering: fake=0, real=1
FAKE_CLASS_IDX = 0
device = torch.device('cpu')
extractor = FaceExtractor(margin_ratio=0.2)
transform = get_inference_transforms(image_size=config.IMAGE_SIZE)

def load_model(name, ckpt_file):
    model = get_model(name, num_classes=2, pretrained=False)
    ckpt = torch.load(config.CHECKPOINT_DIR / ckpt_file, map_location=device)
    sd = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    return model

def infer(model, img_path):
    img = Image.open(img_path).convert('RGB')
    face = extractor.extract_face(img)
    t = transform(face).unsqueeze(0).to(device)
    with torch.no_grad():
        prob_fake = torch.softmax(model(t), dim=1)[0][FAKE_CLASS_IDX].item()
    verdict = 'FAKE' if prob_fake >= 0.5 else 'REAL'
    return prob_fake, verdict

print('Loading EfficientNet-B3...')
model = load_model('efficientnet_b3.ra2_in1k', 'efficientnet_b3.ra2_in1k_best.pth')

# Test on REAL CelebDF images
real_imgs = list(Path('data/processed/celebdf/test/real').glob('*.jpg'))[:5]
fake_imgs = list(Path('data/processed/celebdf/test/fake').glob('*.jpg'))[:5]
test_cases = [(str(p), 'REAL') for p in real_imgs] + [(str(p), 'FAKE') for p in fake_imgs]
random.shuffle(test_cases)

print()
print('=== EfficientNet-B3 on CelebDF Test Images ===')
print(f"{'File':<50} {'Expected':<8} {'FakeScore':>10}  {'Verdict':<8}  {'OK?'}")
print('-' * 92)
correct = 0
for path, expected in test_cases:
    prob, verdict = infer(model, path)
    ok = 'CORRECT' if verdict == expected else 'WRONG'
    if verdict == expected:
        correct += 1
    print(f"{Path(path).name:<50} {expected:<8} {prob:>10.4f}  {verdict:<8}  {ok}")

print()
print(f'Accuracy on CelebDF: {correct}/{len(test_cases)} ({correct/len(test_cases)*100:.1f}%)')
print()

# Check what score range is for fake vs real
real_scores = []
fake_scores = []
for path, expected in test_cases:
    prob, _ = infer(model, path)
    if expected == 'REAL':
        real_scores.append(prob)
    else:
        fake_scores.append(prob)

print(f'Avg fake_prob for REAL images: {sum(real_scores)/len(real_scores):.4f}')
print(f'Avg fake_prob for FAKE images: {sum(fake_scores)/len(fake_scores):.4f}')
print(f'(Higher score = more likely FAKE; threshold=0.50)')
