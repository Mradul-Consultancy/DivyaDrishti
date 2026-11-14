"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for Deepfake Detection.

Supports:
- timm models: EfficientNet-B3, ConvNeXt-Tiny, ResNet, etc.
- Ultralytics YOLO26-cls models
- Overlay generation on original face images with colormaps
- Multi-model comparative Grad-CAM grids (Original | YOLO | ConvNeXt | EfficientNet)
"""

from pathlib import Path
from typing import Optional, Tuple, Dict, Union
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless backend for Kaggle / CI environments
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class GradCAM:
    """
    Computes Grad-CAM heatmaps for a given convolutional / feature target layer.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float]:
        """
        Generates a 2D normalized Grad-CAM heatmap for input_tensor.

        Args:
            input_tensor: Tensor of shape (1, C, H, W)
            target_class: Target class index (0 for fake, 1 for real). If None, uses argmax.

        Returns:
            Tuple of (heatmap: np.ndarray [H, W], predicted_class: int, confidence: float)
        """
        self.model.eval()
        self.activations = None
        self.gradients = None

        # Ensure gradient tracking
        input_tensor = input_tensor.clone().requires_grad_(True)
        logits = self.model(input_tensor)

        # Handle various output formats (e.g. YOLO returns tuple or tensor)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        probs = F.softmax(logits, dim=1)
        if target_class is None:
            target_class = int(torch.argmax(probs, dim=1).item())

        confidence = float(probs[0, target_class].item())

        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks failed to capture activations or gradients.")

        # Pool gradients across channels
        # activations shape: (1, C, H, W)
        # For ConvNeXt, activations can be channels-last (1, H, W, C) depending on layer
        grads = self.gradients
        acts = self.activations

        if grads.ndim == 4 and acts.ndim == 4:
            if grads.shape[1] == acts.shape[1] and grads.shape[1] > grads.shape[2]:
                # Standard (B, C, H, W)
                weights = torch.mean(grads, dim=(2, 3), keepdim=True)
                cam = torch.sum(weights * acts, dim=1).squeeze(0)
            elif grads.shape[-1] == acts.shape[-1]:
                # Channels-last (B, H, W, C)
                weights = torch.mean(grads, dim=(1, 2), keepdim=True)
                cam = torch.sum(weights * acts, dim=-1).squeeze(0)
            else:
                weights = torch.mean(grads, dim=(2, 3), keepdim=True)
                cam = torch.sum(weights * acts, dim=1).squeeze(0)
        else:
            cam = torch.sum(grads * acts, dim=1).squeeze(0)

        # ReLU on heatmap
        cam = F.relu(cam)
        cam_np = cam.detach().cpu().numpy()

        # Normalize to [0, 1]
        max_val = np.max(cam_np)
        if max_val > 1e-8:
            cam_np = (cam_np - np.min(cam_np)) / (max_val - np.min(cam_np))
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np, target_class, confidence


def get_default_target_layer(model: nn.Module, model_type: str = "timm") -> nn.Module:
    """
    Finds the optimal feature extraction layer for Grad-CAM across architectures.
    """
    model_name_lower = type(model).__name__.lower()

    # 1. Ultralytics YOLO ClassificationModel
    if hasattr(model, "model") and isinstance(model.model, nn.Sequential):
        # Look for the last feature block before the Classify head
        # In YOLOv11/26: model.model[9] is C2PSA, model.model[8] is C3k2
        if len(model.model) >= 10:
            return model.model[9]  # C2PSA attention block
        return model.model[-2]

    # 2. ConvNeXt
    if hasattr(model, "stages"):
        return model.stages[-1].blocks[-1]

    # 3. EfficientNet
    if hasattr(model, "conv_head"):
        return model.conv_head
    if hasattr(model, "blocks"):
        return model.blocks[-1][-1]

    # 4. ResNet
    if hasattr(model, "layer4"):
        return model.layer4[-1]

    # Fallback: find the last Conv2d in the network
    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is not None:
        return last_conv

    raise ValueError(f"Could not automatically detect target layer for model: {type(model).__name__}")


def overlay_cam_on_image(
    pil_image: Image.Image,
    cam_heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET
) -> Image.Image:
    """
    Overlays a Grad-CAM heatmap on top of a PIL image.
    """
    w, h = pil_image.size
    cam_resized = cv2.resize(cam_heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    orig_np = np.array(pil_image)
    overlay = np.uint8(alpha * heatmap_rgb + (1.0 - alpha) * orig_np)
    return Image.fromarray(overlay)


def generate_gradcam_for_model(
    model: nn.Module,
    model_name: str,
    pil_image: Image.Image,
    arch_type: str = "timm",
    target_class: int = config.FAKE_LABEL_IDX,
    device: Optional[torch.device] = None,
    alpha: float = 0.55,
) -> Tuple[Image.Image, np.ndarray, int, float]:
    """
    Generates a Grad-CAM overlay for a SINGLE model on an already-cropped face.

    Uses the model's own evaluation transform, so the explanation is computed
    on exactly the tensor the classifier saw. The older comparative helper
    hardcoded the torchvision normalisation, which does not match the timm
    backbones' native pipeline and therefore mislocated the activations.

    Returns:
        (overlay_image, heatmap, predicted_class, confidence)
    """
    device = device or config.DEVICE
    from evaluation.evaluate import get_eval_transform

    model = model.to(device)
    model.eval()

    transform = get_eval_transform(model, model_name)
    input_batch = transform(pil_image).unsqueeze(0).to(device)

    target_layer = get_default_target_layer(model, model_type=arch_type)
    cam_engine = GradCAM(model, target_layer)
    try:
        cam, pred_cls, conf = cam_engine.generate_heatmap(input_batch, target_class=target_class)
    finally:
        cam_engine.remove_hooks()

    overlay = overlay_cam_on_image(pil_image, cam, alpha=alpha)
    return overlay, cam, pred_cls, conf


def generate_comparative_gradcam(
    image_path: Union[str, Path],
    models_dict: Dict[str, Tuple[nn.Module, str]],
    output_path: Optional[Path] = None,
    target_class: int = config.FAKE_LABEL_IDX,
    device: Optional[torch.device] = None
) -> Image.Image:
    """
    Generates a comparative side-by-side Grad-CAM visualization for multiple models.

    Args:
        image_path: Path to face image
        models_dict: Dict of model_name -> (model_instance, arch_type)
        output_path: Path to save the final image
        target_class: Class to explain (default: 0 for fake)
        device: Torch device

    Returns:
        PIL Image of the comparative grid.
    """
    device = device or config.DEVICE
    image_path = Path(image_path)
    pil_img = Image.open(image_path).convert("RGB").resize(config.IMAGE_SIZE)

    # Base transforms
    mean = torch.tensor(config.NORMALIZE_MEAN).view(3, 1, 1)
    std = torch.tensor(config.NORMALIZE_STD).view(3, 1, 1)
    img_tensor = (torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).float() / 255.0 - mean) / std
    input_batch = img_tensor.unsqueeze(0).to(device)

    num_panels = 1 + len(models_dict)
    fig, axes = plt.subplots(1, num_panels, figsize=(4.2 * num_panels, 4.5))
    if num_panels == 1:
        axes = [axes]

    # Panel 0: Original
    axes[0].imshow(pil_img)
    axes[0].set_title(f"Input Face\n{image_path.name[:20]}", fontweight="bold", fontsize=11)
    axes[0].axis("off")

    for idx, (m_name, (model, arch_type)) in enumerate(models_dict.items(), start=1):
        model = model.to(device)
        model.eval()
        target_layer = get_default_target_layer(model, model_type=arch_type)
        cam_engine = GradCAM(model, target_layer)

        try:
            cam, pred_cls, conf = cam_engine.generate_heatmap(input_batch, target_class=target_class)
            overlay = overlay_cam_on_image(pil_img, cam, alpha=0.55)
            pred_tag = "FAKE" if pred_cls == config.FAKE_LABEL_IDX else "REAL"

            axes[idx].imshow(overlay)
            color = "red" if pred_tag == "FAKE" else "green"
            axes[idx].set_title(
                f"{m_name}\nPred: {pred_tag} ({conf*100:.1f}%)",
                fontweight="bold",
                fontsize=11,
                color=color
            )
        finally:
            cam_engine.remove_hooks()

        axes[idx].axis("off")

    plt.suptitle(
        f"Grad-CAM Heatmap Analysis (Explaining: {'FAKE' if target_class == config.FAKE_LABEL_IDX else 'REAL'})",
        fontweight="bold",
        fontsize=13,
        y=0.98
    )
    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=250, bbox_inches="tight")

    # Convert canvas to PIL Image
    fig.canvas.draw()
    rgba_bytes = fig.canvas.buffer_rgba()
    result_img = Image.frombuffer("RGBA", fig.canvas.get_width_height(), rgba_bytes, "raw", "RGBA", 0, 1)
    plt.close()

    return result_img

