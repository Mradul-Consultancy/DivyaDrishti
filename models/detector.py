"""
Model architectures for Deepfake Detection:
1. MesoNet-4: A specialized compact architecture for facial forgery detection.
2. ResNet (ResNet-18 / ResNet-50): Transfer learning backbone for robust feature extraction.
3. EfficientNet-B0: High-accuracy scalable CNN backbone.
"""

import torch
import torch.nn as nn
from torchvision import models


class Meso4(nn.Module):
    """
    MesoNet-4 architecture: Compact neural network designed for
    detecting facial video/image manipulations at the mesoscopic scale.
    Reference: Afchar et al., 2018 (https://arxiv.org/abs/1809.00888)
    """

    def __init__(self, num_classes: int = 2):
        super(Meso4, self).__init__()

        # Conv Block 1
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv Block 2
        self.conv2 = nn.Conv2d(8, 8, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(8)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv Block 3
        self.conv3 = nn.Conv2d(8, 16, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(16)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv Block 4
        self.conv4 = nn.Conv2d(16, 16, kernel_size=5, padding=2)
        self.bn4 = nn.BatchNorm2d(16)
        self.pool4 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(16 * 4 * 4, 16)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Layer 1
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        # Layer 2
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        # Layer 3
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        # Layer 4
        x = self.pool4(self.relu(self.bn4(self.conv4(x))))

        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)

        x = self.dropout(x)
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def get_model(model_name: str = "resnet18", num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    Model factory to instantiate deepfake detection models.

    Args:
        model_name: One of ['mesonet', 'resnet18', 'resnet50', 'efficientnet_b0'].
        num_classes: Number of target classes (default: 2 -> 'real', 'fake').
        pretrained: Whether to use ImageNet pre-trained weights for transfer learning.
    """
    model_name = model_name.lower().strip()

    if model_name in ["mesonet", "meso4", "mesonet4"]:
        return Meso4(num_classes=num_classes)

    elif model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
        return model

    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, num_classes)
        )
        return model

    elif model_name in ["efficientnet", "efficientnet_b0"]:
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
        return model

    else:
        raise ValueError(
            f"Unsupported model architecture: '{model_name}'. "
            f"Choose from ['mesonet', 'resnet18', 'resnet50', 'efficientnet_b0']"
        )

