"""ResNet-18 / ResNet-50 pré-entraînés ImageNet, fc -> 43 classes, normalisation intégrée au modèle.

Le modèle prend des images dans [0,1] : les dégradations s'appliquent donc avant la normalisation,
dans l'espace des pixels, et le checkpoint est autonome (pas de Normalize à oublier à l'inférence).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

from .utils import load_checkpoint

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_BUILDERS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
}


class Normalize(nn.Module):
    def __init__(self, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class TrafficSignResNet(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.normalize = Normalize()
        self.backbone = backbone

    @property
    def target_layer(self) -> nn.Module:
        """Dernier étage convolutif, utilisé par Grad-CAM."""
        return self.backbone.layer4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.normalize(x))


def build_model(name: str = "resnet18", num_classes: int = 43, pretrained: bool = True,
                remove_maxpool: bool = True) -> TrafficSignResNet:
    """`remove_maxpool` : pour des entrées 64x64, layer4 sort en 4x4 au lieu de 2x2
    (meilleure résolution spatiale, cartes Grad-CAM plus lisibles). Les poids ImageNet restent valides."""
    if name not in _BUILDERS:
        raise ValueError(f"modèle inconnu : {name} (choix : {list(_BUILDERS)})")
    fn, weights = _BUILDERS[name]
    net = fn(weights=weights if pretrained else None)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    if remove_maxpool:
        net.maxpool = nn.Identity()
    return TrafficSignResNet(net)


def load_model(checkpoint_path: str | Path, device: torch.device) -> tuple[TrafficSignResNet, dict]:
    """Reconstruit le modèle depuis un checkpoint (best.pt ou last.pt) sans retélécharger les poids ImageNet."""
    ckpt = load_checkpoint(Path(checkpoint_path))
    mcfg = ckpt["config"]["model"]
    model = build_model(mcfg["name"], mcfg.get("num_classes", 43), pretrained=False,
                        remove_maxpool=mcfg.get("remove_maxpool", True))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt
