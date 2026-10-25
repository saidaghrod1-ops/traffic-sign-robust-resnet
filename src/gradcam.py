"""Grad-CAM (Selvaraju et al., 2017) implémenté par hooks sur layer4, sans librairie externe.

    alpha_k^c = moyenne spatiale de d y^c / d A^k        L^c = ReLU( sum_k alpha_k^c A^k )

Exemple :
    python -m src.gradcam --runs baseline robust --corruptions fog snow --severity 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from .corruptions import CORRUPTIONS, corrupt, corruption_seed, make_generator  # noqa: E402
from .data import CLASS_NAMES, load_test_tensors  # noqa: E402
from .models import load_model  # noqa: E402
from .utils import get_device, resolve_path  # noqa: E402

# Stop, cédez le passage, limite 30, route prioritaire, sens interdit, travaux, piétons, contournement droite
DEFAULT_CLASSES = [14, 13, 1, 12, 17, 25, 27, 38]


class GradCAM:
    """Hook avant sur la couche cible (activations A) + hook sur le tenseur de sortie (gradients dy/dA)."""

    def __init__(self, model: nn.Module, layer: nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        self._handle = layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output
        if output.requires_grad:
            output.register_hook(self._save_gradients)

    def _save_gradients(self, grad):
        self.gradients = grad

    def __call__(self, x: torch.Tensor, target: torch.Tensor | None = None):
        """Cartes (B,H,W) normalisées dans [0,1] et logits ; cible par défaut = classe prédite."""
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = self.model(x)
            if target is None:
                target = logits.argmax(1)
            logits.gather(1, target.view(-1, 1)).sum().backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
        cam = cam - cam.amin(dim=(1, 2), keepdim=True)
        cam = cam / cam.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
        return cam.detach(), logits.detach()

    def remove(self) -> None:
        self._handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()


def pick_images(y: torch.Tensor, n: int, seed: int = 0) -> torch.Tensor:
    """Premier exemple de chaque classe de DEFAULT_CLASSES, complété au hasard si besoin."""
    idx = [int((y == c).nonzero()[0]) for c in DEFAULT_CLASSES if (y == c).any()][:n]
    if len(idx) < n:
        rest = [i for i in torch.randperm(len(y), generator=torch.Generator().manual_seed(seed)).tolist()
                if i not in idx]
        idx += rest[:n - len(idx)]
    return torch.tensor(idx)


def gradcam_figure(models: dict, X: torch.Tensor, y: torch.Tensor, kind: str, severity: int,
                   device: torch.device, out: Path, seed: int = 0) -> None:
    x_clean = X.to(device).float() / 255
    x_corr = corrupt(x_clean, kind, severity, generator=make_generator(device, corruption_seed(kind, severity, seed)))
    cams = {}
    for run, model in models.items():
        with GradCAM(model, model.target_layer) as cam:
            cams[run] = {"clean": cam(x_clean), "corr": cam(x_corr)}

    runs = list(models)
    cols = ["propre"] + [f"CAM {r}" for r in runs] + [f"{kind} niv. {severity}"] + [f"CAM {r}" for r in runs]
    n = len(y)
    fig, axes = plt.subplots(n, len(cols), figsize=(1.9 * len(cols), 2.05 * n), squeeze=False)
    for i in range(n):
        for block, (images, key) in enumerate([(x_clean, "clean"), (x_corr, "corr")]):
            c0 = block * (1 + len(runs))
            img = images[i].permute(1, 2, 0).cpu().numpy()
            axes[i, c0].imshow(img)
            for j, run in enumerate(runs):
                heat, logits = cams[run][key]
                pred = int(logits[i].argmax())
                ax = axes[i, c0 + 1 + j]
                ax.imshow(img)
                ax.imshow(heat[i].cpu().numpy(), cmap="jet", alpha=0.45, vmin=0, vmax=1)
                ok = pred == int(y[i])
                ax.set_xlabel(f"{'✓' if ok else '✗'} {CLASS_NAMES[pred]}", fontsize=7,
                              color="green" if ok else "red")
        axes[i, 0].set_ylabel(CLASS_NAMES[int(y[i])], fontsize=8)
    for j, title in enumerate(cols):
        axes[0, j].set_title(title, fontsize=9)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Grad-CAM (layer4) - image propre vs {kind} niveau {severity}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Cartes Grad-CAM baseline vs robust, propre vs dégradé")
    p.add_argument("--runs", nargs="+", default=["baseline", "robust"])
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--corruptions", nargs="+", default=["fog", "snow"], choices=CORRUPTIONS)
    p.add_argument("--severity", type=int, default=4)
    p.add_argument("--num-images", type=int, default=6)
    p.add_argument("--data-root", default="data")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device")
    args = p.parse_args()

    device = get_device(args.device)
    results_dir = resolve_path(args.results_dir)
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    models, img_size = {}, None
    for run in args.runs:
        model, ckpt = load_model(results_dir / run / args.checkpoint, device)
        models[run], img_size = model, ckpt["config"]["data"]["img_size"]
    X, y = load_test_tensors(args.data_root, img_size)
    idx = pick_images(y, args.num_images, args.seed)
    for kind in args.corruptions:
        out = fig_dir / f"gradcam_{kind}_s{args.severity}.png"
        gradcam_figure(models, X[idx], y[idx], kind, args.severity, device, out, args.seed)
        print(f"Grad-CAM : {out}")


if __name__ == "__main__":
    main()
