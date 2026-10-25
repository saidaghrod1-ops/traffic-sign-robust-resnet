"""Dégradations visuelles en PyTorch pur (CPU ou GPU), sur des batchs (B,3,H,W) dans [0,1], 5 niveaux.

Toutes les fonctions passent par corrupt(x, kind, severity) -> x clampé dans [0,1].
Les paramètres sont calibrés pour des images 64x64 ; les tailles spatiales sont mises à l'échelle
proportionnellement à H/64 pour d'autres résolutions.

Usage : python -m src.corruptions --grid   (sauvegarde results/corruption_grid.png)
"""

from __future__ import annotations

import argparse
import math
import zlib
from pathlib import Path

import torch
import torch.nn.functional as F

CORRUPTIONS = [
    "gaussian_noise", "gaussian_blur", "motion_blur", "fog", "rain",
    "snow", "dark", "overexposure", "occlusion",
]
SEVERITIES = [1, 2, 3, 4, 5]

# Un tuple par niveau 1..5 (voir README pour les formules).
PARAMS = {
    "gaussian_noise": [0.04, 0.08, 0.12, 0.17, 0.23],                     # sigma
    "gaussian_blur": [0.6, 0.9, 1.2, 1.6, 2.0],                           # sigma (px)
    "motion_blur": [3, 5, 7, 9, 12],                                      # longueur L (px)
    "fog": [0.30, 0.42, 0.54, 0.65, 0.75],                                # opacité t du voile
    "rain": [(0.010, 6, 0.45, 0.05), (0.015, 8, 0.55, 0.08), (0.022, 10, 0.62, 0.12),
             (0.030, 12, 0.70, 0.16), (0.040, 14, 0.78, 0.20)],           # (densité, L, opacité, assombr.)
    "snow": [(0.010, 0.6, 0.10), (0.018, 0.7, 0.18), (0.026, 0.8, 0.26),
             (0.035, 0.9, 0.34), (0.045, 1.0, 0.42)],                     # (densité, sigma flocon, blanchiment)
    "dark": [1.6, 2.2, 2.8, 3.5, 4.5],                                    # gamma
    "overexposure": [(1.40, 0.05), (1.75, 0.10), (2.10, 0.15), (2.60, 0.20), (3.20, 0.25)],  # (gain, offset)
    "occlusion": [0.15, 0.22, 0.30, 0.38, 0.46],                          # côté du carré / côté de l'image
}


# --------------------------------------------------------------------------- aléatoire reproductible

def make_generator(device: torch.device | str, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


def corruption_seed(kind: str, severity: int, base: int = 0) -> int:
    """Graine stable par (type, niveau) : baseline, robust et BN-adapt voient exactement les mêmes images."""
    return (zlib.crc32(kind.encode()) + 1000 * severity + base) % (2 ** 31)


def _rand(shape, x, g):
    return torch.rand(shape, device=x.device, dtype=x.dtype, generator=g)


def _randn(shape, x, g):
    return torch.randn(shape, device=x.device, dtype=x.dtype, generator=g)


# --------------------------------------------------------------------------- noyaux de convolution

def _gaussian_kernel1d(sigma: float, x: torch.Tensor) -> torch.Tensor:
    radius = max(1, math.ceil(3 * sigma))
    t = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-t ** 2 / (2 * sigma ** 2))
    return k / k.sum()


def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Flou gaussien séparable, convolution par canal (groups=C), bords en réflexion."""
    c = x.shape[1]
    k = _gaussian_kernel1d(sigma, x)
    r = k.numel() // 2
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)
    return x


def _line_kernels(length: float, angles: torch.Tensor, normalize: bool) -> torch.Tensor:
    """Un noyau (B,k,k) par image : segment anti-aliasé de longueur `length` orienté selon `angles` (rad)."""
    size = 2 * math.ceil(length / 2) + 1
    r = size // 2
    coords = torch.arange(size, device=angles.device, dtype=angles.dtype) - r
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    cos, sin = angles.cos().view(-1, 1, 1), angles.sin().view(-1, 1, 1)
    d_perp = (-sin * xx + cos * yy).abs()
    d_par = (cos * xx + sin * yy).abs()
    k = (1 - d_perp).clamp(min=0) * (d_par <= (length - 1) / 2 + 0.5)
    if normalize:
        k = k / k.sum(dim=(1, 2), keepdim=True)
    return k


def _per_image_conv(x: torch.Tensor, kernels: torch.Tensor) -> torch.Tensor:
    """Convolue chaque image b (tous ses canaux) avec son propre noyau kernels[b] via une conv groupée."""
    b, c, h, w = x.shape
    r = kernels.shape[-1] // 2
    weight = kernels.repeat_interleave(c, dim=0).unsqueeze(1)          # (B*C,1,k,k)
    out = F.conv2d(F.pad(x.reshape(1, b * c, h, w), (r, r, r, r), mode="reflect"), weight, groups=b * c)
    return out.view(b, c, h, w)


# --------------------------------------------------------------------------- dégradations

def gaussian_noise(x, p, s, g):
    return x + p * _randn(x.shape, x, g)


def gaussian_blur(x, p, s, g):
    return _gaussian_blur(x, p * s)


def motion_blur(x, p, s, g):
    angles = _rand((x.shape[0],), x, g) * math.pi                        # direction aléatoire par image
    return _per_image_conv(x, _line_kernels(p * s, angles, normalize=True))


def fog(x, p, s, g):
    b, _, h, w = x.shape
    # voile gris clair inhomogène : bruit basse fréquence (4x4 + 8x8) interpolé à la taille de l'image
    z = F.interpolate(_randn((b, 1, 4, 4), x, g), size=(h, w), mode="bicubic", align_corners=False)
    z = z + 0.5 * F.interpolate(_randn((b, 1, 8, 8), x, g), size=(h, w), mode="bicubic", align_corners=False)
    veil = (0.75 + 0.12 * z).clamp(0, 1)
    return (1 - p) * x + p * veil


def rain(x, p, s, g):
    density, length, opacity, dim = p
    b, _, h, w = x.shape
    drops = (_rand((b, 1, h, w), x, g) < density).to(x.dtype)
    angles = math.pi / 2 + (_rand((b,), x, g) - 0.5) * math.radians(30)  # quasi verticales, +-15 deg
    streaks = _per_image_conv(drops, _line_kernels(length * s, angles, normalize=False)).clamp(0, 1)
    x = x * (1 - dim)                                                    # ciel couvert
    return x * (1 - opacity * streaks) + opacity * streaks * 0.9


def snow(x, p, s, g):
    density, sigma, whiten = p
    b, _, h, w = x.shape
    gray = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
    x = (1 - whiten) * x + whiten * torch.maximum(x, gray * 1.5 + 0.5)   # paysage blanchi
    flakes = (_rand((b, 1, h, w), x, g) < density).to(x.dtype) * (0.6 + 0.4 * _rand((b, 1, h, w), x, g))
    flakes = (_gaussian_blur(flakes, sigma * s) * 2 * math.pi * (sigma * s) ** 2).clamp(0, 1)
    return 1 - (1 - x) * (1 - flakes)                                    # fusion "screen"


def dark(x, p, s, g):
    return x.clamp(min=0) ** p


def overexposure(x, p, s, g):
    gain, offset = p
    return gain * x + offset


def occlusion(x, p, s, g):
    b, c, h, w = x.shape
    side = max(1, round(p * min(h, w)))
    top = (_rand((b,), x, g) * (h - side + 1)).long().view(b, 1, 1, 1)
    left = (_rand((b,), x, g) * (w - side + 1)).long().view(b, 1, 1, 1)
    rows = torch.arange(h, device=x.device).view(1, 1, h, 1)
    cols = torch.arange(w, device=x.device).view(1, 1, 1, w)
    mask = (rows >= top) & (rows < top + side) & (cols >= left) & (cols < left + side)
    color = _rand((b, c, 1, 1), x, g)                                     # autocollant de couleur aléatoire
    return torch.where(mask, color, x)


_FUNCS = {
    "gaussian_noise": gaussian_noise, "gaussian_blur": gaussian_blur, "motion_blur": motion_blur,
    "fog": fog, "rain": rain, "snow": snow, "dark": dark, "overexposure": overexposure, "occlusion": occlusion,
}


def corrupt(x: torch.Tensor, kind: str, severity: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """Applique la dégradation `kind` au niveau `severity` (1..5 ; 0 = image intacte). Sortie clampée dans [0,1]."""
    if severity == 0:
        return x
    if kind not in _FUNCS:
        raise ValueError(f"dégradation inconnue : {kind} (choix : {CORRUPTIONS})")
    if not 1 <= severity <= 5:
        raise ValueError(f"niveau {severity} hors de [1,5]")
    squeeze = x.dim() == 3
    if squeeze:
        x = x.unsqueeze(0)
    scale = x.shape[-2] / 64
    out = _FUNCS[kind](x, PARAMS[kind][severity - 1], scale, generator).clamp(0, 1)
    return out.squeeze(0) if squeeze else out


# --------------------------------------------------------------------------- grille d'exemples

def save_corruption_grid(img: torch.Tensor, out_path: Path, kinds=CORRUPTIONS, severities=SEVERITIES,
                         seed: int = 0, holdout=()) -> None:
    """Grille type x niveau pour une image (3,H,W) dans [0,1]."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = [0] + list(severities)
    fig, axes = plt.subplots(len(kinds), len(cols), figsize=(1.45 * len(cols), 1.45 * len(kinds)))
    for i, kind in enumerate(kinds):
        g = make_generator(img.device, corruption_seed(kind, 0, seed))
        for j, sev in enumerate(cols):
            out = corrupt(img.unsqueeze(0), kind, sev, generator=g)[0]
            ax = axes[i, j]
            ax.imshow(out.permute(1, 2, 0).cpu().numpy(), interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title("propre" if sev == 0 else f"niveau {sev}", fontsize=9)
            if j == 0:
                ax.set_ylabel(kind + ("\n(non vue)" if kind in holdout else ""), fontsize=8)
    fig.suptitle("Dégradations (type x niveau)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    from .data import load_test_tensors
    from .utils import resolve_path

    p = argparse.ArgumentParser(description="Grille d'exemples des dégradations")
    p.add_argument("--grid", action="store_true")
    p.add_argument("--data-root", default="data")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--img-size", type=int, default=64)
    p.add_argument("--class-id", type=int, default=14, help="classe de l'image affichée (14 = stop)")
    p.add_argument("--holdout", nargs="*", default=["snow", "motion_blur"])
    args = p.parse_args()
    if not args.grid:
        p.print_help()
        return
    X, y = load_test_tensors(args.data_root, args.img_size)
    idx = (y == args.class_id).nonzero()
    img = X[int(idx[0]) if len(idx) else 0].float() / 255
    out = resolve_path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_corruption_grid(img, out / "corruption_grid.png", holdout=args.holdout)
    print(f"Grille des dégradations : {out / 'corruption_grid.png'}")


if __name__ == "__main__":
    main()
