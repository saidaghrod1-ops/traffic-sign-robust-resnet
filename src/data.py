"""GTSRB : téléchargement (torchvision), split train/val par track, DataLoaders, cache du test en tenseur.

Usage : python -m src.data --examples   (télécharge GTSRB, affiche les stats, sauvegarde results/gtsrb_examples.png)
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import GTSRB
from torchvision.transforms import v2

from .utils import resolve_path

NUM_CLASSES = 43

CLASS_NAMES = [
    "Limite 20", "Limite 30", "Limite 50", "Limite 60", "Limite 70", "Limite 80", "Fin limite 80",
    "Limite 100", "Limite 120", "Dépassement interdit", "Dépassement interdit >3,5t",
    "Priorité ponctuelle", "Route prioritaire", "Cédez le passage", "Stop", "Circulation interdite",
    "Interdit >3,5t", "Sens interdit", "Danger", "Virage à gauche", "Virage à droite", "Double virage",
    "Cassis", "Chaussée glissante", "Rétrécissement à droite", "Travaux", "Feux tricolores",
    "Passage piétons", "Enfants", "Cyclistes", "Neige / verglas", "Animaux sauvages",
    "Fin d'interdictions", "Tourner à droite", "Tourner à gauche", "Tout droit",
    "Tout droit ou à droite", "Tout droit ou à gauche", "Contournement par la droite",
    "Contournement par la gauche", "Giratoire", "Fin dépassement interdit", "Fin dépassement interdit >3,5t",
]


# --------------------------------------------------------------------------- transformations

def build_transform(img_size: int, train: bool, aug: dict | None = None) -> v2.Compose:
    """Redimensionnement + augmentations légères (rotation, ColorJitter) ; pas de flip horizontal."""
    t = [v2.ToImage(), v2.Resize((img_size, img_size), antialias=True)]
    if train:
        aug = aug or {}
        if aug.get("rotation"):
            t.append(v2.RandomRotation(aug["rotation"]))
        if aug.get("color_jitter"):
            t.append(v2.ColorJitter(**aug["color_jitter"]))
    t.append(v2.ToDtype(torch.float32, scale=True))
    return v2.Compose(t)


class TransformSubset(Dataset):
    """Sous-ensemble d'un dataset PIL avec sa propre transformation (train et val partagent la même base)."""

    def __init__(self, base: GTSRB, indices: list[int], transform):
        self.base, self.indices, self.transform = base, indices, transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i):
        img, label = self.base[self.indices[i]]
        return self.transform(img), label


# --------------------------------------------------------------------------- split par track

def track_id(path: str) -> str:
    """GTSRB nomme les images d'entraînement <track>_<frame>.ppm : un track = un panneau physique."""
    return Path(path).name.split("_")[0]


def track_split(samples: list[tuple[str, int]], val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Split stratifié par classe au niveau des tracks : aucun panneau physique à la fois en train et en val."""
    groups: dict[tuple[int, str], list[int]] = defaultdict(list)
    for i, (path, label) in enumerate(samples):
        groups[(label, track_id(path))].append(i)
    by_class: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for key in sorted(groups):
        by_class[key[0]].append(key)

    rng = random.Random(seed)
    train_idx, val_idx = [], []
    for label in sorted(by_class):
        keys = by_class[label]
        rng.shuffle(keys)
        n_val = max(1, round(len(keys) * val_fraction))
        for k in keys[:n_val]:
            val_idx += groups[k]
        for k in keys[n_val:]:
            train_idx += groups[k]

    train_tracks = {(samples[i][1], track_id(samples[i][0])) for i in train_idx}
    val_tracks = {(samples[i][1], track_id(samples[i][0])) for i in val_idx}
    assert not train_tracks & val_tracks, "fuite : un track est à la fois en train et en val"
    return sorted(train_idx), sorted(val_idx)


# --------------------------------------------------------------------------- DataLoaders

def get_dataloaders(cfg: dict, device: torch.device, seed: int):
    """Retourne (train_loader, val_loader, loader_generator, infos)."""
    dcfg = cfg["data"]
    root = resolve_path(dcfg["root"])
    base = GTSRB(root, split="train", download=True)
    train_idx, val_idx = track_split(base._samples, dcfg["val_fraction"], seed)

    subset = dcfg.get("subset")
    if subset:  # smoke test : sous-échantillon aléatoire, split par track conservé
        rng = random.Random(seed)
        train_idx = sorted(rng.sample(train_idx, min(subset, len(train_idx))))
        val_idx = sorted(rng.sample(val_idx, min(max(subset // 4, 64), len(val_idx))))

    size = dcfg["img_size"]
    train_ds = TransformSubset(base, train_idx, build_transform(size, train=True, aug=dcfg.get("augment")))
    val_ds = TransformSubset(base, val_idx, build_transform(size, train=False))

    nw = dcfg["num_workers"]
    common = dict(num_workers=nw, pin_memory=device.type == "cuda", persistent_workers=nw > 0)
    g = torch.Generator().manual_seed(seed)
    bs = cfg["train"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=len(train_ds) > bs,
                              generator=g, **common)
    val_loader = DataLoader(val_ds, batch_size=bs * 2, shuffle=False, **common)
    infos = {"n_train": len(train_ds), "n_val": len(val_ds),
             "n_train_tracks": len({(base._samples[i][1], track_id(base._samples[i][0])) for i in train_idx}),
             "n_val_tracks": len({(base._samples[i][1], track_id(base._samples[i][0])) for i in val_idx})}
    return train_loader, val_loader, g, infos


# --------------------------------------------------------------------------- test en tenseur (évaluation rapide)

def load_test_tensors(root: str | Path, img_size: int, max_samples: int | None = None,
                      num_workers: int = 2, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Jeu de test GTSRB (12 630 images) redimensionné, en uint8 (N,3,H,W), mis en cache sur disque.

    Les dégradations sont ensuite appliquées sur GPU, batch par batch, sans repasser par PIL.
    """
    root = resolve_path(root)
    cache = root / "cache" / f"gtsrb_test_{img_size}.pt"
    if cache.exists():
        d = torch.load(cache, weights_only=True)
        X, y = d["X"], d["y"]
    else:
        tf = v2.Compose([v2.ToImage(), v2.Resize((img_size, img_size), antialias=True)])
        ds = GTSRB(root, split="test", transform=tf, download=True)
        loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=num_workers)
        xs, ys = [], []
        for xb, yb in loader:
            xs.append(xb.as_subclass(torch.Tensor))
            ys.append(yb)
        X, y = torch.cat(xs), torch.cat(ys).long()
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"X": X, "y": y}, cache)
    if max_samples and max_samples < len(y):
        idx = torch.randperm(len(y), generator=torch.Generator().manual_seed(seed))[:max_samples].sort().values
        X, y = X[idx], y[idx]
    return X, y


# --------------------------------------------------------------------------- aperçu

def save_examples(root: Path, out_path: Path, n: int = 32, seed: int = 0) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = GTSRB(root, split="train", download=True)
    rng = random.Random(seed)
    idx = rng.sample(range(len(ds)), n)
    cols = 8
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.7, rows * 1.9))
    for ax, i in zip(axes.flat, idx):
        img, label = ds[i]
        ax.imshow(img)
        ax.set_title(f"{label} · {CLASS_NAMES[label]}", fontsize=6.5)
        ax.axis("off")
    for ax in list(axes.flat)[len(idx):]:
        ax.axis("off")
    fig.suptitle("GTSRB : exemples d'entraînement (taille d'origine)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Téléchargement et aperçu de GTSRB")
    p.add_argument("--data-root", default="data")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--img-size", type=int, default=64)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--examples", action="store_true", help="sauvegarde results/gtsrb_examples.png")
    args = p.parse_args()

    root = resolve_path(args.data_root)
    train = GTSRB(root, split="train", download=True)
    tr, va = track_split(train._samples, args.val_fraction, args.seed)
    n_tracks = len({(lbl, track_id(pth)) for pth, lbl in train._samples})
    X, y = load_test_tensors(root, args.img_size)
    print(f"GTSRB train : {len(train)} images, {n_tracks} tracks, {NUM_CLASSES} classes")
    print(f"  -> split par track : train {len(tr)} images / val {len(va)} images ({len(va) / len(train):.1%})")
    print(f"GTSRB test  : {len(y)} images, tenseur {tuple(X.shape)} {X.dtype} (cache {root / 'cache'})")
    if args.examples:
        out = resolve_path(args.results_dir)
        out.mkdir(parents=True, exist_ok=True)
        save_examples(root, out / "gtsrb_examples.png", seed=args.seed)
        print(f"Exemples : {out / 'gtsrb_examples.png'}")


if __name__ == "__main__":
    main()
