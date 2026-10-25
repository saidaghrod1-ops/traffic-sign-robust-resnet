"""Utilitaires communs : seed, logging, configs, checkpoints, métriques, Hugging Face Hub."""

from __future__ import annotations

import csv
import logging
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- chemins / config

def resolve_path(p: str | os.PathLike) -> Path:
    """Chemin absolu tel quel, sinon relatif à la racine du dépôt."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def load_config(path: str | os.PathLike) -> dict:
    with open(resolve_path(path), encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(cfg: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


# --------------------------------------------------------------------------- reproductibilité

def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def get_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_rng_state() -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


# --------------------------------------------------------------------------- logging

def get_logger(name: str, log_file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file is not None:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


class CSVLogger:
    """Courbes d'entraînement en CSV ; à la reprise, supprime les époques non couvertes par le checkpoint."""

    FIELDS = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "corrupted_batches", "time_s"]

    def __init__(self, path: Path, resume_epoch: int = 0):
        self.path = path
        rows = []
        if resume_epoch > 0 and path.exists():
            with open(path, encoding="utf-8", newline="") as f:
                rows = [r for r in csv.DictReader(f) if int(r["epoch"]) <= resume_epoch]
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.FIELDS)
            w.writeheader()
            w.writerows(rows)

    def log(self, row: dict) -> None:
        with open(self.path, "a", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=self.FIELDS).writerow(row)


# --------------------------------------------------------------------------- checkpoints

def save_checkpoint(state: dict, path: Path) -> None:
    """Écriture atomique : une coupure de session pendant la sauvegarde ne corrompt pas l'ancien fichier."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path, map_location="cpu") -> dict:
    # Fichiers produits par ce dépôt (contiennent l'état RNG numpy) : weights_only=False.
    return torch.load(path, map_location=map_location, weights_only=False)


# --------------------------------------------------------------------------- métriques

class AverageMeter:
    def __init__(self):
        self.sum, self.count = 0.0, 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


def expected_calibration_error(conf: torch.Tensor, correct: torch.Tensor, n_bins: int = 15) -> float:
    """ECE = sum_b |B_b|/n * |acc(B_b) - conf(B_b)|, bins réguliers sur la confiance max."""
    conf, correct = conf.float().cpu(), correct.float().cpu()
    edges = torch.linspace(0, 1, n_bins + 1)
    ece = torch.zeros(())
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            ece += in_bin.float().mean() * (correct[in_bin].mean() - conf[in_bin].mean()).abs()
    return float(ece)


def confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> torch.Tensor:
    idx = y_true.cpu().long() * num_classes + y_pred.cpu().long()
    return torch.bincount(idx, minlength=num_classes ** 2).view(num_classes, num_classes)


# --------------------------------------------------------------------------- figures / markdown

def plot_training_curves(csv_paths: dict[str, Path], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.2))
    for i, (name, path) in enumerate(csv_paths.items()):
        if not Path(path).exists():
            continue
        df = pd.read_csv(path)
        color = f"C{i}"
        ax_loss.plot(df["epoch"], df["train_loss"], "--", color=color, label=f"{name} train")
        ax_loss.plot(df["epoch"], df["val_loss"], "-", color=color, label=f"{name} val")
        ax_acc.plot(df["epoch"], df["train_acc"], "--", color=color, label=f"{name} train")
        ax_acc.plot(df["epoch"], df["val_acc"], "-", color=color, label=f"{name} val")
    ax_loss.set(xlabel="Époque", ylabel="Loss (label smoothing 0.1)", title="Loss")
    ax_acc.set(xlabel="Époque", ylabel="Précision (%)", title="Précision")
    for ax in (ax_loss, ax_acc):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def df_to_markdown(df) -> str:
    """Tableau markdown sans dépendance à `tabulate`."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def update_readme_section(readme: Path, marker: str, content: str) -> bool:
    """Remplace le contenu entre <!-- MARKER:START --> et <!-- MARKER:END -->."""
    if not readme.exists():
        return False
    text = readme.read_text(encoding="utf-8")
    pattern = re.compile(rf"(<!-- {marker}:START -->)(.*?)(<!-- {marker}:END -->)", re.S)
    if not pattern.search(text):
        return False
    text = pattern.sub(lambda m: f"{m.group(1)}\n{content.strip()}\n{m.group(3)}", text)
    readme.write_text(text, encoding="utf-8")
    return True


# --------------------------------------------------------------------------- Hugging Face Hub (optionnel)
# Sans Google Drive, les fichiers de /content disparaissent si la VM Colab est recyclée.
# Si HF_TOKEN et HF_REPO_ID sont définis, les checkpoints sont aussi poussés sur un dépôt privé du Hub.

def hub_enabled() -> bool:
    return bool(os.environ.get("HF_TOKEN")) and bool(os.environ.get("HF_REPO_ID"))


def hub_upload(local_path: Path, path_in_repo: str) -> bool:
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ["HF_TOKEN"])
        repo_id = os.environ["HF_REPO_ID"]
        api.create_repo(repo_id, private=True, exist_ok=True)
        api.upload_file(path_or_fileobj=str(local_path), path_in_repo=path_in_repo, repo_id=repo_id)
        return True
    except Exception as e:  # le Hub ne doit jamais interrompre l'entraînement
        print(f"[hub] échec de l'upload de {local_path.name} : {e}")
        return False


def hub_download(path_in_repo: str, local_path: Path) -> bool:
    try:
        from huggingface_hub import hf_hub_download

        tmp = hf_hub_download(os.environ["HF_REPO_ID"], path_in_repo, token=os.environ["HF_TOKEN"])
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(Path(tmp).read_bytes())
        return True
    except Exception as e:
        print(f"[hub] {path_in_repo} indisponible : {e}")
        return False
