"""Adaptation des statistiques BatchNorm au test ("BN adapt", Schneider et al., 2020).

Pour chaque (type, niveau) : copie du modèle, remise à zéro de running_mean / running_var, couches BN
en mode train (tout le reste en eval), forward sans gradient sur des images dégradées - les labels ne
sont jamais utilisés -, puis réévaluation en mode eval sur le jeu de test dégradé.

Exemple :
    python -m src.bn_adapt --runs baseline robust --update-readme
"""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from .corruptions import CORRUPTIONS, SEVERITIES, corrupt, corruption_seed, make_generator  # noqa: E402
from .data import load_test_tensors  # noqa: E402
from .evaluate import predict  # noqa: E402
from .models import load_model  # noqa: E402
from .utils import df_to_markdown, get_device, resolve_path, update_readme_section  # noqa: E402


@torch.no_grad()
def adapt_bn(model: nn.Module, X_adapt: torch.Tensor, kind: str | None, severity: int, device: torch.device,
             batch_size: int = 256, seed: int = 0) -> nn.Module:
    """Retourne une copie du modèle dont les statistiques BN sont estimées sur X_adapt dégradé."""
    adapted = copy.deepcopy(model).eval()
    for m in adapted.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.reset_running_stats()
            m.momentum = None  # moyenne cumulative exacte sur tous les batchs d'adaptation
            m.train()
    # graine différente de l'évaluation : les images d'adaptation ne sont pas celles qui sont notées
    g = make_generator(device, corruption_seed(kind, severity, seed + 7919)) if kind else None
    for i in range(0, len(X_adapt), batch_size):
        xb = X_adapt[i:i + batch_size].to(device).float().div_(255)
        if kind:
            xb = corrupt(xb, kind, severity, generator=g)
        adapted(xb)
    return adapted.eval()


def accuracy(model, X, y, kind, severity, device, batch_size, seed) -> float:
    _, pred = predict(model, X, kind, severity, device, batch_size, seed)
    return 100 * (pred == y).float().mean().item()


def plot_bn(df: pd.DataFrame, runs, kinds, out: Path) -> None:
    groups = kinds + ["moyenne"]
    width = 0.8 / (2 * len(runs))
    fig, ax = plt.subplots(figsize=(max(8, 1.25 * len(groups)), 4.2))
    x = np.arange(len(groups))
    for i, run in enumerate(runs):
        d = df[(df.run == run) & (df.corruption != "clean")]
        sans = [d[d.corruption == k].acc_sans_adapt.mean() for k in kinds]
        avec = [d[d.corruption == k].acc_avec_adapt.mean() for k in kinds]
        sans.append(d.acc_sans_adapt.mean())
        avec.append(d.acc_avec_adapt.mean())
        ax.bar(x + (2 * i) * width - 0.4 + width / 2, sans, width, color=f"C{i}", alpha=0.45, label=f"{run}")
        ax.bar(x + (2 * i + 1) * width - 0.4 + width / 2, avec, width, color=f"C{i}", label=f"{run} + BN adapt")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=30, ha="right")
    ax.set_ylabel("Précision moyenne sur les niveaux (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Adaptation des statistiques BatchNorm au test")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, ncol=len(runs))
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Adaptation BN au test : sans vs avec")
    p.add_argument("--runs", nargs="+", default=["baseline", "robust"])
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--corruptions", nargs="+", default=CORRUPTIONS, choices=CORRUPTIONS)
    p.add_argument("--severities", nargs="+", type=int, default=SEVERITIES)
    p.add_argument("--adapt-samples", type=int, default=2048, help="images (non étiquetées) pour estimer les stats")
    p.add_argument("--adapt-batch-size", type=int, default=256)
    p.add_argument("--max-samples", type=int)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--data-root", default="data")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--readme", default="README.md")
    p.add_argument("--update-readme", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device")
    args = p.parse_args()

    device = get_device(args.device)
    results_dir = resolve_path(args.results_dir)
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    kinds, sevs = args.corruptions, sorted(args.severities)

    models = {run: load_model(results_dir / run / args.checkpoint, device) for run in args.runs}
    img_size = next(iter(models.values()))[1]["config"]["data"]["img_size"]
    X, y = load_test_tensors(args.data_root, img_size, args.max_samples)
    X = X.to(device)  # y reste sur CPU, comme les prédictions renvoyées par predict()
    perm = torch.randperm(len(y), generator=torch.Generator().manual_seed(args.seed + 1))[:args.adapt_samples]
    X_adapt = X[perm.to(device)]
    print(f"Test : {len(y)} images | adaptation sur {len(X_adapt)} images non étiquetées | device {device}")

    rows = []
    for run, (model, _) in models.items():
        print(f"  [{run}]")
        for kind, sev in [(None, 0)] + [(k, s) for k in kinds for s in sevs]:
            before = accuracy(model, X, y, kind, sev, device, args.batch_size, args.seed)
            adapted = adapt_bn(model, X_adapt, kind, sev, device, args.adapt_batch_size, args.seed)
            after = accuracy(adapted, X, y, kind, sev, device, args.batch_size, args.seed)
            del adapted
            rows.append({"run": run, "corruption": kind or "clean", "severity": sev,
                         "acc_sans_adapt": round(before, 2), "acc_avec_adapt": round(after, 2),
                         "gain": round(after - before, 2)})
            print(f"    {kind or 'clean':15s} niv. {sev} : {before:5.1f}% -> {after:5.1f}%")

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "bn_adapt_table.csv", index=False)
    plot_bn(df, args.runs, kinds, fig_dir / "bn_adapt.png")

    table = []
    for k in ["clean"] + kinds + ["moyenne dégradée"]:
        row = {"Dégradation": k}
        for run in args.runs:
            d = df[df.run == run]
            d = d[d.corruption != "clean"] if k == "moyenne dégradée" else d[d.corruption == k]
            row[f"{run}"] = f"{d.acc_sans_adapt.mean():.1f} %"
            row[f"{run} + BN"] = f"{d.acc_avec_adapt.mean():.1f} %"
        table.append(row)
    md = ("_Précision moyenne sur les niveaux ; statistiques BN réestimées sur "
          f"{len(X_adapt)} images dégradées non étiquetées (généré par `src/bn_adapt.py`)._\n\n"
          + df_to_markdown(pd.DataFrame(table)))
    (results_dir / "bn_adapt_table.md").write_text(md, encoding="utf-8")
    print("\n" + df_to_markdown(pd.DataFrame(table)) + "\n")
    print(f"Tableaux : {results_dir / 'bn_adapt_table.csv'} | figure : {fig_dir / 'bn_adapt.png'}")

    if args.update_readme:
        readme = resolve_path(args.readme)
        rel = Path(os.path.relpath(fig_dir, readme.parent)).as_posix()
        ok = update_readme_section(readme, "BN_ADAPT", md + f"\n\n![Adaptation BN]({rel}/bn_adapt.png)")
        print(f"README {'mis à jour' if ok else 'non trouvé ou sans marqueurs'} : {readme}")


if __name__ == "__main__":
    main()
