"""Évaluation de la robustesse : Acc propre, Acc[type][niveau], CE / mCE, ECE, matrices de confusion, figures.

Exemple :
    python -m src.evaluate --runs baseline robust --baseline baseline --update-readme

Sorties : results/robustness_table.csv, results/robustness_table.md, results/figures/*.png,
          results/<run>/eval.json, section "Résultats" du README (avec --update-readme).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from .corruptions import CORRUPTIONS, SEVERITIES, corrupt, corruption_seed, make_generator  # noqa: E402
from .data import NUM_CLASSES, load_test_tensors  # noqa: E402
from .models import load_model  # noqa: E402
from .utils import (confusion_matrix, df_to_markdown, expected_calibration_error, get_device,  # noqa: E402
                    plot_training_curves, resolve_path, update_readme_section)


# --------------------------------------------------------------------------- inférence

@torch.no_grad()
def predict(model, X: torch.Tensor, kind: str | None, severity: int, device: torch.device,
            batch_size: int = 512, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """(confiance max, prédiction) sur X uint8 (N,3,H,W) dégradé par (kind, severity).

    Le générateur est réinitialisé par (type, niveau) : tous les modèles voient exactement les mêmes images.
    """
    model.eval()
    use_amp = device.type == "cuda"
    g = make_generator(device, corruption_seed(kind, severity, seed)) if kind else None
    confs, preds = [], []
    for i in range(0, len(X), batch_size):
        xb = X[i:i + batch_size].to(device, non_blocking=True).float().div_(255)
        if kind and severity:
            xb = corrupt(xb, kind, severity, generator=g)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(xb)
        conf, pred = logits.float().softmax(1).max(1)
        confs.append(conf.cpu())
        preds.append(pred.cpu())
    return torch.cat(confs), torch.cat(preds)


def evaluate_model(model, X, y, kinds, severities, device, batch_size=512, seed=0) -> dict:
    out = {"clean": {}, "corruptions": {k: {} for k in kinds}}
    conf, pred = predict(model, X, None, 0, device, batch_size, seed)
    correct = pred == y
    out["clean"] = {"acc": 100 * correct.float().mean().item(), "ece": 100 * expected_calibration_error(conf, correct)}
    out["cm_clean"] = confusion_matrix(y, pred, NUM_CLASSES)
    top = max(severities)
    top_true, top_pred = [], []
    for kind in kinds:
        for sev in severities:
            conf, pred = predict(model, X, kind, sev, device, batch_size, seed)
            correct = pred == y
            out["corruptions"][kind][sev] = {"acc": 100 * correct.float().mean().item(),
                                             "ece": 100 * expected_calibration_error(conf, correct)}
            if sev == top:
                top_true.append(y)
                top_pred.append(pred)
        print(f"    {kind:15s} " + " ".join(f"s{s}={out['corruptions'][kind][s]['acc']:5.1f}%" for s in severities))
    out["cm_top"] = confusion_matrix(torch.cat(top_true), torch.cat(top_pred), NUM_CLASSES)
    out["top_severity"] = top
    return out


# --------------------------------------------------------------------------- métriques agrégées

def corruption_errors(res: dict, base: dict, kinds, severities) -> dict[str, float]:
    """CE_c = sum_s E_{c,s}(modèle) / sum_s E_{c,s}(baseline), en %, avec E = 100 - Acc."""
    ce = {}
    for k in kinds:
        e_model = sum(100 - res["corruptions"][k][s]["acc"] for s in severities)
        e_base = sum(100 - base["corruptions"][k][s]["acc"] for s in severities)
        ce[k] = 100 * e_model / e_base if e_base > 0 else float("nan")
    return ce


def _nanmean(values) -> float:
    values = [v for v in values if not math.isnan(v)]
    return float(np.mean(values)) if values else float("nan")


def _fmt(v: float) -> str:
    return "–" if math.isnan(v) else f"{v:.1f} %"


# --------------------------------------------------------------------------- figures

def plot_acc_vs_severity(results: dict, kinds, severities, unseen, out: Path) -> None:
    ncols = min(3, len(kinds))
    nrows = math.ceil(len(kinds) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.2 * nrows), squeeze=False)
    xs = [0] + list(severities)
    for ax, kind in zip(axes.flat, kinds):
        for i, (run, res) in enumerate(results.items()):
            ys = [res["clean"]["acc"]] + [res["corruptions"][kind][s]["acc"] for s in severities]
            ax.plot(xs, ys, "o-", color=f"C{i}", label=run, ms=4)
        ax.set_title(kind + (" (non vue)" if kind in unseen else ""), fontsize=10)
        ax.set_xticks(xs)
        ax.set_xticklabels(["propre"] + [str(s) for s in severities], fontsize=8)
        ax.set_ylim(0, 101)
        ax.grid(alpha=0.3)
    for ax in list(axes.flat)[len(kinds):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=8)
    fig.supxlabel("Niveau de dégradation")
    fig.supylabel("Précision (%)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_heatmap(res: dict, run: str, kinds, severities, out: Path) -> None:
    mat = np.array([[res["clean"]["acc"]] + [res["corruptions"][k][s]["acc"] for s in severities] for k in kinds])
    fig, ax = plt.subplots(figsize=(1.1 * (len(severities) + 1) + 2.5, 0.5 * len(kinds) + 1.5))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(["propre"] + [f"niv. {s}" for s in severities])
    ax.set_yticks(range(len(kinds)))
    ax.set_yticklabels(kinds)
    ax.set_title(f"Précision (%) type x niveau - {run}")
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_confusion(cm: torch.Tensor, title: str, out: Path) -> None:
    cmn = (cm.float() / cm.sum(1, keepdim=True).clamp_min(1)).numpy()
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.tick_params(labelsize=6)
    ax.set_xlabel("Classe prédite")
    ax.set_ylabel("Classe réelle")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- tableaux

def build_tables(results: dict, configs: dict, baseline: str, kinds, severities):
    seen_by_robust = set()
    unseen = set()
    for cfg in configs.values():
        if cfg["robust"].get("enabled"):
            seen_by_robust |= set(cfg["robust"]["train_corruptions"])
            unseen |= set(cfg["robust"].get("holdout_corruptions", []))
    seen_kinds = [k for k in kinds if k in seen_by_robust]
    unseen_kinds = [k for k in kinds if k in unseen]

    long_rows, summary_rows, ce_all = [], [], {}
    for run, res in results.items():
        ce = corruption_errors(res, results[baseline], kinds, severities)
        ce_all[run] = ce
        rcfg = configs[run]["robust"]
        long_rows.append({"run": run, "corruption": "clean", "vue_entrainement": "",
                          **{f"acc_s{s}": "" for s in severities},
                          "acc_moy": round(res["clean"]["acc"], 2), "CE": ""})
        for k in kinds:
            accs = [res["corruptions"][k][s]["acc"] for s in severities]
            seen = rcfg.get("enabled") and k in rcfg["train_corruptions"]
            long_rows.append({"run": run, "corruption": k, "vue_entrainement": "oui" if seen else "non",
                              **{f"acc_s{s}": round(a, 2) for s, a in zip(severities, accs)},
                              "acc_moy": round(float(np.mean(accs)), 2), "CE": round(ce[k], 2)})
        corr_accs = [res["corruptions"][k][s]["acc"] for k in kinds for s in severities]
        corr_eces = [res["corruptions"][k][s]["ece"] for k in kinds for s in severities]
        summary_rows.append({
            "Modèle": run,
            "Acc. propre": _fmt(res["clean"]["acc"]),
            "Acc. moy. dégradée": _fmt(float(np.mean(corr_accs))),
            "mCE": _fmt(_nanmean(ce.values())),
            "mCE (vues)": _fmt(_nanmean([ce[k] for k in seen_kinds])) if seen_kinds else "–",
            "mCE (non vues)": _fmt(_nanmean([ce[k] for k in unseen_kinds])) if unseen_kinds else "–",
            "ECE propre": _fmt(res["clean"]["ece"]),
            "ECE dégradée": _fmt(float(np.mean(corr_eces))),
        })

    per_type = []
    for k in kinds:
        row = {"Dégradation": k, "Vue par robust": "oui" if k in seen_by_robust else "non"}
        for run, res in results.items():
            row[f"Acc {run}"] = _fmt(float(np.mean([res["corruptions"][k][s]["acc"] for s in severities])))
            if run != baseline:
                row[f"CE {run}"] = _fmt(ce_all[run][k])
        per_type.append(row)
    return pd.DataFrame(long_rows), pd.DataFrame(summary_rows), pd.DataFrame(per_type), unseen


def per_level_markdown(res: dict, run: str, kinds, severities) -> str:
    rows = [{"Dégradation": k, **{f"niv. {s}": f"{res['corruptions'][k][s]['acc']:.1f}" for s in severities}}
            for k in kinds]
    return f"**{run}** (propre : {res['clean']['acc']:.1f} %)\n\n" + df_to_markdown(pd.DataFrame(rows))


# --------------------------------------------------------------------------- main

def main() -> None:
    p = argparse.ArgumentParser(description="Évaluation de la robustesse aux dégradations")
    p.add_argument("--runs", nargs="+", default=["baseline", "robust"])
    p.add_argument("--baseline", default=None, help="run de référence pour CE / mCE (défaut : premier run)")
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--corruptions", nargs="+", default=CORRUPTIONS, choices=CORRUPTIONS)
    p.add_argument("--severities", nargs="+", type=int, default=SEVERITIES)
    p.add_argument("--max-samples", type=int, help="sous-échantillon du test (smoke test)")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--data-root", default="data")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--readme", default="README.md")
    p.add_argument("--update-readme", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device")
    args = p.parse_args()

    baseline = args.baseline or args.runs[0]
    if baseline not in args.runs:
        raise ValueError(f"--baseline {baseline} doit faire partie de --runs")
    kinds, sevs = args.corruptions, sorted(args.severities)
    device = get_device(args.device)
    results_dir = resolve_path(args.results_dir)
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    models, configs = {}, {}
    for run in args.runs:
        model, ckpt = load_model(results_dir / run / args.checkpoint, device)
        models[run], configs[run] = model, ckpt["config"]
    img_sizes = {c["data"]["img_size"] for c in configs.values()}
    if len(img_sizes) != 1:
        raise ValueError(f"tailles d'image différentes entre runs : {img_sizes}")
    X, y = load_test_tensors(args.data_root, img_sizes.pop(), args.max_samples)
    X = X.to(device)
    print(f"Test : {len(y)} images | dégradations {kinds} | niveaux {sevs} | device {device}")

    results = {}
    for run, model in models.items():
        print(f"  [{run}]")
        res = evaluate_model(model, X, y, kinds, sevs, device, args.batch_size, args.seed)
        results[run] = res
        plot_confusion(res["cm_clean"], f"{run} - images propres (acc {res['clean']['acc']:.1f} %)",
                       fig_dir / f"confusion_{run}_clean.png")
        plot_confusion(res["cm_top"], f"{run} - niveau {res['top_severity']}, toutes dégradations",
                       fig_dir / f"confusion_{run}_s{res['top_severity']}.png")
        plot_heatmap(res, run, kinds, sevs, fig_dir / f"heatmap_{run}.png")
        with open(results_dir / run / "eval.json", "w", encoding="utf-8") as f:
            json.dump({"clean": res["clean"], "corruptions": res["corruptions"], "n_test": len(y)}, f, indent=2)

    long_df, summary_df, per_type_df, unseen = build_tables(results, configs, baseline, kinds, sevs)
    long_df.to_csv(results_dir / "robustness_table.csv", index=False)
    plot_acc_vs_severity(results, kinds, sevs, unseen, fig_dir / "acc_vs_severity.png")
    plot_training_curves({run: results_dir / run / "metrics.csv" for run in args.runs},
                         fig_dir / "training_curves.png")

    head = [f"_Généré par `src/evaluate.py` le {datetime.now():%Y-%m-%d %H:%M} — GTSRB test, {len(y)} images, "
            f"{len(kinds)} dégradations x {len(sevs)} niveaux. CE et mCE relatifs à `{baseline}` (100 %)._",
            "", "#### Résumé", "", df_to_markdown(summary_df),
            "", "#### Par type de dégradation (précision moyenne sur les niveaux)", "", df_to_markdown(per_type_df)]
    levels = ["", "#### Précision par niveau", ""]
    for run, res in results.items():
        levels += [per_level_markdown(res, run, kinds, sevs), ""]
    (results_dir / "robustness_table.md").write_text("\n".join(head + levels), encoding="utf-8")

    print("\n" + df_to_markdown(summary_df) + "\n")
    print(f"Tableaux : {results_dir / 'robustness_table.csv'} | {results_dir / 'robustness_table.md'}")
    print(f"Figures  : {fig_dir}")

    if args.update_readme:
        readme = resolve_path(args.readme)
        rel = Path(os.path.relpath(fig_dir, readme.parent)).as_posix()
        block = "\n".join(head) + "\n\n" + "\n".join([
            f"![Précision en fonction du niveau]({rel}/acc_vs_severity.png)",
            "",
            f"![Courbes d'entraînement]({rel}/training_curves.png)",
        ])
        ok = update_readme_section(readme, "RESULTS", block)
        print(f"README {'mis à jour' if ok else 'non trouvé ou sans marqueurs'} : {readme}")


if __name__ == "__main__":
    main()
