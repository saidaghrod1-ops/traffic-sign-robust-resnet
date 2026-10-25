"""Entraînement baseline / robust.

Exemples :
    python -m src.train --config configs/baseline.yaml
    python -m src.train --config configs/robust.yaml --resume          # reprend depuis results/robust/last.pt
    python -m src.train --config configs/robust.yaml --model resnet50  # run "robust_resnet50"
"""

from __future__ import annotations

import argparse
import random
import time

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR

from .corruptions import CORRUPTIONS, corrupt
from .data import get_dataloaders
from .models import build_model
from .utils import (AverageMeter, CSVLogger, get_device, get_logger, get_rng_state, hub_download, hub_enabled,
                    hub_upload, load_checkpoint, load_config, plot_training_curves, resolve_path, save_checkpoint,
                    save_config, set_rng_state, set_seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entraînement ResNet sur GTSRB")
    p.add_argument("--config", required=True, help="configs/baseline.yaml ou configs/robust.yaml")
    p.add_argument("--resume", nargs="?", const="auto", default=None,
                   help="reprend depuis results/<run>/last.pt (ou le chemin donné) ; sans checkpoint, démarre à zéro")
    p.add_argument("--epochs", type=int)
    p.add_argument("--model", choices=["resnet18", "resnet50"])
    p.add_argument("--run-name")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--subset", type=int, help="limite le nombre d'images d'entraînement (smoke test)")
    p.add_argument("--num-workers", type=int)
    p.add_argument("--data-root")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--device", help="cuda / cpu (défaut : cuda si disponible)")
    return p.parse_args()


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    if args.model and args.model != cfg["model"]["name"]:
        cfg["model"]["name"] = args.model
        cfg["run_name"] = f"{cfg['run_name']}_{args.model}"
    if args.run_name:
        cfg["run_name"] = args.run_name
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["train"]["batch_size"] = args.batch_size
    if args.subset:
        cfg["data"]["subset"] = args.subset
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    return cfg


def check_robust_config(rcfg: dict) -> None:
    if not rcfg.get("enabled"):
        return
    unknown = set(rcfg["train_corruptions"]) - set(CORRUPTIONS)
    leak = set(rcfg["train_corruptions"]) & set(rcfg.get("holdout_corruptions", []))
    if unknown:
        raise ValueError(f"dégradations inconnues : {unknown}")
    if leak:
        raise ValueError(f"{leak} sont à la fois dans train_corruptions et holdout_corruptions")


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, cfg, use_amp, logger, epoch):
    model.train()
    rcfg = cfg["robust"]
    loss_m, correct, seen, n_corrupted = AverageMeter(), 0, 0, 0
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if rcfg.get("enabled") and random.random() < rcfg["p"]:
            kind = random.choice(rcfg["train_corruptions"])
            severity = random.choice(rcfg["severities"])
            x = corrupt(x, kind, severity)
            n_corrupted += 1

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        loss_m.update(loss.item(), y.size(0))
        correct += (logits.argmax(1) == y).sum().item()
        seen += y.size(0)
        if (i + 1) % cfg["train"]["log_every"] == 0:
            logger.info(f"  époque {epoch + 1} | batch {i + 1}/{len(loader)} | loss {loss_m.avg:.4f} "
                        f"| acc {100 * correct / seen:.2f}%")
    return loss_m.avg, 100 * correct / seen, n_corrupted


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp):
    model.eval()
    loss_m, correct, seen = AverageMeter(), 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)
        loss_m.update(loss.item(), y.size(0))
        correct += (logits.argmax(1) == y).sum().item()
        seen += y.size(0)
    return loss_m.avg, 100 * correct / seen


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args)
    run_name = cfg["run_name"]
    run_dir = resolve_path(args.results_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("train", run_dir / "train.log")
    device = get_device(args.device)

    # ----------------------------------------------------------------- reprise éventuelle
    ckpt = None
    if args.resume:
        path = run_dir / "last.pt" if args.resume == "auto" else resolve_path(args.resume)
        if not path.exists() and hub_enabled():
            logger.info("Checkpoint local absent : tentative de récupération depuis le Hugging Face Hub")
            if hub_download(f"{run_name}/last.pt", path):
                hub_download(f"{run_name}/metrics.csv", run_dir / "metrics.csv")
                hub_download(f"{run_name}/best.pt", run_dir / "best.pt")
        if path.exists():
            ckpt = load_checkpoint(path)
            logger.info(f"Reprise depuis {path} (époque {ckpt['epoch']} terminée)")
        else:
            logger.info(f"Aucun checkpoint trouvé ({path}) : entraînement depuis le début")
    if ckpt is not None:
        saved = ckpt["config"]
        if saved["train"]["epochs"] != cfg["train"]["epochs"]:
            logger.warning(f"--epochs ignoré à la reprise : le planning OneCycle du checkpoint "
                           f"({saved['train']['epochs']} époques) est conservé")
        saved["data"]["root"], saved["data"]["num_workers"] = cfg["data"]["root"], cfg["data"]["num_workers"]
        cfg = saved

    check_robust_config(cfg["robust"])
    set_seed(cfg["seed"])
    save_config(cfg, run_dir / "config.yaml")
    tcfg = cfg["train"]
    epochs = tcfg["epochs"]

    # ----------------------------------------------------------------- données, modèle, optimisation
    train_loader, val_loader, loader_gen, infos = get_dataloaders(cfg, device, cfg["seed"])
    logger.info(f"Run '{run_name}' | device {device}"
                + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    logger.info(f"Train {infos['n_train']} images ({infos['n_train_tracks']} tracks) | "
                f"val {infos['n_val']} images ({infos['n_val_tracks']} tracks) | {len(train_loader)} batches/époque")
    if cfg["robust"].get("enabled"):
        logger.info(f"Mode robust : p={cfg['robust']['p']}, dégradations {cfg['robust']['train_corruptions']}, "
                    f"non vues {cfg['robust'].get('holdout_corruptions', [])}")

    mcfg = cfg["model"]
    model = build_model(mcfg["name"], mcfg["num_classes"], mcfg["pretrained"], mcfg["remove_maxpool"]).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=tcfg["label_smoothing"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    scheduler = OneCycleLR(optimizer, max_lr=tcfg["lr"], total_steps=epochs * len(train_loader),
                           pct_start=tcfg["pct_start"])
    use_amp = tcfg["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, best_acc = 0, 0.0
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch, best_acc = ckpt["epoch"], ckpt["best_val_acc"]
        set_rng_state(ckpt["rng"])
    csv_logger = CSVLogger(run_dir / "metrics.csv", resume_epoch=start_epoch)
    if start_epoch >= epochs:
        logger.info(f"Entraînement déjà terminé ({start_epoch}/{epochs} époques, meilleure val acc {best_acc:.2f}%)")

    # ----------------------------------------------------------------- boucle
    for epoch in range(start_epoch, epochs):
        loader_gen.manual_seed(cfg["seed"] + epoch)  # ordre des batchs reproductible même après reprise
        t0 = time.time()
        tr_loss, tr_acc, n_corrupted = train_one_epoch(model, train_loader, criterion, optimizer, scheduler,
                                                       scaler, device, cfg, use_amp, logger, epoch)
        va_loss, va_acc = validate(model, val_loader, criterion, device, use_amp)
        dt = time.time() - t0

        is_best = va_acc > best_acc
        best_acc = max(best_acc, va_acc)
        csv_logger.log({"epoch": epoch + 1, "train_loss": round(tr_loss, 5), "train_acc": round(tr_acc, 3),
                        "val_loss": round(va_loss, 5), "val_acc": round(va_acc, 3),
                        "lr": f"{scheduler.get_last_lr()[0]:.3e}", "corrupted_batches": n_corrupted,
                        "time_s": round(dt, 1)})
        save_checkpoint({"epoch": epoch + 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                         "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                         "best_val_acc": best_acc, "config": cfg, "rng": get_rng_state()}, run_dir / "last.pt")
        if is_best:
            save_checkpoint({"epoch": epoch + 1, "model": model.state_dict(), "val_acc": va_acc, "config": cfg},
                            run_dir / "best.pt")
        logger.info(f"Époque {epoch + 1}/{epochs} | train loss {tr_loss:.4f} acc {tr_acc:.2f}% | "
                    f"val loss {va_loss:.4f} acc {va_acc:.2f}%{' *' if is_best else ''} | "
                    f"lr {scheduler.get_last_lr()[0]:.2e} | {dt:.0f}s")

        if hub_enabled():
            hub_upload(run_dir / "last.pt", f"{run_name}/last.pt")
            hub_upload(run_dir / "metrics.csv", f"{run_name}/metrics.csv")
            if is_best:
                hub_upload(run_dir / "best.pt", f"{run_name}/best.pt")

    plot_training_curves({run_name: run_dir / "metrics.csv"}, run_dir / "curves.png")
    logger.info(f"Terminé. Meilleure val acc : {best_acc:.2f}% -> {run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
