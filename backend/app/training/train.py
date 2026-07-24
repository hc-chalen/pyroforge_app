"""Train the served ``model_thermo`` checkpoint on the parent-grouped train split.

``model_thermo`` predicts absolute dG from an averaged ensemble of towers. The
tower architecture is either the hand-picked ``DEFAULT_DG_WIDTHS`` or, with
``--use-nas``, the autoPyTorch NAS best config (see ``autopytorch_tune``). ΔΔG is
handled separately by ``model_ddg`` (see ``train_ddg``) and absolute Tm by the
Meltome-trained ``model_tm`` (see ``train_tm_meltome``).

Only ``esm2_train.pt`` is read here. The held-out ``esm2_test.pt`` is never
touched during training; it is scored separately by
``app.training.evaluate_and_visualize`` after the final model is chosen.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from app.core.config import settings
from app.data.dataset import build_parent_split, load_raw_data
from app.training.thermo_multitask import train_model_thermo


@torch.no_grad()
def _fit_thermo_calibration(model, data, split, device) -> dict:
    """Val-set linear recalibration (slope, intercept) for the dG head.

    Correlation-based epoch selection leaves a systematic scale/offset on the
    absolute dG prediction; a linear fit on the held-out validation rows removes
    it, lowering MAE without changing correlation. Applied at inference.
    """
    mean = split.embedding_mean
    std = split.embedding_std
    emb = ((data["embeddings"][split.val_idx].numpy() - mean) / std).astype("float32")
    dg_p = np.empty(len(emb), dtype="float32")
    model.eval()
    for s in range(0, len(emb), 8192):
        e = min(s + 8192, len(emb))
        g = model(torch.tensor(emb[s:e], device=device))
        dg_p[s:e] = g.cpu().numpy()
    cal, resid = {}, {}
    for col, pred in (("dg", dg_p),):
        y = data[col].numpy()[split.val_idx]
        ok = ~np.isnan(y)
        if ok.sum() > 2:
            slope, intercept = np.polyfit(pred[ok], y[ok], 1)
            cal[col] = {"slope": float(slope), "intercept": float(intercept)}
            resid[col] = float(np.std((slope * pred[ok] + intercept) - y[ok]))
    return cal, resid


def _pick_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_training(
    data_path: Path | None = None,
    checkpoint_dir: Path | None = None,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str | None = None,
    seed: int = 42,
    val_fraction: float = 0.1,
    use_nas: bool = False,
    n_dg: int = 3,
) -> dict:
    data_path = data_path or settings.data_path
    checkpoint_dir = checkpoint_dir or settings.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    device = _pick_device(device)

    data = load_raw_data(data_path)
    split = build_parent_split(data, seed=seed, val_fraction=val_fraction)
    manifest = split.manifest()

    # Architecture + training hyperparameters for the dG ensemble. By default the
    # hand-picked production tower is used; with ``use_nas`` we build the ensemble
    # from the autoPyTorch NAS best config so the search can improve the served
    # model. weight_decay/dropout default to the production values if absent.
    dg_widths = None
    weight_decay = 1e-4
    dropout = 0.2
    nas_source: dict | None = None
    if use_nas:
        from app.training.autopytorch_tune import load_dg_nas_config

        nas_source = load_dg_nas_config(checkpoint_dir)
        if nas_source is None:
            raise FileNotFoundError(
                "use_nas=True but no dG NAS config found. Run `pyroforge-autopytorch` "
                "first to produce checkpoints/autopytorch/autopytorch_report.json."
            )
        dg_widths = nas_source["widths"]
        dropout = nas_source["dropout"]
        weight_decay = nas_source["weight_decay"]
        lr = nas_source["lr"]
        batch_size = nas_source["batch_size"]

    model_thermo, metrics_thermo = train_model_thermo(
        data, split, device, epochs, batch_size, lr,
        weight_decay=weight_decay, dg_widths=dg_widths, dropout=dropout, n_dg=n_dg,
    )
    thermo_config = metrics_thermo.get("config", {})
    thermo_config["architecture_source"] = "autopytorch_nas" if use_nas else "default"
    if nas_source is not None:
        thermo_config["nas_config"] = nas_source
    thermo_calibration, thermo_residual_std = _fit_thermo_calibration(model_thermo, data, split, device)

    norm_stats = {
        "embedding_mean": split.embedding_mean.tolist(),
        "embedding_std": split.embedding_std.tolist(),
    }

    torch.save(
        {
            "model_name": "model_thermo",
            "state_dict": model_thermo.state_dict(),
            "config": thermo_config,
            "calibration": thermo_calibration,
            "residual_std": thermo_residual_std,
            "norm_stats": norm_stats,
            "split": manifest,
        },
        checkpoint_dir / "model_thermo.pt",
    )

    report = {
        "design": "dg_ensemble",
        "architecture_source": "autopytorch_nas" if use_nas else "default",
        "nas_config": nas_source,
        "training_data_path": str(data_path),
        "device": device,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "split": manifest,
        "model_thermo": metrics_thermo,
        "checkpoint_dir": str(checkpoint_dir),
    }
    with open(checkpoint_dir / "training_report.json", "w") as f:
        json.dump(report, f, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train PyroForge model_thermo (dG ensemble) on a parent-grouped split"
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument(
        "--use-nas",
        action="store_true",
        help=(
            "Build the dG ensemble from the autoPyTorch NAS best config "
            "(run `pyroforge-autopytorch` first). Overrides --lr/--batch-size."
        ),
    )
    parser.add_argument("--n-dg", type=int, default=3, help="dG ensemble size (towers)")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    report = run_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
        val_fraction=args.val_fraction,
        use_nas=args.use_nas,
        n_dg=args.n_dg,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
