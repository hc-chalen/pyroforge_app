"""Fine-tune hybrid thermo model with autoPyTorch-style Hyperband NAS."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

from app.core.config import settings
from app.data.dataset import build_split, load_raw_data
from app.training.features import build_model_dg_frames
from app.training.hyperband_nas import HyperbandRegressor, NASConfig

os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.simplefilter(action="ignore", category=UserWarning)

# Production-scale search space for the served dG tower. Wider than the ΔΔG NAS
# default (max width 480) so the search can reach — and improve on — the
# hand-picked [1024, 512, 256, 128] tower in regression.DEFAULT_DG_WIDTHS.
DG_WIDTH_CHOICES = (128, 256, 512, 768, 1024)
DG_MIN_LAYERS = 2
DG_MAX_LAYERS = 5


def load_dg_nas_config(checkpoint_dir: Path | None = None) -> dict | None:
    """Return the dG NAS best config (widths/dropout/lr/weight_decay/batch_size).

    Reads ``autopytorch/autopytorch_report.json`` written by
    :func:`run_autopytorch_tuning`. Returns ``None`` if no search has been run,
    so callers can fall back to the hand-picked production defaults.
    """
    checkpoint_dir = checkpoint_dir or settings.checkpoint_dir
    report_path = checkpoint_dir / "autopytorch" / "autopytorch_report.json"
    if not report_path.is_file():
        return None
    with open(report_path) as f:
        report = json.load(f)
    best = (report.get("model_thermo_dg") or {}).get("best_config")
    if not best or not best.get("widths"):
        return None
    return {
        "widths": [int(w) for w in best["widths"]],
        "dropout": float(best.get("dropout", 0.2)),
        "lr": float(best.get("lr", 1e-3)),
        "weight_decay": float(best.get("weight_decay", 1e-4)),
        "batch_size": int(best.get("batch_size", 256)),
    }


def _metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64).ravel()
    target = np.asarray(target, dtype=np.float64).ravel()
    mae = float(np.mean(np.abs(pred - target)))
    rmse = float(np.sqrt(np.mean((pred - target) ** 2)))
    r, _ = pearsonr(pred, target) if len(pred) > 1 else (0.0, 1.0)
    return {"mae": mae, "rmse": rmse, "pearson_r": float(r)}


def _run_nas_regression(
    x_train,
    y_train,
    x_val,
    y_val,
    dataset_name: str,
    time_limit: int,
    output_dir: Path,
    seed: int = 42,
    device: str = "cpu",
    max_train_rows: int | None = None,
    width_choices: tuple[int, ...] = DG_WIDTH_CHOICES,
    min_layers: int = DG_MIN_LAYERS,
    max_layers: int = DG_MAX_LAYERS,
) -> tuple[HyperbandRegressor, dict]:
    output_dir.mkdir(parents=True, exist_ok=True)

    nas = HyperbandRegressor(
        nas_config=NASConfig(
            max_runtime=time_limit,
            seed=seed,
            width_choices=tuple(width_choices),
            min_layers=min_layers,
            max_layers=max_layers,
        ),
        device=device,
    )

    X_train = x_train.values.astype(np.float32)
    Y_train = y_train.values.astype(np.float32)
    X_valid = x_val.values.astype(np.float32)
    Y_valid = y_val.values.astype(np.float32)

    # Optional row cap so architecture search evaluates many configs on large
    # targets (e.g. the ~0.5M-row dG frame). We subsample TRAIN only and keep the
    # FULL validation set, so reported val_metrics stay comparable across runs.
    # NAS is a benchmark/search over architecture; the winning shape is retrained
    # on all rows by app.training.train, so subsampling here is not a data loss.
    n_train_full = X_train.shape[0]
    subsampled_rows = None
    if max_train_rows is not None and n_train_full > max_train_rows:
        rng = np.random.default_rng(seed)
        sel = rng.choice(n_train_full, size=max_train_rows, replace=False)
        X_train = X_train[sel]
        Y_train = Y_train[sel]
        subsampled_rows = int(max_train_rows)

    trial = nas.fit(X_train, Y_train, X_valid, Y_valid)
    y_pred = nas.predict(X_valid)
    val_metrics = _metrics(y_pred, Y_valid)

    result = {
        "dataset_name": dataset_name,
        "time_limit_secs": time_limit,
        "train_rows_full": n_train_full,
        "train_rows_searched": subsampled_rows or n_train_full,
        "val_rows": int(X_valid.shape[0]),
        "val_metrics": val_metrics,
        "best_config": {k: v for k, v in trial.config.items() if not k.startswith("_")},
        "best_budget_epochs": trial.budget_epochs,
        "search_history": nas.search_history,
    }

    with open(output_dir / "search_result.json", "w") as f:
        json.dump(result, f, indent=2)

    return nas, result


# Target registry: report-key → (dataset name, frame builder, budget weight).
# Weights are used only across the SELECTED targets and are renormalized, so
# dropping/adding a target still spends the whole budget on what remains.
# Tm was retired here alongside the removal of model_thermo's Tm head — absolute
# Tm is served by model_tm (Meltome, app.models.tm_regressor), not by this NAS.
_TARGETS = {
    "dg": ("model_thermo_dg", "pyroforge_thermo_dg", build_model_dg_frames, 1.0),
}


def run_autopytorch_tuning(
    data_path: Path | None = None,
    checkpoint_dir: Path | None = None,
    time_limit: int = 1800,
    seed: int = 42,
    device: str | None = None,
    targets: list[str] | None = None,
    max_train_rows: int | None = None,
    width_choices: tuple[int, ...] = DG_WIDTH_CHOICES,
    min_layers: int = DG_MIN_LAYERS,
    max_layers: int = DG_MAX_LAYERS,
) -> dict:
    """
    Run Hyperband NAS on absolute dG (``model_thermo_dg``, embedding features).

    ``targets`` selects which registered targets to search (default: all of
    ``_TARGETS``). ``max_train_rows`` caps the rows the architecture search trains
    on per target — important for dG (~0.5M rows), where an uncapped search only
    completes a single trial. The full validation set is always used for scoring.

    These NAS models are saved as ``.pkl`` regressors for benchmarking.
    Production inference uses PyTorch checkpoints from ``app.training.train``.
    """
    data_path = data_path or settings.data_path
    checkpoint_dir = checkpoint_dir or settings.checkpoint_dir
    device = device or settings.resolved_device
    apt_dir = checkpoint_dir / "autopytorch"
    apt_dir.mkdir(parents=True, exist_ok=True)

    selected = targets or list(_TARGETS)
    unknown = [t for t in selected if t not in _TARGETS]
    if unknown:
        raise ValueError(f"Unknown target(s) {unknown}; choose from {list(_TARGETS)}")
    weight_total = sum(_TARGETS[t][3] for t in selected)

    data = load_raw_data(data_path)
    split = build_split(data)

    report = {
        "method": "Hyperband NAS (autoPyTorch BOHB methodology)",
        "design": "hybrid_" + "_".join(selected),
        "note": (
            "Official autoPyTorch package requires sklearn<0.25 and is incompatible "
            "with Python 3.12. This implements the same multi-fidelity MLP search. "
            "NAS pickles are for benchmarking; deploy PyTorch checkpoints from train.py."
        ),
        "total_time_limit_secs": time_limit,
        "targets": selected,
        "max_train_rows": max_train_rows,
        "search_space": {
            "width_choices": list(width_choices),
            "min_layers": min_layers,
            "max_layers": max_layers,
        },
        "time_budget": {},
        "split": split.manifest(),
        "training_data_path": str(data_path),
        "evaluation_scope": "internal_validation_only",
        "checkpoint_dir": str(apt_dir),
        "seed": seed,
        "device": device,
    }

    for target in selected:
        report_key, dataset_name, build_frames, weight = _TARGETS[target]
        time_target = int(time_limit * weight / weight_total)
        x_train, y_train, x_val, y_val = build_frames(data, split)
        model, result = _run_nas_regression(
            x_train, y_train, x_val, y_val,
            dataset_name=dataset_name,
            time_limit=time_target,
            output_dir=apt_dir / report_key,
            seed=seed,
            device=device,
            max_train_rows=max_train_rows,
            width_choices=width_choices,
            min_layers=min_layers,
            max_layers=max_layers,
        )
        with open(apt_dir / f"{report_key}.pkl", "wb") as f:
            pickle.dump(model, f)
        report["time_budget"][report_key.replace("model_", "")] = time_target
        report[report_key] = result

    report_path = apt_dir / "autopytorch_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune hybrid thermo model with autoPyTorch-style NAS (default: 30 min total)"
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=1800,
        help="Total wall-clock budget in seconds (default: 1800 = 30 minutes)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--targets",
        type=str,
        default="dg",
        help="Comma-separated targets to search (default: dg; only dg is available)",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help=(
            "Cap training rows per target during the architecture search "
            "(recommended ~50000 for dG's ~0.5M rows; default: no cap)"
        ),
    )
    parser.add_argument(
        "--width-choices",
        type=str,
        default=",".join(map(str, DG_WIDTH_CHOICES)),
        help=f"Comma-separated layer widths to sample (default: {','.join(map(str, DG_WIDTH_CHOICES))})",
    )
    parser.add_argument("--min-layers", type=int, default=DG_MIN_LAYERS)
    parser.add_argument("--max-layers", type=int, default=DG_MAX_LAYERS)
    args = parser.parse_args()

    report = run_autopytorch_tuning(
        data_path=args.data_path,
        checkpoint_dir=args.checkpoint_dir,
        time_limit=args.time_limit,
        seed=args.seed,
        device=args.device,
        targets=[t.strip() for t in args.targets.split(",") if t.strip()],
        max_train_rows=args.max_train_rows,
        width_choices=tuple(int(w) for w in args.width_choices.split(",") if w.strip()),
        min_layers=args.min_layers,
        max_layers=args.max_layers,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
