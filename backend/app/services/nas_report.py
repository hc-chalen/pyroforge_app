"""Load and summarize Hyperband NAS (autoPyTorch-style) tuning reports."""

from __future__ import annotations

import json
from pathlib import Path

from app.core.config import settings

_MODEL_KEYS = (
    ("model_thermo_dg", "ΔG", "kcal/mol"),
)

# Fixed choices in hyperband_nas._DynamicMLP / _train_one (not part of the search space).
_FIXED_HYPERPARAMS = {
    "activation": "GELU",
    "optimizer": "AdamW",
    "input_normalization": "BatchNorm1d",
    "loss": "MSE",
}


def _architecture_label(config: dict) -> str:
    widths = config.get("widths") or []
    parts = [str(config.get("input_dim", "?")), *map(str, widths), "1"]
    return " → ".join(parts)


def _algorithm_names(config: dict) -> dict[str, str]:
    n_layers = len(config.get("widths") or [])
    return {
        "search": "Hyperband NAS (BOHB)",
        "architecture": f"{n_layers}-layer MLP" if n_layers else "MLP",
        "activation": _FIXED_HYPERPARAMS["activation"],
        "optimizer": _FIXED_HYPERPARAMS["optimizer"],
        "normalization": _FIXED_HYPERPARAMS["input_normalization"],
        "loss": _FIXED_HYPERPARAMS["loss"],
    }


def _search_time_share(time_limit: int | None, total: int | None) -> float | None:
    if time_limit is None or not total:
        return None
    return round(time_limit / total, 3)


def _format_weight_decay(value: float | None) -> str:
    if value is None:
        return "—"
    if value == 0:
        return "0 (none)"
    return f"{value:.2e}"


def _summarize_model(
    key: str,
    label: str,
    unit: str,
    raw: dict,
    total_time_limit: int | None,
) -> dict:
    config = raw.get("best_config") or {}
    metrics = raw.get("val_metrics") or {}
    merged = {**_FIXED_HYPERPARAMS, **config}
    time_limit = raw.get("time_limit_secs")
    return {
        "key": key,
        "label": label,
        "unit": unit,
        "architecture": _architecture_label(config),
        "n_layers": len(config.get("widths") or []),
        "best_budget_epochs": raw.get("best_budget_epochs"),
        "time_limit_secs": time_limit,
        "algorithm_names": _algorithm_names(config),
        "weighting": {
            "loss_task_weights": "single-target (1.0)",
            "weight_decay": config.get("weight_decay"),
            "weight_decay_label": _format_weight_decay(config.get("weight_decay")),
            "search_time_share": _search_time_share(time_limit, total_time_limit),
        },
        "val_metrics": {
            "mae": metrics.get("mae"),
            "rmse": metrics.get("rmse"),
            "pearson_r": metrics.get("pearson_r"),
        },
        "hyperparameters": {
            "activation": merged.get("activation"),
            "optimizer": merged.get("optimizer"),
            "input_normalization": merged.get("input_normalization"),
            "loss": merged.get("loss"),
            "dropout": merged.get("dropout"),
            "learning_rate": merged.get("lr"),
            "weight_decay": merged.get("weight_decay"),
            "batch_size": merged.get("batch_size"),
            "hidden_widths": config.get("widths"),
        },
    }


# Production PyTorch checkpoints (inference path) — not NAS benchmarks.
# model_thermo is a dG-only ensemble now (Tm tower removed); absolute Tm is served
# by the separate Meltome model_tm and ΔΔG by model_ddg. Optimization ranks
# mutations purely by ΔΔG.
_PRODUCTION_WEIGHTING = {
    "thermo_dg": {"dg": 1.0},
    "optimization_score": {"ddg": 1.0, "formula": "ΔΔG"},
    "adamw_weight_decay": 0.0,
}


def load_nas_report(checkpoint_dir: Path | None = None) -> dict | None:
    checkpoint_dir = checkpoint_dir or settings.checkpoint_dir
    report_path = checkpoint_dir / "autopytorch" / "autopytorch_report.json"
    if not report_path.is_file():
        return None

    with open(report_path) as f:
        raw = json.load(f)

    models = []
    total_time = raw.get("total_time_limit_secs")
    for key, label, unit in _MODEL_KEYS:
        model_raw = raw.get(key)
        if model_raw:
            models.append(_summarize_model(key, label, unit, model_raw, total_time))

    time_budget = raw.get("time_budget") or {}
    search_budget_weights = None
    if time_budget and total_time:
        search_budget_weights = {
            k: round(v / total_time, 3) for k, v in time_budget.items()
        }

    comparison_path = checkpoint_dir / "autopytorch" / "plots" / "comparison_report.json"
    comparison = None
    if comparison_path.is_file():
        with open(comparison_path) as f:
            comparison = json.load(f)

    return {
        "available": bool(models),
        "method": raw.get("method"),
        "design": raw.get("design"),
        "note": raw.get("note"),
        "total_time_limit_secs": raw.get("total_time_limit_secs"),
        "time_budget": raw.get("time_budget"),
        "search_budget_weights": search_budget_weights,
        "production_weighting": _PRODUCTION_WEIGHTING,
        "seed": raw.get("seed"),
        "device": raw.get("device"),
        "models": models,
        "comparison": comparison,
    }
