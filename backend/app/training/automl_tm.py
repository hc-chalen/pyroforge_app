"""AutoML search (hyperparameters + model families + stacking) for absolute Tm.

The served Tm model (:class:`~app.models.tm_regressor.TmRegressor`, PCA-100 →
Ridge+GBM) uses hand-picked, untuned hyperparameters. Absolute Tm is a small-data
(~4k Meltome proteins) per-protein regression near its data ceiling, so this does
a disciplined AutoML pass rather than adding capacity:

  1. K-fold CV on the Meltome **train** split (each row is one protein, so plain
     K-fold has no leakage) over several estimator families — Ridge, ElasticNet,
     SVR(RBF), kNN, HistGradientBoosting — each grid-tuned inside a
     StandardScaler → PCA pipeline (PCA components are part of the search).
  2. A stacking ensemble of the tuned base families.
  3. Refit the CV winner on all train rows.
  4. Save the AutoML model + a comparison report/plots. It is promoted to the served
     ``model_tm.joblib`` only if its **cross-validated train MAE** beats the
     baseline's (or with ``--force-promote``); the previous checkpoint is backed up
     first. The held-out test split is scored for the record but is deliberately NOT
     part of the promotion decision — gating on it would make the reported test
     metrics a selection signal rather than an unbiased estimate of the served model.

Run:  uv run pyroforge-automl-tm
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from app.core.config import settings
from app.models.tm_regressor import TmAutoMLRegressor, TmRegressor

logger = logging.getLogger(__name__)


def _metrics(pred, true) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    return {
        "n": int(len(pred)),
        "pearson": float(pearsonr(pred, true)[0]),
        "spearman": float(spearmanr(pred, true).correlation),
        "mae": float(np.mean(np.abs(pred - true))),
        "rmse": float(np.sqrt(np.mean((pred - true) ** 2))),
    }


def _pipe(estimator) -> Pipeline:
    """StandardScaler → PCA → estimator. PCA components are searched per family."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(random_state=0)),
        ("est", estimator),
    ])


def _candidate_grids(quick: bool) -> dict[str, tuple[Pipeline, dict]]:
    """Return {family: (pipeline, param_grid)}. ``quick`` shrinks grids for smoke runs."""
    pca_components = [100] if quick else [50, 100, 150]
    grids: dict[str, tuple[Pipeline, dict]] = {
        "ridge": (_pipe(Ridge()), {
            "pca__n_components": pca_components,
            "est__alpha": [10.0, 50.0, 100.0] if quick else [1.0, 10.0, 50.0, 100.0, 300.0],
        }),
        "elasticnet": (_pipe(ElasticNet(max_iter=5000)), {
            "pca__n_components": pca_components,
            "est__alpha": [0.01, 0.1] if quick else [0.001, 0.01, 0.1, 1.0],
            "est__l1_ratio": [0.5] if quick else [0.1, 0.5, 0.9],
        }),
        "svr": (_pipe(SVR(kernel="rbf")), {
            "pca__n_components": pca_components,
            "est__C": [10.0] if quick else [1.0, 10.0, 100.0],
            "est__gamma": ["scale"] if quick else ["scale", 0.01],
            "est__epsilon": [1.0] if quick else [0.5, 1.0, 2.0],
        }),
        "knn": (_pipe(KNeighborsRegressor()), {
            "pca__n_components": pca_components,
            "est__n_neighbors": [10] if quick else [5, 10, 20, 40],
            "est__weights": ["distance"] if quick else ["uniform", "distance"],
        }),
        "histgbm": (_pipe(HistGradientBoostingRegressor(random_state=0)), {
            "pca__n_components": pca_components,
            "est__max_depth": [3] if quick else [3, None],
            "est__learning_rate": [0.05] if quick else [0.03, 0.05, 0.1],
            "est__max_iter": [400] if quick else [300, 600],
            "est__l2_regularization": [1.0] if quick else [0.1, 1.0],
        }),
    }
    return grids


def _search_family(name, pipe, grid, X, y, cv, n_jobs) -> dict:
    gs = GridSearchCV(pipe, grid, scoring="neg_mean_absolute_error", cv=cv, n_jobs=n_jobs)
    gs.fit(X, y)
    logger.info("  %-11s CV MAE=%.3f  best=%s", name, -gs.best_score_,
                {k: v for k, v in gs.best_params_.items()})
    return {"estimator": gs.best_estimator_, "cv_mae": float(-gs.best_score_),
            "best_params": {k: (v if isinstance(v, (int, float, str)) else str(v))
                            for k, v in gs.best_params_.items()}}


def _cv_mae(estimator, X, y, cv) -> float:
    from sklearn.model_selection import cross_val_score
    return float(-cross_val_score(estimator, X, y, scoring="neg_mean_absolute_error", cv=cv).mean())


def run_automl_tm(
    meltome_npz: Path | None = None,
    out_dir: Path | None = None,
    cv_folds: int = 5,
    seed: int = 42,
    quick: bool = False,
    n_jobs: int = -1,
    force_promote: bool = False,
) -> dict:
    ck = settings.checkpoint_dir / "improved"
    meltome_npz = meltome_npz or ck / "meltome_tm.npz"
    plots_dir = (out_dir or settings.checkpoint_dir / "plots") / "tm_automl"
    plots_dir.mkdir(parents=True, exist_ok=True)

    M = np.load(meltome_npz, allow_pickle=True)
    emb, tm, split = M["emb"].astype(np.float32), M["tm"].astype(np.float64), M["split"].astype(str)
    tr, te = split != "test", split == "test"
    Xtr, ytr, Xte, yte = emb[tr], tm[tr], emb[te], tm[te]
    logger.info("Meltome: %d train / %d test proteins", tr.sum(), te.sum())

    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    # Baseline: current served recipe (hand-picked HPs), same train/test.
    baseline = TmRegressor().fit(Xtr, ytr)
    baseline_test = _metrics(baseline.predict(Xte), yte)
    # Baseline CV MAE on the *training* rows — this, not the test score, is what the
    # promotion decision compares against (see `beats` below).
    baseline_cv_mae = _cv_mae(TmRegressor(), Xtr, ytr, cv)
    logger.info("baseline (PCA100+Ridge+GBM)  CV MAE=%.3f | test MAE=%.3f r=%.3f",
                baseline_cv_mae, baseline_test["mae"], baseline_test["pearson"])

    # 1) tune each family
    logger.info("searching estimator families (%d-fold CV) ...", cv_folds)
    families = {}
    for name, (pipe, grid) in _candidate_grids(quick).items():
        families[name] = _search_family(name, pipe, grid, Xtr, ytr, cv, n_jobs)

    # 2) stacking ensemble of the tuned base families
    base = [(n, families[n]["estimator"]) for n in ("ridge", "svr", "histgbm") if n in families]
    stack = StackingRegressor(estimators=base, final_estimator=Ridge(alpha=10.0), cv=cv_folds)
    stack_cv_mae = _cv_mae(stack, Xtr, ytr, cv)
    logger.info("  %-11s CV MAE=%.3f", "stack", stack_cv_mae)

    # 3) pick the CV winner
    candidates = {n: f["cv_mae"] for n, f in families.items()}
    candidates["stack"] = stack_cv_mae
    winner = min(candidates, key=candidates.get)
    logger.info("CV winner: %s (MAE=%.3f)", winner, candidates[winner])

    if winner == "stack":
        best_est = stack.fit(Xtr, ytr)
    else:
        best_est = families[winner]["estimator"]  # already refit on full train by GridSearchCV

    automl = TmAutoMLRegressor(best_est, recipe=f"automl:{winner}")
    automl_test = _metrics(automl.predict(Xte), yte)
    # Held-out residual, not train: the served ±95% interval must reflect error on
    # unseen proteins. A train residual understates it badly (the boosted/stacked
    # members fit the training rows closely).
    resid = float(np.std(automl.predict(Xte) - yte))
    logger.info("AutoML winner (%s)  test MAE=%.3f r=%.3f", winner,
                automl_test["mae"], automl_test["pearson"])

    # Promote on cross-validated *training* error, never on the test score. Gating on
    # test turns the held-out split into a model-selection signal, so the reported
    # test metrics stop being an unbiased estimate of the served model.
    beats = candidates[winner] < baseline_cv_mae
    report = {
        "winner": winner,
        "cv_folds": cv_folds,
        "cv_mae_by_candidate": candidates,
        "family_best_params": {n: f["best_params"] for n, f in families.items()},
        "selection_metric": "cv_mae_on_train",
        "baseline_cv_mae": baseline_cv_mae,
        "automl_cv_mae": candidates[winner],
        "baseline_test": baseline_test,
        "automl_test": automl_test,
        "beats_baseline_cv_mae": bool(beats),
        "test_used_for_selection": False,
        "n_train": int(tr.sum()),
    }

    # save AutoML model (always) in the model_tm.joblib schema
    automl_path = ck / "model_tm_automl.joblib"
    joblib.dump({"model": automl, "recipe": automl.recipe,
                 "residual_std": {"tm": resid}, "n_train": int(tr.sum()),
                 "test_metrics": automl_test, "cv_mae": candidates[winner]}, automl_path)

    # promote to served model_tm.joblib if it wins (or forced), backing up the old one
    served = ck / "model_tm.joblib"
    promoted = False
    if force_promote or beats:
        if served.exists():
            shutil.copy2(served, ck / "model_tm_previous.joblib")
        shutil.copy2(automl_path, served)
        promoted = True
    report["promoted_to_served"] = promoted
    report["automl_checkpoint"] = str(automl_path)

    _plot_comparison(baseline_test, automl_test, yte, baseline.predict(Xte),
                     automl.predict(Xte), winner, plots_dir)
    with open(plots_dir / "tm_automl_report.json", "w") as f:
        json.dump(report, f, indent=2)
    logger.info("%s | report+plots in %s",
                "PROMOTED to model_tm.joblib" if promoted else "kept baseline (AutoML did not win)",
                plots_dir)
    return report


def _plot_comparison(base_m, apt_m, yte, base_pred, apt_pred, winner, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    # MAE / Pearson bars
    axes[0].bar(["baseline", f"automl\n({winner})"], [base_m["mae"], apt_m["mae"]],
                color=["#3b82f6", "#f59e0b"])
    axes[0].set_title("Test MAE (°C) — lower is better")
    for i, v in enumerate([base_m["mae"], apt_m["mae"]]):
        axes[0].text(i, v, f"{v:.2f}", ha="center", va="bottom")
    axes[1].bar(["baseline", f"automl\n({winner})"], [base_m["pearson"], apt_m["pearson"]],
                color=["#3b82f6", "#f59e0b"])
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Test Pearson r — higher is better")
    for i, v in enumerate([base_m["pearson"], apt_m["pearson"]]):
        axes[1].text(i, v, f"{v:.3f}", ha="center", va="bottom")
    # scatter of AutoML winner
    lo, hi = min(yte.min(), apt_pred.min()), max(yte.max(), apt_pred.max())
    axes[2].scatter(yte, apt_pred, alpha=0.3, s=10, edgecolors="none", c="#f59e0b")
    axes[2].plot([lo, hi], [lo, hi], "r--", lw=1)
    axes[2].set_xlabel("Actual Tm (°C)"); axes[2].set_ylabel("Predicted Tm (°C)")
    axes[2].set_title(f"AutoML ({winner}) — r={apt_m['pearson']:.3f}, MAE={apt_m['mae']:.2f}")
    fig.suptitle("Absolute Tm: baseline vs AutoML", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "tm_automl_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="AutoML search for the served absolute-Tm model")
    ap.add_argument("--meltome", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None, help="plots root (default: checkpoints/plots)")
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true", help="shrink grids for a fast sanity run")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--force-promote", action="store_true",
                    help="overwrite served model_tm.joblib even if AutoML doesn't beat baseline")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    report = run_automl_tm(
        meltome_npz=args.meltome, out_dir=args.out_dir, cv_folds=args.cv_folds,
        seed=args.seed, quick=args.quick, n_jobs=args.n_jobs, force_promote=args.force_promote,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
