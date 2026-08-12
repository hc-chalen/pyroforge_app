"""Grouped K-fold cross-validation for model_ddg — the real held-out benchmark.

**Why this exists.** Every ΔΔG number the project currently reports comes from one
of two sets, and neither can carry a generalization claim:

  * the *validation* split, which ``train_ddg`` also uses for epoch selection and
    for the residual σ, so it is optimistic by construction;
  * ``esm2_test.pt``, which after WT-anchoring contains **4 parent proteins**, 98%
    of its rows belonging to one of them — protein-level statistics from it are
    noise (see ``model_ddg_calibration.json``'s ``test_check.caveat``).

``esm2_train.pt`` has 499 parents, **263 of which carry ≥20 anchored ΔΔG mutations**.
Rotating them through K folds gives an out-of-fold prediction for every one of those
263 proteins, each produced by a model that never saw it — a two-order-of-magnitude
improvement in evaluation power over the 4-protein file, using no new data.

**Why not an MMseqs2 identity-clustered split.** The exported tensors carry no
sequence column (only integer ``source_sequence_ids``), so sequence clustering is
not possible on this file. Grouping by ``parent_protein_ids`` is the strongest
split available here; homologous *parents* could still straddle folds, so these
numbers remain an upper bound on performance for a genuinely novel fold. Say so
when quoting them.

**What it produces**, all measured out-of-fold:
  1. the within-protein Spearman *distribution* over 263 proteins, not a median
     over 3;
  2. top-k precision — what the optimizer's shortlist is actually worth — with a
     bootstrap CI over proteins, directly comparable to ``calibrate_ddg``'s
     validation numbers;
  3. a clean residual σ, never touched by early stopping, for the served ±95% band;
  4. the ΔΔG(wt→wt) offset per fold, so it is clear whether the ≈−0.24 kcal/mol
     level bias of the shipped checkpoint is a stable property or a one-off.

Honest-by-construction: within each fold the early-stopping slice is carved out of
the *training* parents, so the evaluation parents are never seen, not even for
choosing an epoch. Normalization statistics are refit per fold on training rows only.

Run::

    uv run python -m scripts.benchmark_grouped_cv --quick     # ~2 min smoke test
    uv run python -m scripts.benchmark_grouped_cv             # full run, hours
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.model_selection import GroupKFold

from app.core.config import settings
from app.data.dataset import wt_reference_index
from app.training.calibrate_ddg import ranking_precision
from app.training.features import anchored_ddg_rows, build_ddg_delta_eval_arrays
from app.training.train_ddg import _predict, _train_full, antisymmetric_augment

logger = logging.getLogger("benchmark_grouped_cv")

# Fraction of *training* parents reserved for early stopping inside each fold.
EARLY_STOP_PARENT_FRACTION = 0.1


def _pick_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_architecture(checkpoint_dir: Path) -> dict:
    """Reuse the shipped NAS architecture instead of re-searching it per fold.

    The question here is how well *the served recipe* generalizes, so the
    architecture is held fixed and only the weights are refit. Re-running NAS
    inside each fold would answer a different (and far more expensive) question.
    """
    ckpt = torch.load(checkpoint_dir / "model_ddg.pt", map_location="cpu", weights_only=False)
    config = dict(ckpt["nas"]["best_config"])
    config.setdefault("input_dim", ckpt["config"]["input_dim"])
    return config


def _embedding_stats(data: dict, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standardization statistics from the given rows only (refit per fold)."""
    emb = data["embeddings"][rows].numpy()
    mean = emb.mean(axis=0)
    std = emb.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean, std


def _self_ddg_offset(model, data: dict, rows: np.ndarray, wt_ref: np.ndarray,
                     mean: np.ndarray, std: np.ndarray, device: str) -> float:
    """Mean predicted ΔΔG for a null (wild-type → wild-type) mutation.

    Structurally this must be 0. Measuring it per fold shows whether the shipped
    checkpoint's ≈−0.24 kcal/mol offset is a stable, correctable property of the
    recipe or an artefact of one training run.
    """
    wt_rows = np.unique(wt_ref[rows])
    wt_rows = wt_rows[wt_rows >= 0][:2000]
    if len(wt_rows) == 0:
        return float("nan")
    emb = data["embeddings"][wt_rows].numpy()
    norm = (emb - mean) / std
    # Feature layout is [mutant ⊕ Δemb]; a null mutation has Δemb = 0.
    x = np.concatenate([norm, np.zeros_like(norm)], axis=1).astype(np.float32)
    return float(np.mean(_predict(model, x, device)))


def run(
    data_path: Path | None = None,
    checkpoint_dir: Path | None = None,
    out_path: Path | None = None,
    folds: int = 5,
    epochs: int = 12,
    patience: int = 3,
    device: str | None = None,
    seed: int = 42,
    max_train_rows: int | None = None,
    max_parents: int | None = None,
) -> dict:
    data_path = data_path or settings.data_path
    checkpoint_dir = checkpoint_dir or settings.checkpoint_dir
    out_path = out_path or checkpoint_dir / "ddg_grouped_cv.json"
    device = _pick_device(device)
    rng = np.random.default_rng(seed)

    logger.info("loading %s ...", data_path)
    data = torch.load(data_path, map_location="cpu", weights_only=False, mmap=True)
    wt_ref = wt_reference_index(data)
    all_rows = anchored_ddg_rows(data, np.arange(data["embeddings"].shape[0]), wt_ref)
    pid_all = data["parent_protein_ids"].numpy()
    parents = np.unique(pid_all[all_rows])

    if max_parents:  # smoke-test mode
        parents = rng.choice(parents, min(max_parents, len(parents)), replace=False)
        all_rows = all_rows[np.isin(pid_all[all_rows], parents)]

    logger.info(
        "%d anchored ΔΔG rows across %d parent proteins → %d-fold grouped CV",
        len(all_rows), len(parents), folds,
    )

    config = _load_architecture(checkpoint_dir)
    logger.info("architecture (from shipped NAS config): widths=%s dropout=%.3f "
                "lr=%.2e bs=%d", config["widths"], config["dropout"],
                config["lr"], config["batch_size"])

    # Out-of-fold predictions: every protein scored by a model that never saw it.
    oof_pred = np.full(len(all_rows), np.nan, dtype=np.float32)
    oof_true = data["ddg"].numpy()[all_rows].astype(np.float32)
    oof_pid = pid_all[all_rows]
    fold_reports: list[dict] = []

    splitter = GroupKFold(n_splits=folds)
    for fold, (tr_pos, ev_pos) in enumerate(
        splitter.split(all_rows, groups=oof_pid), start=1
    ):
        t0 = time.time()
        tr_rows, ev_rows = all_rows[tr_pos], all_rows[ev_pos]

        # Early stopping must not see the evaluation parents — carve its slice out
        # of the training parents instead.
        tr_parents = np.unique(pid_all[tr_rows])
        n_es = max(1, int(len(tr_parents) * EARLY_STOP_PARENT_FRACTION))
        es_parents = set(rng.permutation(tr_parents)[:n_es].tolist())
        is_es = np.isin(pid_all[tr_rows], list(es_parents))
        fit_rows, es_rows = tr_rows[~is_es], tr_rows[is_es]

        if max_train_rows and len(fit_rows) > max_train_rows:
            fit_rows = rng.choice(fit_rows, max_train_rows, replace=False)

        # Normalization refit on this fold's training rows only.
        mean, std = _embedding_stats(data, fit_rows)

        x_tr, y_tr, _ = build_ddg_delta_eval_arrays(data, fit_rows, mean, std, concat_mut=True)
        x_es, y_es, _ = build_ddg_delta_eval_arrays(data, es_rows, mean, std, concat_mut=True)
        x_ev, y_ev, _ = build_ddg_delta_eval_arrays(data, ev_rows, mean, std, concat_mut=True)
        x_tr, y_tr = antisymmetric_augment(x_tr, y_tr)

        logger.info(
            "fold %d/%d — train %d rows (%d parents, incl. reverse aug) | "
            "early-stop %d rows (%d parents) | eval %d rows (%d parents)",
            fold, folds, len(y_tr), len(tr_parents) - n_es, len(y_es), n_es,
            len(y_ev), len(np.unique(pid_all[ev_rows])),
        )

        model = _train_full(config, x_tr, y_tr, x_es, y_es, device, epochs, patience)
        pred = _predict(model, x_ev, device)
        oof_pred[ev_pos] = pred

        offset = _self_ddg_offset(model, data, ev_rows, wt_ref, mean, std, device)
        fold_reports.append({
            "fold": fold,
            "train_parents": int(len(tr_parents) - n_es),
            "eval_parents": int(len(np.unique(pid_all[ev_rows]))),
            "eval_rows": int(len(y_ev)),
            "pearson_r": float(pearsonr(pred, y_ev)[0]),
            "mae": float(np.mean(np.abs(pred - y_ev))),
            "residual_std": float(np.std(pred - y_ev)),
            "self_ddg_offset": offset,
            "seconds": round(time.time() - t0, 1),
        })
        logger.info(
            "fold %d done in %.0fs — r=%.3f MAE=%.3f σ=%.3f ΔΔG(wt→wt)=%+.3f",
            fold, time.time() - t0, fold_reports[-1]["pearson_r"],
            fold_reports[-1]["mae"], fold_reports[-1]["residual_std"], offset,
        )

    scored = ~np.isnan(oof_pred)
    pred, true, pid = oof_pred[scored], oof_true[scored], oof_pid[scored]

    # Keep the raw out-of-fold predictions. This run costs hours, and summary
    # statistics alone cannot answer questions asked later (a different k, a
    # per-protein breakdown, another selection rule) without repeating all of it.
    preds_path = out_path.with_suffix(".predictions.npz")
    np.savez_compressed(
        preds_path, pred=pred, true=true, parent_id=pid, row_index=all_rows[scored]
    )

    # Same metric function the served calibration report uses, so the two are
    # directly comparable — this is the honest counterpart of that optimistic number.
    # The retired 1.96σ gate is scored on the identical proteins so the decision to
    # replace it with top-k rests on this set, not on the 22-protein validation one.
    legacy_gate = 1.96 * float(
        (torch.load(checkpoint_dir / "model_ddg.pt", map_location="cpu",
                    weights_only=False).get("residual_std") or {}).get("ddg", float("nan"))
    )
    ranking = ranking_precision(
        pred, true, pid, legacy_gate=legacy_gate if np.isfinite(legacy_gate) else None
    )
    # Random-shortlist control on the identical proteins: a top-k precision only
    # means something against the rate you would get by picking blind.
    rand = ranking_precision(rng.permutation(pred), true, pid)

    report = {
        "benchmark": "grouped_kfold_cv",
        "purpose": (
            "Out-of-fold ΔΔG performance over every parent protein with enough "
            "anchored mutations, replacing the 4-protein esm2_test.pt estimate."
        ),
        "data_path": str(data_path),
        # Which checkout produced these numbers. More than one PyroForge tree can
        # exist on a machine, and an activated environment can silently put the
        # wrong one on sys.path — a report that does not say which code it ran is
        # not a result anyone can act on. See backend/conftest.py.
        "code_path": str(Path(__import__("app").__file__).resolve().parent),
        "device": device,
        "folds": folds,
        "seed": seed,
        "epochs_max": epochs,
        "patience": patience,
        "architecture": {k: v for k, v in config.items()},
        "n_rows_scored": int(scored.sum()),
        "n_parents_total": int(len(parents)),
        "pooled": {
            "pearson_r": float(pearsonr(pred, true)[0]),
            "mae": float(np.mean(np.abs(pred - true))),
            "residual_std": float(np.std(pred - true)),
        },
        "ranking": ranking,
        "random_shortlist_control": {
            "top_k_precision": rand["top_k_precision"],
            "median_within_protein_spearman": rand["median_within_protein_spearman"],
        },
        "self_ddg_offset": {
            "per_fold": [f["self_ddg_offset"] for f in fold_reports],
            "mean": float(np.nanmean([f["self_ddg_offset"] for f in fold_reports])),
            "note": (
                "Structurally must be 0. A stable nonzero mean across folds is a "
                "constant level bias on every served ΔΔG and is correctable; a "
                "fold-varying one is run-to-run noise and is not."
            ),
        },
        "folds_detail": fold_reports,
        "predictions_path": str(preds_path),
        "caveats": [
            "Grouped by parent_protein_ids. The exported tensors carry no sequence "
            "column, so homologous parents cannot be detected and may straddle "
            "folds — treat these as an upper bound for a genuinely novel fold.",
            "Architecture is held fixed at the shipped NAS config; this measures "
            "the generalization of the served recipe, not of the search.",
        ],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    tk = ranking["top_k_precision"]
    rk = rand["top_k_precision"]
    print("\n" + "=" * 68)
    print(f"  Grouped {folds}-fold CV — {ranking['n_proteins']} proteins scored out-of-fold")
    print("=" * 68)
    print(f"  pooled r          {report['pooled']['pearson_r']:.3f}"
          f"   MAE {report['pooled']['mae']:.3f}"
          f"   σ {report['pooled']['residual_std']:.3f}")
    print(f"  within-protein ρ  {ranking['median_within_protein_spearman']:.3f} (median)"
          f"   |  random {rand['median_within_protein_spearman']:+.3f}")
    print(f"  base rate ΔΔG>0   {ranking['base_rate_stabilizing']:.1%}")
    for k in sorted(tk, key=int):
        print(f"  top-{k:<3} precision  {tk[k]:.1%}   vs random {rk.get(k, float('nan')):.1%}")
    ci = ranking.get("headline_precision_ci95")
    if ci:
        print(f"  top-{ranking['headline_k']} 95% CI over proteins  "
              f"[{ci[0]:.1%}, {ci[1]:.1%}]"
              f"   ({ranking['proteins_below_half_precision']} proteins below 50%)")
    print(f"  ΔΔG(wt→wt)        {report['self_ddg_offset']['mean']:+.3f} kcal/mol "
          f"(per fold: {', '.join(f'{v:+.2f}' for v in report['self_ddg_offset']['per_fold'])})")
    print("=" * 68)
    print(f"  written to {out_path}\n")
    return report


def main() -> None:
    p = argparse.ArgumentParser(
        description="Grouped K-fold CV benchmark for the served ΔΔG model"
    )
    p.add_argument("--data-path", type=Path, default=None)
    p.add_argument("--checkpoint-dir", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=12, help="max epochs per fold")
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train-rows", type=int, default=None,
                   help="subsample training rows per fold (speed)")
    p.add_argument("--quick", action="store_true",
                   help="2 folds, 1 epoch, 40 parents, 20k rows — wiring smoke test")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    out_path = args.out
    if args.quick:
        args.folds, args.epochs, args.patience = 2, 1, 1
        args.max_train_rows, max_parents = 20_000, 40
        # Never let a smoke run land on the real report's path — a 1-epoch,
        # 40-parent result must not be mistakable for the benchmark.
        if out_path is None:
            ckpt = args.checkpoint_dir or settings.checkpoint_dir
            out_path = ckpt / "ddg_grouped_cv_SMOKE.json"
        logger.warning(
            "--quick is a wiring smoke test (1 epoch, 40 parents). Its numbers are "
            "NOT a benchmark result; writing to %s", out_path.name,
        )
    else:
        max_parents = None

    run(
        data_path=args.data_path,
        checkpoint_dir=args.checkpoint_dir,
        out_path=out_path,
        folds=args.folds,
        epochs=args.epochs,
        patience=args.patience,
        device=args.device,
        seed=args.seed,
        max_train_rows=args.max_train_rows,
        max_parents=max_parents,
    )


if __name__ == "__main__":
    main()
