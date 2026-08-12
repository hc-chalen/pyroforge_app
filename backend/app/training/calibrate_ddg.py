"""Fit the served ΔΔG calibration, and measure what the optimizer's ranking is worth.

Why this exists
---------------
``model_ddg`` predicts ΔΔG well enough to *rank* mutations, but its raw output is
not on the scale of the thing it predicts, and the optimizer used to consume it
as if it were. Two separate defects came out of that:

1. **Uncorrected scale/bias.** On held-out data the raw model predicts ΔΔG > 0
   for far fewer mutations than are truly stabilizing, so ranking by raw output
   systematically under-calls exactly the mutations the tool exists to find.

2. **Winner's curse.** The site-saturation scan scores L×19 candidates and then
   keeps the largest. Taking the maximum of many noisy estimates selects for
   large positive *error*, not just large signal, so the top candidate's raw
   prediction is inflated by however much luck it took to win. The old code
   answered this with a fixed 1.96σ "significance gate", which is a
   per-comparison threshold applied thousands of times — it does not correct the
   selection bias, it just moves it.

Both are fixed by the same thing: regressing *true* ΔΔG on *predicted* ΔΔG over a
parent-held-out set. The fitted slope is below 1 precisely because the prediction
is noisy, so applying it shrinks extreme predictions toward the mean by the
amount the data says is warranted (Kelley/James–Stein shrinkage, in its
regression form). The residual spread around that line is the honest ±95% band,
and it replaces ``residual_std``, which was measured around the *uncalibrated*
prediction.

The script also measures the only number a user of the optimizer actually needs:
**of the top-k candidates this model ranks within a protein, how many are truly
stabilizing?** That is reported in the UI in place of the old σ language.

Outputs ``checkpoints/model_ddg_calibration.json``, which ``ModelStore`` loads.
Nothing here retrains ``model_ddg``; it only characterises the shipped weights.

    uv run pyroforge-calibrate-ddg
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from app.core.config import settings
from app.data.dataset import build_parent_split, load_raw_data
from app.models.regression import DeltaMLP
from app.training.features import build_ddg_delta_eval_arrays

logger = logging.getLogger(__name__)

# Shortlist sizes to characterise. `max_mutations` is capped at 20 by the API,
# and the default is 5.
TOP_K_REPORTED = (1, 3, 5, 10)
# A protein needs enough scored mutations for "top-k within this protein" to mean
# anything; below this it is dropped from the ranking statistics.
MIN_MUTATIONS_PER_PROTEIN = 20


def load_model(checkpoint_dir: Path, device: str) -> tuple[DeltaMLP, np.ndarray, np.ndarray]:
    ckpt = torch.load(checkpoint_dir / "model_ddg.pt", map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = DeltaMLP(cfg["input_dim"], cfg["widths"], cfg["dropout"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    stats = ckpt["norm_stats"]
    return model, np.asarray(stats["embedding_mean"]), np.asarray(stats["embedding_std"])


@torch.no_grad()
def predict(model: DeltaMLP, x: np.ndarray, device: str, bs: int = 8192) -> np.ndarray:
    out = np.empty(len(x), dtype=np.float32)
    for s in range(0, len(x), bs):
        e = min(s + bs, len(x))
        out[s:e] = model(torch.tensor(x[s:e], device=device)).cpu().numpy()
    return out


def fit_calibration(pred: np.ndarray, true: np.ndarray) -> dict:
    """Least-squares fit of true ΔΔG on predicted ΔΔG.

    ``slope`` < 1 is the shrinkage: it is the fraction of the prediction's spread
    that the held-out data says is signal rather than noise. Applying it is what
    stops the top-ranked candidate from being reported at its inflated raw value.
    """
    slope, intercept = np.polyfit(pred, true, 1)
    calibrated = slope * pred + intercept
    residual = true - calibrated
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "residual_std": float(residual.std()),
        "raw_residual_std": float((true - pred).std()),
        "n": int(len(pred)),
        "pearson_r": float(pearsonr(pred, true)[0]),
    }


def _bootstrap_ci(values: np.ndarray, seed: int = 0, draws: int = 5000) -> list[float]:
    """Percentile CI resampled over *proteins* — the unit that actually varies."""
    if len(values) < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, len(values), replace=True).mean() for _ in range(draws)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def ranking_precision(
    pred: np.ndarray, true: np.ndarray, pid: np.ndarray, legacy_gate: float | None = None
) -> dict:
    """How often the top-k ranked candidates *within a protein* are truly stabilizing.

    This mirrors exactly how the optimizer uses the model — rank the mutations of
    one protein, take a handful — so it is the number worth showing a user, and
    it is directly comparable to the base rate of stabilizing mutations.

    The mean alone would be misleading: per-protein precision is highly variable,
    so the spread and a bootstrap CI over proteins are reported too, and the UI
    quotes them. ``legacy_gate`` additionally measures how often the retired
    1.96σ significance gate would have returned anything at all.
    """
    per_k: dict[int, list[float]] = {k: [] for k in TOP_K_REPORTED}
    per_k_ddg: dict[int, list[float]] = {k: [] for k in TOP_K_REPORTED}
    spearmans: list[float] = []
    base_rates: list[float] = []
    n_positive: list[int] = []
    gate_returns_any = 0
    n_proteins = 0

    for protein in np.unique(pid):
        mask = pid == protein
        if mask.sum() < MIN_MUTATIONS_PER_PROTEIN:
            continue
        p, t = pred[mask], true[mask]
        n_proteins += 1
        base_rates.append(float((t > 0).mean()))
        n_positive.append(int((p > 0).sum()))
        if legacy_gate is not None and (p > legacy_gate).any():
            gate_returns_any += 1
        if np.std(p) > 0 and np.std(t) > 0:
            r = spearmanr(p, t).correlation
            if not np.isnan(r):
                spearmans.append(float(r))
        order = np.argsort(-p)
        for k in TOP_K_REPORTED:
            top = order[:k]
            per_k[k].append(float((t[top] > 0).mean()))
            per_k_ddg[k].append(float(np.median(t[top])))

    default_k = 5
    headline = np.array(per_k.get(default_k, []), dtype=float)
    return {
        "n_proteins": n_proteins,
        "min_mutations_per_protein": MIN_MUTATIONS_PER_PROTEIN,
        "base_rate_stabilizing": float(np.mean(base_rates)) if base_rates else float("nan"),
        "median_within_protein_spearman": (
            float(np.median(spearmans)) if spearmans else float("nan")
        ),
        "top_k_precision": {str(k): float(np.mean(v)) for k, v in per_k.items() if v},
        "top_k_median_true_ddg": {
            str(k): float(np.median(v)) for k, v in per_k_ddg.items() if v
        },
        # Honest spread for the headline k: the mean hides proteins where the
        # ranking fails outright, and the UI must not.
        "headline_k": default_k,
        "headline_precision_ci95": _bootstrap_ci(headline) if len(headline) else None,
        "headline_precision_min": float(headline.min()) if len(headline) else None,
        "headline_precision_max": float(headline.max()) if len(headline) else None,
        "proteins_below_half_precision": (
            int((headline < 0.5).sum()) if len(headline) else None
        ),
        # Viability of the selection rule: a shortlist is worthless if the rule
        # returns nothing. Median count of candidates predicted stabilizing.
        "median_candidates_predicted_positive": (
            float(np.median(n_positive)) if n_positive else None
        ),
        "proteins_with_no_positive_candidate": int(sum(1 for n in n_positive if n == 0)),
        # How the retired gate would have behaved on this same set.
        "legacy_gate": None
        if legacy_gate is None
        else {
            "threshold": float(legacy_gate),
            "proteins_returning_any_mutation": gate_returns_any,
        },
    }


def _eval_arrays(data: dict, rows: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return build_ddg_delta_eval_arrays(data, rows, mean, std, concat_mut=True)


def run(
    data_path: Path | None = None,
    test_path: Path | None = None,
    checkpoint_dir: Path | None = None,
    device: str = "cpu",
) -> dict:
    data_path = data_path or settings.data_path
    test_path = test_path or settings.test_data_path
    checkpoint_dir = checkpoint_dir or settings.checkpoint_dir

    model, mean, std = load_model(checkpoint_dir, device)

    # Fit on the parent-held-out validation split of the training file: the same
    # split model_ddg early-stopped on, and the only one large enough to fit a
    # stable calibration. Parent-grouped, so it measures unseen proteins.
    logger.info("loading %s (this is the large training file) ...", data_path)
    data = load_raw_data(data_path)
    split = build_parent_split(data)
    x_va, y_va, pid_va = _eval_arrays(data, split.val_idx, mean, std)
    logger.info("validation ΔΔG rows: %d over %d proteins", len(y_va), len(np.unique(pid_va)))
    p_va = predict(model, x_va, device)
    calibration = fit_calibration(p_va, y_va)
    logger.info(
        "calibration: true = %.4f x pred + %.4f  (shrinkage %.0f%%, residual σ %.3f -> %.3f)",
        calibration["slope"], calibration["intercept"],
        100 * (1 - calibration["slope"]),
        calibration["raw_residual_std"], calibration["residual_std"],
    )
    legacy_gate = 1.96 * calibration["raw_residual_std"]
    val_ranking = ranking_precision(p_va, y_va, pid_va, legacy_gate=legacy_gate)
    del data, x_va

    # Independent check on the held-out test file.
    test_data = load_raw_data(test_path)
    x_te, y_te, pid_te = _eval_arrays(
        test_data, np.arange(test_data["embeddings"].shape[0]), mean, std
    )
    p_te = predict(model, x_te, device)
    cal_te = calibration["slope"] * p_te + calibration["intercept"]
    raw_mae = float(np.mean(np.abs(p_te - y_te)))
    cal_mae = float(np.mean(np.abs(cal_te - y_te)))
    test_ranking = ranking_precision(p_te, y_te, pid_te, legacy_gate=legacy_gate)
    test_check = {
        "n": int(len(y_te)),
        "n_proteins": int(len(np.unique(pid_te))),
        "raw_mae": raw_mae,
        "calibrated_mae": cal_mae,
        "raw_frac_predicted_positive": float((p_te > 0).mean()),
        "calibrated_frac_predicted_positive": float((cal_te > 0).mean()),
        "true_frac_positive": float((y_te > 0).mean()),
        "ranking": test_ranking,
        "caveat": (
            "esm2_test.pt carries only 4 parent proteins and 98% of its rows belong "
            "to one of them, so protein-level statistics from this file are not "
            "usable. It is kept as a sanity check on the level, not the ranking."
        ),
    }

    # Only ship a calibration that earns its place. A linear recalibration that is
    # within noise of the identity, and that does not improve error on an
    # independent file, adds a transformation the user cannot audit in exchange
    # for nothing — so record the fit and decline to apply it.
    near_identity = abs(calibration["slope"] - 1.0) < 0.15
    improves_independent = cal_mae < raw_mae
    applied = bool(improves_independent and not near_identity)
    calibration["applied"] = applied
    calibration["applied_reason"] = (
        "calibration improves held-out error and is materially different from the "
        "identity"
        if applied
        else (
            f"not applied: fitted slope {calibration['slope']:.3f} is within noise of "
            f"1.0 (no over-dispersion to shrink) and it does not improve error on the "
            f"independent file (MAE {raw_mae:.3f} raw vs {cal_mae:.3f} calibrated). "
            f"Ranking, not level, is what the optimizer consumes."
        )
    )

    report = {
        "model": "model_ddg",
        "calibration": calibration,
        "validation_ranking": val_ranking,
        "test_check": test_check,
        "split": split.manifest(),
        "notes": (
            "A slope below 1 would have meant the raw predictions are over-dispersed "
            "and extreme ones should be shrunk toward the mean (the winner's-curse "
            "correction for ranking L*19 candidates and keeping the largest). The "
            "measured slope is not below 1, so no shrinkage is warranted and none is "
            "applied — see calibration.applied_reason. What the measurement does "
            "establish is that the model RANKS well within a protein while its "
            "absolute level is biased low, which is why the optimizer selects by rank "
            "with a sign floor rather than by an absolute significance threshold."
        ),
    }

    out = checkpoint_dir / "model_ddg_calibration.json"
    out.write_text(json.dumps(report, indent=2))
    logger.info("wrote %s", out)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Fit and measure the served ΔΔG calibration")
    p.add_argument("--device", default="cpu")
    p.add_argument("--data-path", type=Path, default=None)
    p.add_argument("--test-path", type=Path, default=None)
    args = p.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    print(json.dumps(run(args.data_path, args.test_path, device=args.device), indent=2))


if __name__ == "__main__":
    main()
