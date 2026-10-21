"""Why does model_tm report Pearson 0.810 but Spearman 0.621?

A 0.19 gap in that direction is a warning sign, not a result. Pearson rewards a
good *linear* fit; Spearman only rewards correct *ordering*. Pearson can only
exceed Spearman by that much for one of two reasons:

  1. **High-leverage points.** A handful of extreme Tm values sit far from the
     bulk, and getting them roughly right inflates r while the crowded middle —
     where nearly every real query lands — is ranked poorly.
  2. **Cluster separation.** The data contains groups with very different mean
     Tm (in the Meltome Atlas, thermophiles vs mesophiles). Predicting *which
     group* a protein belongs to produces a strong global correlation even if
     ranking proteins *within* a group is near-random.

Both are tested here, and they have the same practical consequence: a user
optimizing one protein needs within-group resolution, which neither failure mode
provides. If either is confirmed, the README's headline r = 0.81 overstates
usable accuracy and Spearman should lead instead.

**No species labels are available.** ``meltome_tm.npz`` stores only
``emb/tm/split/source``, and the upstream FLIP FASTA anonymizes its headers
(``>Sequence0``), so the organism cannot be recovered from either. K-means over
the ESM-2 embeddings is used as a proxy for group structure instead. This is
arguably the better test regardless: it asks whether the correlation is carried
by *any* coarse grouping the model can already see, which is the mechanism that
matters, without needing the groups to be species.

Run::

    uv run python -m scripts.bench.tm_diagnosis
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger("bench.tm")

_BLUE, _ORANGE, _GREEN, _RED, _GREY = "#2f6f9f", "#c26a12", "#1d874b", "#c0392f", "#8fa0ad"


def _square_axes(ax, lo: float, hi: float, pad: float = 0.04) -> None:
    """Give x and y the same data scale so one unit has the same length on both."""
    span = float(hi) - float(lo)
    margin = pad * span if span > 0 else 1.0
    ax.set_xlim(lo - margin, hi + margin)
    ax.set_ylim(lo - margin, hi + margin)
    ax.set_aspect("equal", adjustable="box")


def _corr(pred, true) -> tuple[float, float]:
    from scipy.stats import pearsonr, spearmanr

    if len(pred) < 3 or np.std(pred) == 0 or np.std(true) == 0:
        return float("nan"), float("nan")
    return float(pearsonr(pred, true)[0]), float(spearmanr(pred, true).correlation)


def run(npz_path: Path, model_path: Path, out_dir: Path, plots_dir: Path) -> dict:
    import joblib
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture

    data = np.load(npz_path, allow_pickle=True)
    emb, tm, split = data["emb"], data["tm"], data["split"]
    test = split == "test"
    x, y = emb[test], tm[test].astype(np.float64)

    # The checkpoint is a dict wrapper (model + recipe + residual_std + reported
    # metrics), matching how train_tm_meltome saves and ModelStore loads it.
    bundle = joblib.load(model_path)
    model = bundle["model"] if isinstance(bundle, dict) else bundle
    pred = np.asarray(model.predict(x), dtype=np.float64)
    if isinstance(bundle, dict) and "test_metrics" in bundle:
        report_reference = bundle["test_metrics"]
    else:
        report_reference = {}

    pearson, spearman = _corr(pred, y)
    report = {
        "diagnosis": "model_tm pearson/spearman gap",
        "n_test": int(test.sum()),
        # Reproduce the shipped tm_report.json numbers before interpreting them —
        # if these disagree, the diagnosis is of a different model than the one served.
        "reported_in_checkpoint": report_reference,
        "global": {"pearson_r": pearson, "spearman_rho": spearman,
                   "gap": pearson - spearman},
        "tm_distribution": {
            "mean": float(y.mean()), "std": float(y.std()),
            "min": float(y.min()), "max": float(y.max()),
            "q05": float(np.percentile(y, 5)), "q50": float(np.percentile(y, 50)),
            "q95": float(np.percentile(y, 95)),
        },
    }

    # --- 1. Is the Tm distribution bimodal? -------------------------------- #
    # A two-component mixture fitting decisively better than one is the
    # signature of the thermophile/mesophile split the Meltome Atlas contains.
    bic = {}
    for k in (1, 2, 3):
        gm = GaussianMixture(n_components=k, random_state=0).fit(y.reshape(-1, 1))
        bic[k] = float(gm.bic(y.reshape(-1, 1)))
    best_k = min(bic, key=bic.get)
    report["bimodality"] = {
        "bic_by_components": bic,
        "best_n_components": best_k,
        "bimodal": best_k > 1,
        "bic_improvement_1_to_2": bic[1] - bic[2],
    }

    # --- 2. Outlier leverage, corrected for range restriction --------------- #
    # Trimming extremes *always* lowers correlation, even for a flawless model,
    # because it shrinks the spread of the target. So the raw drop proves nothing.
    # The Thorndike case-II formula gives the drop expected from restriction
    # alone; only the excess beyond that is evidence of leverage.
    sd_full = float(y.std())
    r_full = pearson
    leverage = {}
    for pct in (1, 5, 10, 20):
        lo, hi = np.percentile(y, pct), np.percentile(y, 100 - pct)
        keep = (y >= lo) & (y <= hi)
        p, s = _corr(pred[keep], y[keep])
        u = float(y[keep].std()) / sd_full
        expected = r_full * u / np.sqrt(1 - r_full**2 + r_full**2 * u**2)
        leverage[f"trim_{pct}pct"] = {
            "n": int(keep.sum()), "sd_ratio_u": u,
            "pearson_r": p, "pearson_r_expected_from_restriction": float(expected),
            "excess_beyond_restriction": float(p - expected), "spearman_rho": s,
        }
    report["outlier_leverage"] = leverage

    # --- 2b. Where does the skill actually live? ---------------------------- #
    # The decisive test, and deliberately stratified on the *prediction* rather
    # than the truth: conditioning on observed Tm would select against the model
    # and inflate the apparent failure. A constant baseline fitted on the
    # training split (no peeking) is the thing to beat — if the model cannot beat
    # a constant within a stratum, its output carries no usable information there.
    const = float(tm[~test].astype(np.float64).mean())
    edges = np.percentile(pred, [0, 20, 40, 60, 80, 100])
    strata = []
    for i in range(5):
        keep = (pred >= edges[i]) & (pred <= edges[i + 1])
        if keep.sum() < 20:
            continue
        p, s = _corr(pred[keep], y[keep])
        strata.append({
            "predicted_tm_range": [float(edges[i]), float(edges[i + 1])],
            "n": int(keep.sum()), "pearson_r": p, "spearman_rho": s,
            "model_mae": float(np.mean(np.abs(pred[keep] - y[keep]))),
            "constant_mae": float(np.mean(np.abs(const - y[keep]))),
            "true_sd": float(y[keep].std()),
        })
    report["skill_by_predicted_stratum"] = {
        "constant_baseline_c": const,
        "full_test_model_mae": float(np.mean(np.abs(pred - y))),
        "full_test_constant_mae": float(np.mean(np.abs(const - y))),
        "strata": strata,
        "strata_where_model_beats_constant": sum(
            1 for s in strata if s["model_mae"] < s["constant_mae"]),
    }

    # --- 3. Between- vs within-cluster decomposition ------------------------ #
    # The decisive test. Centering both series by cluster mean removes all
    # between-group signal; what survives is the ranking skill a user actually
    # gets when optimizing a single protein.
    clusters = {}
    for k in (2, 3, 5, 10):
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(x)
        centred_pred = pred.copy()
        centred_true = y.copy()
        per_cluster = []
        for c in np.unique(labels):
            mask = labels == c
            centred_pred[mask] -= pred[mask].mean()
            centred_true[mask] -= y[mask].mean()
            if mask.sum() >= 20:
                cp, cs = _corr(pred[mask], y[mask])
                per_cluster.append({"cluster": int(c), "n": int(mask.sum()),
                                    "pearson_r": cp, "spearman_rho": cs})
        # Between: how well cluster means alone track true cluster means.
        means_pred = np.array([pred[labels == c].mean() for c in np.unique(labels)])
        means_true = np.array([y[labels == c].mean() for c in np.unique(labels)])
        bp, bs = _corr(means_pred, means_true)
        wp, ws = _corr(centred_pred, centred_true)
        clusters[f"k={k}"] = {
            "between_cluster_pearson": bp,
            "within_cluster_pearson": wp,
            "within_cluster_spearman": ws,
            "median_per_cluster_spearman": float(np.median(
                [c["spearman_rho"] for c in per_cluster])) if per_cluster else float("nan"),
            "n_clusters_scored": len(per_cluster),
            "per_cluster": per_cluster,
        }
    report["cluster_decomposition"] = clusters

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tm_diagnosis.json").write_text(json.dumps(report, indent=1))
    _plot(report, y, pred, plots_dir)
    return report


def _plot(report, y, pred, plots_dir: Path) -> None:
    from sklearn.cluster import KMeans  # noqa: F401  (labels recomputed below only if needed)

    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    ax.hist(y, bins=50, color=_BLUE, alpha=0.8)
    bim = report["bimodality"]
    ax.set_xlabel("experimental Tm (°C)"); ax.set_ylabel("proteins")
    ax.set_title(f"Tm distribution — best fit {bim['best_n_components']} component(s)\n"
                 f"(BIC 1→2 improvement {bim['bic_improvement_1_to_2']:.0f})")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.scatter(pred, y, s=8, alpha=0.4, color=_BLUE)
    lims = [min(pred.min(), y.min()) - 2, max(pred.max(), y.max()) + 2]
    ax.plot(lims, lims, "k--", lw=0.8)
    _square_axes(ax, *lims, pad=0)
    g = report["global"]
    ax.set_xlabel("predicted Tm (°C)"); ax.set_ylabel("experimental Tm (°C)")
    ax.set_title(f"model_tm on Meltome test\nr={g['pearson_r']:.3f}  ρ={g['spearman_rho']:.3f}"
                 f"  (gap {g['gap']:.3f})")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    lev = report["outlier_leverage"]
    keys = list(lev)
    xs = np.arange(len(keys))
    ax.bar(xs - 0.2, [lev[k]["pearson_r"] for k in keys], width=0.4, color=_BLUE,
           label="observed r")
    ax.bar(xs + 0.2, [lev[k]["pearson_r_expected_from_restriction"] for k in keys],
           width=0.4, color=_GREY, label="r expected from range restriction alone")
    ax.axhline(report["global"]["pearson_r"], color="k", ls="--", lw=1,
               label=f"untrimmed r={report['global']['pearson_r']:.2f}")
    ax.set_xticks(xs); ax.set_xticklabels([k.replace("trim_", "") for k in keys])
    ax.set_xlabel("extremes trimmed"); ax.set_ylabel("Pearson r")
    ax.set_title("Trimming: observed vs restriction-predicted\n"
                 "(matching bars = no outlier-leverage problem)")
    ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 1]
    strat = report["skill_by_predicted_stratum"]
    rows = strat["strata"]
    xs = np.arange(len(rows))
    ax.bar(xs - 0.2, [r["model_mae"] for r in rows], width=0.4, color=_BLUE,
           label="model MAE")
    ax.bar(xs + 0.2, [r["constant_mae"] for r in rows], width=0.4, color=_GREY,
           label=f"constant {strat['constant_baseline_c']:.1f} °C MAE")
    for i, r in enumerate(rows):
        ax.text(i, max(r["model_mae"], r["constant_mae"]) + 0.4,
                f"ρ={r['spearman_rho']:+.2f}", ha="center", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{r['predicted_tm_range'][0]:.0f}–{r['predicted_tm_range'][1]:.0f}"
                        for r in rows], fontsize=8)
    ax.set_xlabel("predicted Tm quintile (°C)"); ax.set_ylabel("MAE (°C)")
    ax.set_title("Where the skill lives: only the hot tail\n"
                 "(lower is better; ρ is within-stratum ranking)")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    fig.suptitle("model_tm — is r=0.81 real usable accuracy?", fontsize=13)
    fig.tight_layout()
    fig.savefig(plots_dir / "tm_diagnosis.png", dpi=150)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, default=Path("checkpoints/improved/meltome_tm.npz"))
    parser.add_argument("--model", type=Path, default=Path("checkpoints/improved/model_tm.joblib"))
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--plots-dir", type=Path, default=Path("checkpoints/plots"))
    args = parser.parse_args()
    report = run(args.npz, args.model, args.out_dir, args.plots_dir)
    print(json.dumps({k: v for k, v in report.items() if k != "cluster_decomposition"}, indent=1))
    for k, v in report["cluster_decomposition"].items():
        print(f"{k}: between={v['between_cluster_pearson']:.3f} "
              f"within={v['within_cluster_pearson']:.3f} "
              f"median_per_cluster_rho={v['median_per_cluster_spearman']:.3f}")


if __name__ == "__main__":
    main()
