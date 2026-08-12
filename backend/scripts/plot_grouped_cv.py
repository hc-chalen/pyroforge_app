"""Figures for the grouped-K-fold ΔΔG benchmark.

``benchmark_grouped_cv`` writes metrics as JSON and every out-of-fold prediction
as an ``.npz``; this turns those into the four plots worth showing someone. It
reads only those two files — it never touches the model, so it is cheap to re-run
and cannot disagree with the numbers in the report.

The panels, in the order they answer the question "is the shortlist worth
anything":

  A  per-protein Spearman, model vs a random-ranking control — the *distribution*,
     because the median alone hides the proteins where ranking fails outright;
  B  precision@k against the random control and the base rate of stabilizing
     mutations, which is the honest comparison (a shortlist that beats nothing
     beats nothing);
  C  predicted vs true ΔΔG pooled over all folds, showing the regression-to-the-
     mean that makes the *level* untrustworthy even where the *ranking* is good;
  D  the per-protein top-5 precision distribution — the spread behind panel B.

Run::

    uv run python -m scripts.plot_grouped_cv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

MIN_MUTATIONS_PER_PROTEIN = 20
TOP_K = (1, 3, 5, 10)
HEADLINE_K = 5

# One colour for the model, one for every "this is what chance looks like"
# reference. Keeping the control visually flat is the point of the figure.
C_MODEL = "#c1440e"
C_CONTROL = "#9aa0a6"
C_BASE = "#3d5a80"


def per_protein_stats(
    pred: np.ndarray, true: np.ndarray, pid: np.ndarray, rng: np.random.Generator
) -> dict:
    """Recompute, per protein, exactly what ``ranking_precision`` reports in bulk.

    Same ≥20-mutation filter and same "stabilizing means true ΔΔG > 0" definition,
    so the aggregates here reproduce the JSON rather than describing a variant of
    it. The random control permutes predictions *within* each protein, which is
    the null the optimizer actually competes against: same candidate set, no skill
    in the ordering.
    """
    rho, rho_rand = [], []
    prec = {k: [] for k in TOP_K}
    prec_rand = {k: [] for k in TOP_K}
    base_rates = []

    for protein in np.unique(pid):
        mask = pid == protein
        if mask.sum() < MIN_MUTATIONS_PER_PROTEIN:
            continue
        p, t = pred[mask], true[mask]
        base_rates.append(float((t > 0).mean()))

        if np.std(p) > 0 and np.std(t) > 0:
            r = spearmanr(p, t).correlation
            if not np.isnan(r):
                rho.append(float(r))
            r_rand = spearmanr(rng.permutation(p), t).correlation
            if not np.isnan(r_rand):
                rho_rand.append(float(r_rand))

        order = np.argsort(-p)
        order_rand = rng.permutation(len(p))
        for k in TOP_K:
            prec[k].append(float((t[order[:k]] > 0).mean()))
            prec_rand[k].append(float((t[order_rand[:k]] > 0).mean()))

    return {
        "n_proteins": len(base_rates),
        "spearman": np.array(rho),
        "spearman_random": np.array(rho_rand),
        "precision": {k: np.array(v) for k, v in prec.items()},
        "precision_random": {k: np.array(v) for k, v in prec_rand.items()},
        "base_rate": float(np.mean(base_rates)),
    }


def panel_spearman(ax, s: dict) -> None:
    bins = np.linspace(-1, 1, 41)
    ax.hist(s["spearman_random"], bins=bins, color=C_CONTROL, alpha=0.75,
            label=f"random ranking (median {np.median(s['spearman_random']):+.2f})")
    ax.hist(s["spearman"], bins=bins, color=C_MODEL, alpha=0.85,
            label=f"model_ddg (median {np.median(s['spearman']):.2f})")
    ax.axvline(np.median(s["spearman"]), color=C_MODEL, lw=1.6, ls="--")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("within-protein Spearman ρ (predicted vs true ΔΔG)")
    ax.set_ylabel("proteins")
    ax.set_title(
        f"A · Ranking skill per protein (n={s['n_proteins']}, out-of-fold)",
        loc="left", fontweight="bold",
    )
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    # The left tail is the honest caveat, so label it rather than let it hide.
    n_bad = int((s["spearman"] < 0.3).sum())
    ax.text(0.02, 0.62, f"{n_bad} proteins below ρ=0.3", transform=ax.transAxes,
            fontsize=8, color="#444")


def panel_precision(ax, s: dict) -> None:
    ks = list(TOP_K)
    x = np.arange(len(ks))
    model = [s["precision"][k].mean() for k in ks]
    rand = [s["precision_random"][k].mean() for k in ks]
    err = [
        1.96 * s["precision"][k].std(ddof=1) / np.sqrt(len(s["precision"][k]))
        for k in ks
    ]

    ax.bar(x - 0.19, model, 0.38, color=C_MODEL, label="model_ddg shortlist",
           yerr=err, capsize=3, ecolor="#5a2408")
    ax.bar(x + 0.19, rand, 0.38, color=C_CONTROL, label="random shortlist")
    ax.axhline(s["base_rate"], color=C_BASE, ls="--", lw=1.4,
               label=f"base rate stabilizing ({s['base_rate']:.0%})")
    for xi, v, e in zip(x - 0.19, model, err):
        ax.text(xi, v + e + 0.025, f"{v:.0%}", ha="center", fontsize=9,
                fontweight="bold")

    ax.set_xticks(x, [f"top-{k}" for k in ks])
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction of shortlist truly stabilizing")
    ax.set_title("B · What a shortlist is worth", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="upper right")


def panel_scatter(ax, pred: np.ndarray, true: np.ndarray) -> None:
    r = pearsonr(pred, true)[0]
    mae = float(np.abs(pred - true).mean())
    lim = (-6, 3)
    hb = ax.hexbin(true, pred, gridsize=70, extent=(*lim, *lim), bins="log",
                   cmap="magma_r", linewidths=0)
    ax.plot(lim, lim, color="#2b2b2b", lw=1.0, ls="--", label="y = x")
    # Least-squares fit: its slope < 1 is the regression-to-the-mean that makes
    # absolute ΔΔG unusable as a physical quantity.
    slope, intercept = np.polyfit(true, pred, 1)
    xs = np.array(lim)
    ax.plot(xs, slope * xs + intercept, color=C_MODEL, lw=1.6,
            label=f"fit (slope {slope:.2f})")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("true ΔΔG (kcal/mol)")
    ax.set_ylabel("predicted ΔΔG (kcal/mol)")
    ax.set_title(
        f"C · Pooled out-of-fold fit — r={r:.3f}, MAE={mae:.2f} kcal/mol, n={len(pred):,}",
        loc="left", fontweight="bold",
    )
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    plt.colorbar(hb, ax=ax, label="mutations (log)", fraction=0.046, pad=0.02)


def panel_precision_spread(ax, s: dict) -> None:
    p = np.sort(s["precision"][HEADLINE_K])
    y = np.arange(1, len(p) + 1) / len(p)
    ax.step(p, y, where="post", color=C_MODEL, lw=2)
    ax.axvline(p.mean(), color=C_MODEL, ls="--", lw=1.2,
               label=f"mean {p.mean():.0%}")
    ax.axvline(s["base_rate"], color=C_BASE, ls="--", lw=1.2,
               label=f"base rate {s['base_rate']:.0%}")
    frac_bad = float((p < 0.5).mean())
    ax.axvspan(0, 0.5, color="#d94f4f", alpha=0.08)
    ax.text(0.02, 0.9, f"{frac_bad:.0%} of proteins ({int((p < 0.5).sum())})\n"
                       "have <50% of the top-5 stabilizing",
            transform=ax.transAxes, fontsize=8, color="#8a2020")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(f"top-{HEADLINE_K} precision for one protein")
    ax.set_ylabel("cumulative fraction of proteins")
    ax.set_title("D · Per-protein spread behind panel B", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="lower right")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path,
                    default=root / "checkpoints" / "ddg_grouped_cv.predictions.npz")
    ap.add_argument("--report", type=Path,
                    default=root / "checkpoints" / "ddg_grouped_cv.json")
    ap.add_argument("--out-dir", type=Path, default=root / "checkpoints" / "plots")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    d = np.load(args.predictions)
    pred, true, pid = (d["pred"].astype(np.float64), d["true"].astype(np.float64),
                       d["parent_id"])
    report = json.loads(args.report.read_text())

    rng = np.random.default_rng(args.seed)
    s = per_protein_stats(pred, true, pid, rng)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 150})

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    panel_spearman(axes[0, 0], s)
    panel_precision(axes[0, 1], s)
    panel_scatter(axes[1, 0], pred, true)
    panel_precision_spread(axes[1, 1], s)

    fig.suptitle(
        "PyroForge model_ddg — grouped K-fold cross-validation "
        f"({report['folds']} folds, {report['n_parents_total']} parent proteins, "
        f"{report['n_rows_scored']:,} mutations scored out-of-fold)",
        fontsize=13, fontweight="bold", y=0.985,
    )
    fig.text(0.5, 0.008,
             "Folds are grouped by parent protein: every protein is scored by a model "
             "that never saw it. Parents are not sequence-clustered, so homologues may "
             "straddle folds — treat as an upper bound for a genuinely novel fold.",
             ha="center", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.022, 1, 0.965))

    combined = args.out_dir / "ddg_grouped_cv.png"
    fig.savefig(combined, bbox_inches="tight")
    plt.close(fig)

    # Individual panels too — a slide usually wants one, not four.
    for name, fn, size in (
        ("ddg_grouped_cv_spearman", panel_spearman, (7, 4.6)),
        ("ddg_grouped_cv_precision", panel_precision, (7, 4.6)),
        ("ddg_grouped_cv_spread", panel_precision_spread, (7, 4.6)),
    ):
        f, ax = plt.subplots(figsize=size)
        fn(ax, s)
        f.tight_layout()
        f.savefig(args.out_dir / f"{name}.png", bbox_inches="tight")
        plt.close(f)

    f, ax = plt.subplots(figsize=(7, 5.4))
    panel_scatter(ax, pred, true)
    f.tight_layout()
    f.savefig(args.out_dir / "ddg_grouped_cv_scatter.png", bbox_inches="tight")
    plt.close(f)

    print(f"proteins scored      {s['n_proteins']}")
    print(f"median within-protein ρ  {np.median(s['spearman']):.3f} "
          f"(random {np.median(s['spearman_random']):+.3f})")
    for k in TOP_K:
        print(f"top-{k:<3} precision       {s['precision'][k].mean():.1%} "
              f"(random {s['precision_random'][k].mean():.1%})")
    print(f"base rate stabilizing    {s['base_rate']:.1%}")
    print(f"wrote {combined}")


if __name__ == "__main__":
    main()
