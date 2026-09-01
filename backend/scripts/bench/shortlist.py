"""The shortlist test — what the optimizer's top-k is actually worth, externally.

Why a separate script. Top-k precision only means something when a protein has
substantially more than k scored mutations. S669 averages ~7 mutations per
protein, so "top 5 of 7" is nearly "all of them" and the metric collapses toward
the base rate — which is why every method on S669 lands in a narrow 0.30–0.42
band that discriminates nothing. Ssym is the opposite shape: few proteins, but
several of them very deep (1L63A has 118 mutations, 2LZMA 66, 1LZ1A 61). Those
are the only external proteins where a site-saturation shortlist can honestly be
tested at all.

Statistics. With 4–5 qualifying proteins, a bootstrap *over proteins* — the right
choice everywhere else in this suite — has no power and would be misleading. So
this reports **per protein**, and resamples mutations *within* each protein for
the interval. That measures a different thing (how stable this protein's estimate
is), and it is labelled as such; it is not a generalization claim across
proteins. The per-protein numbers are the result, and the aggregate is shown only
as a summary of those few points.

The random control is computed by permutation rather than assumed: for each
protein, shuffle the scores many times and take the top-k. Its expectation is the
protein's base rate, and reporting the realised spread keeps the comparison
honest when n is small.

Comparable internal number: ``ddg_grouped_cv.json`` reports top-5 precision
0.736 against a 0.208 base rate, on 262 proteins with ≥20 mutations each. That is
the claim this script tests externally.

Run::

    uv run python -m scripts.bench.shortlist
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

from scripts.bench import core
from scripts.bench.ssym import fetch, load_ssym

logger = logging.getLogger("bench.shortlist")

_BLUE, _ORANGE, _GREEN, _RED, _GREY = "#2f6f9f", "#c26a12", "#1d874b", "#c0392f", "#8fa0ad"

# Matches app.training.calibrate_ddg.MIN_MUTATIONS_PER_PROTEIN, so the external
# number is measured under the same rule as the internal one it is compared to.
MIN_MUTATIONS = 20
TOP_K = (1, 3, 5, 10)

# The internal claim under test (checkpoints/ddg_grouped_cv.json).
INTERNAL_REFERENCE = {
    "source": "ddg_grouped_cv.json",
    "n_proteins": 262,
    "base_rate": 0.2078,
    "top_k_precision": {"1": 0.7328, "3": 0.7405, "5": 0.7359, "10": 0.7031},
}


def _topk_precision(scores: np.ndarray, stabilizing: np.ndarray, k: int) -> float:
    order = np.argsort(-scores)[:k]
    return float(np.mean(stabilizing[order]))


def _random_control(stabilizing: np.ndarray, k: int, n_perm: int, rng) -> dict:
    """Permutation control: shuffle scores, take top-k, repeat."""
    values = [
        float(np.mean(stabilizing[rng.permutation(len(stabilizing))[:k]]))
        for _ in range(n_perm)
    ]
    return {
        "mean": float(np.mean(values)),
        "ci95": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))],
    }


def _within_bootstrap(scores, stabilizing, k, n_boot, rng) -> list[float]:
    """Resample mutations within this protein — an interval on *this* estimate."""
    out = []
    n = len(scores)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if stabilizing[idx].sum() == 0:
            continue
        out.append(_topk_precision(scores[idx], stabilizing[idx], k))
    return out


def run(data_dir: Path, out_dir: Path, plots_dir: Path,
        min_mutations: int, n_boot: int) -> dict:
    from app.services.esm2 import get_esm2_service
    from app.services.model_store import get_model_store

    fetch(data_dir)
    variants, manifest = load_ssym(data_dir)

    proteins = np.array([v.protein for v in variants])
    counts = {p: int((proteins == p).sum()) for p in np.unique(proteins)}
    deep = sorted([p for p, c in counts.items() if c >= min_mutations],
                  key=lambda p: -counts[p])
    if not deep:
        raise SystemExit(f"no protein has >= {min_mutations} mutations")
    logger.info("deep proteins (>=%d mutations): %s",
                min_mutations, {p: counts[p] for p in deep})

    keep = np.isin(proteins, deep)
    variants = [v for v, k in zip(variants, keep) if k]
    proteins = proteins[keep]
    true_dir = np.array([v.ddg_dir for v in variants], dtype=np.float32)

    store = get_model_store()
    store.load()
    esm2 = get_esm2_service()
    cache_root = Path("cache/bench")

    cache = core.EmbedCache(cache_root / "embeddings")
    embeddings = cache.get_many(
        [v.wt_seq for v in variants] + [v.mut_seq for v in variants], esm2)
    ddg_dir, _ = core.model_ddg_scores(variants, embeddings, store)
    zs_dir, _ = core.esm_zeroshot_scores(variants, esm2, cache_root / "zeroshot")

    rng = np.random.default_rng(0)
    methods = {"model_ddg": ddg_dir, "esm2_zeroshot": zs_dir}

    per_protein = {}
    for protein in deep:
        mask = proteins == protein
        stabilizing = (true_dir[mask] > 0).astype(float)
        entry = {
            "n_mutations": int(mask.sum()),
            "n_stabilizing": int(stabilizing.sum()),
            "base_rate": float(stabilizing.mean()),
            "methods": {},
            "random": {},
        }
        for name, scores in methods.items():
            entry["methods"][name] = {}
            for k in TOP_K:
                if mask.sum() < k:
                    continue
                precision = _topk_precision(scores[mask], stabilizing, k)
                boot = _within_bootstrap(scores[mask], stabilizing, k, n_boot, rng)
                entry["methods"][name][str(k)] = {
                    "precision": precision,
                    "enrichment_vs_base_rate": (
                        float(precision / entry["base_rate"]) if entry["base_rate"] > 0 else None),
                    "within_protein_ci95": (
                        [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
                        if boot else None),
                }
        for k in TOP_K:
            if mask.sum() >= k:
                entry["random"][str(k)] = _random_control(stabilizing, k, n_boot, rng)
        per_protein[protein] = entry

    # Aggregate across the few deep proteins. Reported as a summary of the
    # per-protein points above, not as an independent generalization estimate.
    aggregate = {}
    for name in methods:
        aggregate[name] = {}
        for k in TOP_K:
            values = [per_protein[p]["methods"][name][str(k)]["precision"]
                      for p in deep if str(k) in per_protein[p]["methods"][name]]
            aggregate[name][str(k)] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "n_proteins": len(values),
            }
    aggregate["random"] = {
        str(k): {"mean": float(np.mean([per_protein[p]["random"][str(k)]["mean"]
                                        for p in deep if str(k) in per_protein[p]["random"]]))}
        for k in TOP_K}
    aggregate["base_rate_mean"] = float(np.mean([per_protein[p]["base_rate"] for p in deep]))

    report = {
        "benchmark": "ssym_deep_shortlist",
        "purpose": "External test of the optimizer's top-k shortlist on the only "
                   "external proteins deep enough to support it.",
        "dataset": manifest.get("dataset", "Ssym"),
        "reference": manifest.get("reference"),
        "min_mutations_per_protein": min_mutations,
        "statistics_note": "Intervals resample mutations WITHIN a protein; with "
                           f"{len(deep)} qualifying proteins a bootstrap over proteins "
                           "would have no power. Per-protein values are the result.",
        "n_proteins": len(deep),
        "proteins": {p: counts[p] for p in deep},
        "internal_reference_under_test": INTERNAL_REFERENCE,
        "per_protein": per_protein,
        "aggregate": aggregate,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ssym_shortlist.json").write_text(json.dumps(report, indent=1))
    _plot(report, deep, plots_dir)
    _markdown(report, deep, out_dir / "ssym_shortlist.md")
    return report


def _plot(report, deep, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    width = 0.26
    x = np.arange(len(deep))
    for offset, (name, colour) in zip(
            (-width, 0, width),
            (("model_ddg", _RED), ("esm2_zeroshot", _ORANGE), ("random", _GREY))):
        values = []
        for p in deep:
            entry = report["per_protein"][p]
            if name == "random":
                values.append(entry["random"].get("5", {}).get("mean", np.nan))
            else:
                values.append(entry["methods"][name].get("5", {}).get("precision", np.nan))
        ax.bar(x + offset, values, width, color=colour, label=name)
    base = [report["per_protein"][p]["base_rate"] for p in deep]
    ax.plot(x, base, "k_", markersize=28, markeredgewidth=2, label="base rate")
    ax.axhline(report["internal_reference_under_test"]["top_k_precision"]["5"],
               color=_GREEN, ls="--", lw=1.2, label="internal grouped-CV (0.736)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}\n(n={report['proteins'][p]})" for p in deep], fontsize=8)
    ax.set_ylabel("top-5 precision")
    ax.set_title("Shortlist precision per deep protein (k=5)")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    for name, colour in (("model_ddg", _RED), ("esm2_zeroshot", _ORANGE)):
        ks = [k for k in TOP_K]
        means = [report["aggregate"][name][str(k)]["mean"] for k in ks]
        lo = [report["aggregate"][name][str(k)]["min"] for k in ks]
        hi = [report["aggregate"][name][str(k)]["max"] for k in ks]
        ax.plot(ks, means, "o-", color=colour, label=f"{name} (mean over {len(deep)})")
        ax.fill_between(ks, lo, hi, color=colour, alpha=0.15)
    ax.plot(TOP_K, [report["aggregate"]["random"][str(k)]["mean"] for k in TOP_K],
            "s--", color=_GREY, label="random (permutation)")
    ax.axhline(report["aggregate"]["base_rate_mean"], color="k", ls=":", lw=1,
               label=f"base rate {report['aggregate']['base_rate_mean']:.3f}")
    ax.axhline(report["internal_reference_under_test"]["top_k_precision"]["5"],
               color=_GREEN, ls="--", lw=1.2, label="internal grouped-CV k=5")
    ax.set_xlabel("k"); ax.set_ylabel("precision")
    ax.set_title("Shortlist precision vs k\n(band = min–max across deep proteins)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle("Ssym deep-protein shortlist test — the optimizer's actual task", fontsize=13)
    fig.tight_layout()
    fig.savefig(plots_dir / "ssym_shortlist.png", dpi=150)
    plt.close(fig)


def _markdown(report, deep, path: Path) -> None:
    agg = report["aggregate"]
    lines = [
        "# Ssym deep-protein shortlist test",
        "",
        f"The only external proteins with ≥{report['min_mutations_per_protein']} scored "
        f"mutations — {report['n_proteins']} of them. S669 cannot support this test "
        "(~7 mutations per protein), which is why it is run here.",
        "",
        f"*{report['statistics_note']}*",
        "",
        "| protein | n | stabilizing | base rate | model_ddg @5 | enrichment | zero-shot @5 | random @5 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for p in deep:
        e = report["per_protein"][p]
        m = e["methods"]["model_ddg"].get("5", {})
        z = e["methods"]["esm2_zeroshot"].get("5", {})
        r = e["random"].get("5", {})
        enr = m.get("enrichment_vs_base_rate")
        ci = m.get("within_protein_ci95") or [float("nan"), float("nan")]
        enr_text = f"{enr:.2f}×" if enr else "—"
        lines.append(
            f"| {p} | {e['n_mutations']} | {e['n_stabilizing']} | {e['base_rate']:.3f} "
            f"| **{m.get('precision', float('nan')):.3f}** [{ci[0]:.2f}–{ci[1]:.2f}] "
            f"| {enr_text} "
            f"| {z.get('precision', float('nan')):.3f} "
            f"| {r.get('mean', float('nan')):.3f} |")
    lines += [
        "",
        "## Aggregate over the deep proteins",
        "",
        "| k | model_ddg | zero-shot | random | internal grouped-CV |",
        "|---|---|---|---|---|",
    ]
    for k in TOP_K:
        internal = report["internal_reference_under_test"]["top_k_precision"].get(str(k))
        internal_text = f"{internal:.3f}" if internal else "—"
        lines.append(
            f"| {k} | {agg['model_ddg'][str(k)]['mean']:.3f} "
            f"(range {agg['model_ddg'][str(k)]['min']:.2f}–{agg['model_ddg'][str(k)]['max']:.2f}) "
            f"| {agg['esm2_zeroshot'][str(k)]['mean']:.3f} "
            f"| {agg['random'][str(k)]['mean']:.3f} "
            f"| {internal_text} |")
    lines += ["", f"Mean base rate across these proteins: "
                  f"**{agg['base_rate_mean']:.3f}**.", ""]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/bench"))
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--plots-dir", type=Path, default=Path("checkpoints/plots"))
    parser.add_argument("--min-mutations", type=int, default=MIN_MUTATIONS)
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()
    report = run(args.data_dir, args.out_dir, args.plots_dir,
                 args.min_mutations, args.n_boot)
    print(json.dumps(report["aggregate"], indent=1))


if __name__ == "__main__":
    main()
