"""External benchmark of the served ΔΔG path against S669.

Answers three questions the repo could not previously answer:

1. **Does the trained head beat a free baseline?** ``model_ddg`` vs frozen ESM-2
   masked-marginal log-odds, which needs no trained head at all, and vs DDGun,
   which needs no training data at all.
2. **Does it hold up against published methods?** The S669 archive carries 20
   tools' predictions on these exact variants, so the comparison is exact rather
   than quoted across papers with different splits.
3. **Is the enforced antisymmetry real?** S669 ships both directions, so the
   Pucci 2018 bias statistics can be measured directly.

Everything is reported split by training regime (≤400 residues vs above), since
``model_ddg`` was fit on small designed domains and a pooled number across both
strata would not be interpretable.

Run::

    uv run python -m scripts.bench.run_s669
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.bench import core
from scripts.bench.data import SEQUENCE_ONLY_TOOLS, fetch, load_s669

logger = logging.getLogger("bench.s669")

_BLUE, _ORANGE, _GREEN, _RED, _GREY = "#2f6f9f", "#c26a12", "#1d874b", "#c0392f", "#8fa0ad"


def _square_axes(ax, lo: float, hi: float, pad: float = 0.04) -> None:
    """Give x and y the same data scale so one unit has the same length on both."""
    span = float(hi) - float(lo)
    margin = pad * span if span > 0 else 1.0
    ax.set_xlim(lo - margin, hi + margin)
    ax.set_ylim(lo - margin, hi + margin)
    ax.set_aspect("equal", adjustable="box")


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def evaluate(name, pred_dir, pred_inv, variants, true_dir, proteins) -> dict:
    """All metrics for one method."""
    result = {
        "method": name,
        "direct": core.basic_metrics(pred_dir, true_dir),
        "antisymmetry": core.antisymmetry(pred_dir, pred_inv),
        "within_protein_spearman": core.within_protein_spearman(pred_dir, true_dir, proteins),
        "stabilizing_detection": core.stabilizing_detection(pred_dir, true_dir),
        "top5_precision": core.top_k_precision(pred_dir, true_dir, proteins, k=5),
    }
    in_regime = np.array([v.in_regime for v in variants])
    if in_regime.any():
        result["direct_in_regime"] = core.basic_metrics(pred_dir[in_regime], true_dir[in_regime])
    if (~in_regime).any():
        result["direct_out_of_regime"] = core.basic_metrics(
            pred_dir[~in_regime], true_dir[~in_regime])
    return result


def run(data_dir: Path, out_dir: Path, plots_dir: Path, limit: int | None) -> dict:
    from app.services.esm2 import get_esm2_service
    from app.services.model_store import get_model_store

    fetch(data_dir)
    variants, manifest = load_s669(data_dir)
    if limit:
        variants = variants[:limit]
        manifest["limited_to"] = limit
    proteins = np.array([v.protein for v in variants])
    true_dir = np.array([v.ddg_dir for v in variants], dtype=np.float32)
    true_inv = np.array([v.ddg_inv for v in variants], dtype=np.float32)

    sign = core.detect_sign_convention(true_dir)
    logger.info("S669 sign convention: %s", sign)

    store = get_model_store()
    store.load()
    if store.model_ddg is None:
        raise SystemExit("model_ddg not loaded — nothing to benchmark")
    esm2 = get_esm2_service()

    cache_root = Path("cache/bench")
    started = time.time()

    logger.info("embedding %d variants (%d unique sequences) ...",
                len(variants), len({v.wt_seq for v in variants} | {v.mut_seq for v in variants}))
    cache = core.EmbedCache(cache_root / "embeddings")
    embeddings = cache.get_many([v.wt_seq for v in variants] + [v.mut_seq for v in variants], esm2)
    embed_seconds = time.time() - started
    logger.info("embeddings done in %.1fs (cache hits=%d misses=%d)",
                embed_seconds, cache.hits, cache.misses)

    results = {}

    ddg_dir, ddg_inv = core.model_ddg_scores(variants, embeddings, store)
    results["model_ddg"] = evaluate("model_ddg", ddg_dir, ddg_inv, variants, true_dir, proteins)

    logger.info("ESM-2 zero-shot masked marginals ...")
    zs_started = time.time()
    zs_dir, zs_inv = core.esm_zeroshot_scores(variants, esm2, cache_root / "zeroshot")
    zeroshot_seconds = time.time() - zs_started
    results["esm2_zeroshot"] = evaluate("esm2_zeroshot", zs_dir, zs_inv, variants, true_dir, proteins)

    rnd_dir, rnd_inv = core.random_scores(len(variants))
    results["random"] = evaluate("random", rnd_dir, rnd_inv, variants, true_dir, proteins)

    for tool in sorted({t for v in variants for t in v.tools}):
        pred_dir = np.array([v.tools.get(tool, (np.nan, np.nan))[0] for v in variants], np.float32)
        pred_inv = np.array([v.tools.get(tool, (np.nan, np.nan))[1] for v in variants], np.float32)
        if np.isfinite(pred_dir).sum() < 50:
            continue
        results[tool] = evaluate(tool, pred_dir, pred_inv, variants, true_dir, proteins)
        results[tool]["sequence_only"] = tool in SEQUENCE_ONLY_TOOLS
        results[tool]["published"] = True

    # Paired comparisons against the two baselines that decide whether the
    # trained head is worth shipping.
    comparisons = {}
    # Two baselines that decide whether the trained head earns its place, plus the
    # two strongest *sequence-only* published tools — comparing only against weak
    # baselines would flatter the result.
    contenders = {"esm2_zeroshot": zs_dir, "random": rnd_dir}
    for tool in ("DDGun", "INPS-Seq", "ACDC-NN-Seq"):
        contenders[tool] = np.array(
            [v.tools.get(tool, (np.nan, np.nan))[0] for v in variants], np.float32)
    for other, preds in contenders.items():
        ok = np.isfinite(preds) & np.isfinite(ddg_dir)
        comparisons[f"pooled__model_ddg_vs_{other}"] = core.paired_bootstrap(
            ddg_dir[ok], preds[ok], true_dir[ok], proteins[ok], core.spearman_stat)
        # The statistic the product actually depends on. Reported alongside the
        # pooled one because they disagree, and the disagreement is the finding.
        comparisons[f"within_protein__model_ddg_vs_{other}"] = core.paired_bootstrap_within(
            ddg_dir[ok], preds[ok], true_dir[ok], proteins[ok])

    report = {
        "benchmark": "S669_external",
        "git_sha": _git_sha(),
        "device": str(store.device),
        "manifest": manifest,
        "sign_convention": sign,
        "timing_seconds": {
            "embedding": round(embed_seconds, 1),
            "zeroshot": round(zeroshot_seconds, 1),
            "total": round(time.time() - started, 1),
        },
        "results": results,
        "paired_comparisons": comparisons,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "s669_benchmark.json").write_text(json.dumps(report, indent=1))
    _plots(report, variants, true_dir, true_inv, ddg_dir, ddg_inv, zs_dir, proteins, plots_dir)
    _scorecard(report, out_dir / "s669_scorecard.md")
    return report


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _plots(report, variants, true_dir, true_inv, ddg_dir, ddg_inv, zs_dir, proteins, plots_dir):
    plots_dir.mkdir(parents=True, exist_ok=True)
    results = report["results"]

    def rho(name):
        return results.get(name, {}).get("direct", {}).get("spearman_rho", np.nan)

    # --- 1. Method ranking -------------------------------------------------- #
    named = [(n, rho(n)) for n in results if np.isfinite(rho(n))]
    named.sort(key=lambda t: t[1])
    labels = [n for n, _ in named]
    values = [v for _, v in named]

    def colour(name):
        if name == "model_ddg":
            return _RED
        if name in ("esm2_zeroshot", "random"):
            return _ORANGE
        return _BLUE if results[name].get("sequence_only") else _GREY

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(labels, values, color=[colour(n) for n in labels])
    ax.set_xlabel("Spearman ρ vs experimental ΔΔG (direct)")
    ax.set_title(f"S669 external benchmark — {report['manifest']['rows_scored']} variants, "
                 f"{report['manifest']['proteins_scored']} proteins")
    ax.axvline(0, color="k", lw=0.8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (_RED, _BLUE, _GREY, _ORANGE)]
    ax.legend(handles, ["PyroForge model_ddg", "published, sequence-only",
                        "published, structure-based", "baseline"], loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "s669_method_ranking.png", dpi=150)
    plt.close(fig)

    # --- 2. Four-panel diagnostic ------------------------------------------ #
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    in_regime = np.array([v.in_regime for v in variants])
    ax.scatter(ddg_dir[in_regime], true_dir[in_regime], s=10, alpha=0.5, color=_BLUE,
               label=f"≤400 aa (n={in_regime.sum()})")
    ax.scatter(ddg_dir[~in_regime], true_dir[~in_regime], s=10, alpha=0.5, color=_ORANGE,
               label=f">400 aa (n={(~in_regime).sum()})")
    lims = [min(ddg_dir.min(), true_dir.min()) - 0.5, max(ddg_dir.max(), true_dir.max()) + 0.5]
    ax.plot(lims, lims, "k--", lw=0.8)
    _square_axes(ax, *lims, pad=0)
    ax.set_xlabel("model_ddg predicted ΔΔG"); ax.set_ylabel("experimental ΔΔG")
    ax.set_title(f"model_ddg on S669 (ρ={rho('model_ddg'):.3f})")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.scatter(ddg_dir, ddg_inv, s=10, alpha=0.5, color=_RED, label="model_ddg")
    anti = results["model_ddg"]["antisymmetry"]
    span = [min(ddg_dir.min(), ddg_inv.min()), max(ddg_dir.max(), ddg_inv.max())]
    ax.plot(span, [-x for x in span], "k--", lw=0.8, label="perfect antisymmetry")
    _square_axes(ax, *span)
    ax.set_xlabel("predicted ΔΔG (direct)"); ax.set_ylabel("predicted ΔΔG (inverse)")
    ax.set_title(f"Antisymmetry: r={anti['r_dir_inv']:.3f}, bias δ={anti['bias_mean_sum']:.3f}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    bias_items = [(n, results[n]["antisymmetry"]["r_dir_inv"]) for n in results
                  if np.isfinite(results[n]["antisymmetry"].get("r_dir_inv", np.nan))]
    bias_items.sort(key=lambda t: t[1])
    ax.barh([n for n, _ in bias_items], [v for _, v in bias_items],
            color=[_RED if n == "model_ddg" else _GREY for n, _ in bias_items])
    ax.axvline(-1, color=_GREEN, ls="--", lw=1, label="ideal (−1)")
    ax.set_xlabel("r(ΔΔG_dir, ΔΔG_inv)   — −1 is thermodynamically consistent")
    ax.set_title("Antisymmetry across methods"); ax.legend(fontsize=8)
    ax.tick_params(labelsize=7); ax.grid(axis="x", alpha=0.3)

    ax = axes[1, 1]
    dist = results["model_ddg"]["within_protein_spearman"]
    zs_dist = results["esm2_zeroshot"]["within_protein_spearman"]
    if dist.get("n_proteins"):
        ax.hist(dist["values"], bins=15, alpha=0.65, color=_RED,
                label=f"model_ddg (median {dist['median']:.2f})")
        ax.hist(zs_dist["values"], bins=15, alpha=0.55, color=_ORANGE,
                label=f"ESM-2 zero-shot (median {zs_dist['median']:.2f})")
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("within-protein Spearman ρ"); ax.set_ylabel("proteins")
        ax.set_title(f"Within-protein ranking (n={dist['n_proteins']} proteins)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle("PyroForge ΔΔG — S669 external benchmark", fontsize=13)
    fig.tight_layout()
    fig.savefig(plots_dir / "s669_diagnostics.png", dpi=150)
    plt.close(fig)

    # --- 2b. Pooled vs within-protein — the headline dissociation ---------- #
    pairs = [(n, rho(n), results[n]["within_protein_spearman"].get("median", np.nan))
             for n in results]
    pairs = [p for p in pairs if np.isfinite(p[1]) and np.isfinite(p[2])]
    pairs.sort(key=lambda t: t[2])
    fig, ax = plt.subplots(figsize=(10, 8))
    y = np.arange(len(pairs))
    ax.barh(y - 0.2, [p[1] for p in pairs], height=0.4, color=_GREY, label="pooled ρ")
    ax.barh(y + 0.2, [p[2] for p in pairs], height=0.4, color=_BLUE, label="within-protein ρ")
    for i, p in enumerate(pairs):
        if p[0] == "model_ddg":
            ax.barh(i - 0.2, p[1], height=0.4, color=_ORANGE)
            ax.barh(i + 0.2, p[2], height=0.4, color=_RED)
    ax.set_yticks(y); ax.set_yticklabels([p[0] for p in pairs], fontsize=8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Spearman ρ")
    within_rank = sum(1 for p in pairs if p[2] > dict((q[0], q[2]) for q in pairs)["model_ddg"]) + 1
    pooled_rank = sum(1 for p in pairs if p[1] > dict((q[0], q[1]) for q in pairs)["model_ddg"]) + 1
    ax.set_title("Pooled vs within-protein ranking on S669\n"
                 f"model_ddg (red/orange): #{pooled_rank} of {len(pairs)} pooled, "
                 f"#{within_rank} within-protein")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "s669_pooled_vs_within.png", dpi=150)
    plt.close(fig)

    # --- 3. Stabilizing-variant detection ---------------------------------- #
    auc = [(n, results[n]["stabilizing_detection"].get("roc_auc", np.nan)) for n in results]
    auc = [(n, v) for n, v in auc if np.isfinite(v)]
    auc.sort(key=lambda t: t[1])
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh([n for n, _ in auc], [v for _, v in auc],
            color=[colour(n) for n, _ in auc])
    ax.axvline(0.5, color="k", ls="--", lw=1, label="chance")
    base = results["model_ddg"]["stabilizing_detection"]["base_rate"]
    ax.set_xlabel("ROC-AUC for detecting stabilizing variants (ΔΔG > 0)")
    ax.set_title(f"Finding stabilizing variants — the product's actual job\n"
                 f"base rate {base:.1%} of {report['manifest']['rows_scored']} variants")
    ax.legend(fontsize=8); ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "s669_stabilizing_auc.png", dpi=150)
    plt.close(fig)


def _scorecard(report, path: Path) -> None:
    results = report["results"]
    manifest = report["manifest"]
    rows = sorted(results.items(),
                  key=lambda kv: -(kv[1]["direct"].get("spearman_rho") or -9))
    lines = [
        "# S669 external benchmark — PyroForge ΔΔG",
        "",
        f"Dataset: **S669** ({manifest['reference']}) — {manifest['rows_scored']} variants "
        f"on {manifest['proteins_scored']} proteins.",
        f"Regime split: {manifest['in_regime_rows']} at ≤400 aa, "
        f"{manifest['out_of_regime_rows']} above.",
        f"Code {report['git_sha']} on {report['device']}.",
        "",
        "Two correlations are reported and they disagree. **Pooled ρ** mixes",
        "between-protein level differences with within-protein ranking; **within-protein",
        "ρ** is the median over proteins of the ranking inside each one. The optimizer",
        "only ever ranks mutations within a single protein, so within-protein ρ is the",
        "product-relevant number — and `model_ddg`'s difference-embedding feature",
        "deliberately cancels the per-protein offset that pooled ρ rewards.",
        "",
        "| Method | class | pooled ρ | **within-protein ρ** | top-5 prec | antisym r | stab. AUC |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, res in rows:
        if res["direct"].get("spearman_rho") is None:
            continue
        cls = ("**ours**" if name == "model_ddg"
               else "baseline" if name in ("esm2_zeroshot", "random")
               else "seq-only" if res.get("sequence_only") else "structure")
        within = res["within_protein_spearman"].get("median", float("nan"))
        lines.append(
            f"| {name} | {cls} | {res['direct'].get('spearman_rho', float('nan')):.3f} "
            f"| **{within:.3f}** "
            f"| {res['top5_precision'].get('precision', float('nan')):.3f} "
            f"| {res['antisymmetry'].get('r_dir_inv', float('nan')):.3f} "
            f"| {res['stabilizing_detection'].get('roc_auc', float('nan')):.3f} |")

    ddg = results["model_ddg"]
    lines += [
        "",
        "## Regime split (model_ddg)",
        "",
        f"- ≤400 aa (in regime): ρ = {ddg['direct_in_regime']['spearman_rho']:.3f} "
        f"(n={ddg['direct_in_regime']['n']})",
        f"- >400 aa (out of regime): ρ = {ddg['direct_out_of_regime']['spearman_rho']:.3f} "
        f"(n={ddg['direct_out_of_regime']['n']})",
        "",
        "## Paired comparisons (bootstrap over proteins)",
        "",
    ]
    for key, cmp in report["paired_comparisons"].items():
        ci = cmp.get("ci95")
        lines.append(f"- `{key}`: Δρ = {cmp['delta']:+.3f}"
                     + (f", 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "")
                     + (f", sign-flip rate {cmp['p_sign_flip']:.3f}" if "p_sign_flip" in cmp else ""))
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/bench"))
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--plots-dir", type=Path, default=Path("checkpoints/plots"))
    parser.add_argument("--limit", type=int, default=None, help="smoke-test on N variants")
    args = parser.parse_args()
    report = run(args.data_dir, args.out_dir, args.plots_dir, args.limit)
    print(json.dumps(report["results"]["model_ddg"], indent=1))


if __name__ == "__main__":
    main()
