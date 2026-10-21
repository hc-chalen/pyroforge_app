"""Ssym — the antisymmetry benchmark, and a depth fix for S669's thin coverage.

Ssym (Pucci et al., *Bioinformatics* 34:3659–3665, 2018) is 342 mutations plus
their 342 reverses across 15 protein chains, every one with an experimental
structure for *both* the wild type and the mutant. Two reasons it is worth
running on top of S669:

1. **Depth per protein — but not breadth.** S669's within-protein statistic rests
   on the 27 of its 94 proteins carrying ≥5 mutations. Ssym is the opposite
   shape: deep but narrow, and *heavily* skewed — T4 lysozyme (1L63) alone
   carries 118 of the 342 mutations, and three proteins hold 72% of the set.
   Only 9 of 15 proteins reach ≥5 mutations, so Ssym yields **fewer** usable
   proteins than S669, not more.

   What it does buy is reliability of the individual estimates: a within-protein
   ρ computed over 118 mutations is worth far more than one over 5. It therefore
   complements S669 rather than replacing it, and **it does not fix the
   small-protein-count weakness** — only a broad, deep set (ProteinGym's
   stability split) would.

2. **It is the canonical antisymmetry set**, and the balanced construction means
   the direct and reverse halves have exactly opposite ΔΔG by design.

Sequences. Only the 15 *wild-type* structures are fetched. For a sequence-only
model the reverse variant needs no separate structure: it is simply the forward
mutant sequence with the roles swapped, so it is constructed rather than
downloaded. Mutation positions use PDB numbering, resolved to sequence index by
the same per-protein offset-consensus search as the S669 loader.

Run::

    uv run python -m scripts.bench.ssym --fetch     # download 15 PDBs
    uv run python -m scripts.bench.ssym
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.bench import core
from scripts.bench.data import (
    IN_REGIME_MAX_LENGTH,
    MAX_EMBED_LENGTH,
    Variant,
    _atom_sequence,
    _resolve_offset,
)

logger = logging.getLogger("bench.ssym")

_BLUE, _ORANGE, _RED, _GREY = "#2f6f9f", "#c26a12", "#c0392f", "#8fa0ad"


def _square_axes(ax, lo: float, hi: float, pad: float = 0.04) -> None:
    """Give x and y the same data scale so one unit has the same length on both."""
    span = float(hi) - float(lo)
    margin = pad * span if span > 0 else 1.0
    ax.set_xlim(lo - margin, hi + margin)
    ax.set_ylim(lo - margin, hi + margin)
    ax.set_aspect("equal", adjustable="box")


SSYM_CSV_URL = "https://protddg-bench.github.io/data/ssym.csv"
RCSB_URL = "https://files.rcsb.org/download/{pdb}.pdb"
_MUT_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def fetch(data_dir: Path) -> None:
    root = data_dir / "Ssym"
    (root / "pdbs").mkdir(parents=True, exist_ok=True)
    csv_path = root / "ssym.csv"
    if not csv_path.exists():
        subprocess.run(["curl", "-sL", "-o", str(csv_path), SSYM_CSV_URL], check=True)
    rows = list(csv.DictReader(csv_path.open()))
    for pdb in sorted({r["PDB"] for r in rows if r["DIR/INV"] == "DIR"}):
        dest = root / "pdbs" / f"{pdb}.pdb"
        if not dest.exists():
            subprocess.run(["curl", "-sL", "-o", str(dest),
                            RCSB_URL.format(pdb=pdb)], check=True)
    logger.info("Ssym ready at %s", root)


def load_ssym(data_dir: Path) -> tuple[list[Variant], dict]:
    root = data_dir / "Ssym"
    rows = [r for r in csv.DictReader((root / "ssym.csv").open()) if r["DIR/INV"] == "DIR"]

    by_protein: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_protein[(row["PDB"], row["CHAIN"])].append(row)

    variants: list[Variant] = []
    dropped = 0
    for (pdb, chain), prot_rows in by_protein.items():
        seq = _atom_sequence(root / "pdbs" / f"{pdb}.pdb", chain)
        parsed = []
        for row in prot_rows:
            match = _MUT_RE.match(row["MUT"])
            if match:
                parsed.append((row, match.group(1), int(match.group(2)), match.group(3)))
        offset = _resolve_offset(seq, [(wt, pos) for _, wt, pos, _ in parsed])
        if offset is None:
            # Fall back to a per-row offset only if the whole-protein search fails,
            # and count what could not be placed rather than coercing it.
            dropped += len(prot_rows)
            logger.warning("no consistent offset for %s%s — dropping %d rows",
                           pdb, chain, len(prot_rows))
            continue
        for row, wt_aa, pos, mt_aa in parsed:
            pos0 = pos - 1 + offset
            mut_seq = seq[:pos0] + mt_aa + seq[pos0 + 1:]
            ddg = float(row["DDG"])
            variants.append(Variant(
                protein=f"{pdb}{chain}", wt_seq=seq, mut_seq=mut_seq, pos0=pos0,
                wt_aa=wt_aa, mt_aa=mt_aa,
                # Ssym's balanced construction defines the reverse as the exact
                # negative, which is what makes it an antisymmetry benchmark.
                ddg_dir=ddg, ddg_inv=-ddg, tools={},
            ))

    embeddable = [v for v in variants if v.length <= MAX_EMBED_LENGTH]
    manifest = {
        "dataset": "Ssym",
        "reference": "Pucci et al., Bioinformatics 34(21):3659-3665 (2018)",
        "direct_rows_in_csv": len(rows),
        "rows_resolved": len(variants),
        "rows_dropped_no_offset": dropped,
        "rows_scored": len(embeddable),
        "proteins_scored": len({v.protein for v in embeddable}),
        "mutations_per_protein": dict(sorted(
            ((p, sum(1 for v in embeddable if v.protein == p))
             for p in {v.protein for v in embeddable}), key=lambda t: -t[1])),
        "proteins_with_5plus_mutations": sum(
            1 for p in {v.protein for v in embeddable}
            if sum(1 for v in embeddable if v.protein == p) >= 5),
        "in_regime_rows": sum(1 for v in embeddable if v.in_regime),
        "in_regime_max_length": IN_REGIME_MAX_LENGTH,
    }
    return embeddable, manifest


def run(data_dir: Path, out_dir: Path, plots_dir: Path) -> dict:
    from app.services.esm2 import get_esm2_service
    from app.services.model_store import get_model_store

    variants, manifest = load_ssym(data_dir)
    proteins = np.array([v.protein for v in variants])
    true_dir = np.array([v.ddg_dir for v in variants], dtype=np.float32)

    store = get_model_store()
    store.load()
    esm2 = get_esm2_service()

    cache_root = Path("cache/bench")
    cache = core.EmbedCache(cache_root / "embeddings")
    embeddings = cache.get_many(
        [v.wt_seq for v in variants] + [v.mut_seq for v in variants], esm2)

    ddg_dir, ddg_inv = core.model_ddg_scores(variants, embeddings, store)
    zs_dir, zs_inv = core.esm_zeroshot_scores(variants, esm2, cache_root / "zeroshot")
    rnd_dir, _ = core.random_scores(len(variants))

    results = {}
    for name, pred_dir, pred_inv in (("model_ddg", ddg_dir, ddg_inv),
                                     ("esm2_zeroshot", zs_dir, zs_inv),
                                     ("random", rnd_dir, -rnd_dir)):
        results[name] = {
            "direct": core.basic_metrics(pred_dir, true_dir),
            "antisymmetry": core.antisymmetry(pred_dir, pred_inv),
            "within_protein_spearman": core.within_protein_spearman(
                pred_dir, true_dir, proteins),
            "stabilizing_detection": core.stabilizing_detection(pred_dir, true_dir),
            "top5_precision": core.top_k_precision(pred_dir, true_dir, proteins, k=5),
        }

    comparisons = {
        "within_protein__model_ddg_vs_esm2_zeroshot": core.paired_bootstrap_within(
            ddg_dir, zs_dir, true_dir, proteins),
        "within_protein__model_ddg_vs_random": core.paired_bootstrap_within(
            ddg_dir, rnd_dir, true_dir, proteins),
    }

    report = {"benchmark": "Ssym_external", "manifest": manifest,
              "results": results, "paired_comparisons": comparisons}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ssym_benchmark.json").write_text(json.dumps(report, indent=1))
    _plot(report, ddg_dir, ddg_inv, true_dir, proteins, plots_dir)
    return report


def _plot(report, ddg_dir, ddg_inv, true_dir, proteins, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.scatter(ddg_dir, ddg_inv, s=14, alpha=0.55, color=_RED)
    anti = report["results"]["model_ddg"]["antisymmetry"]
    span = [float(min(ddg_dir.min(), ddg_inv.min())), float(max(ddg_dir.max(), ddg_inv.max()))]
    ax.plot(span, [-s for s in span], "k--", lw=0.9, label="perfect antisymmetry")
    _square_axes(ax, *span)
    ax.set_xlabel("predicted ΔΔG (direct)"); ax.set_ylabel("predicted ΔΔG (reverse)")
    ax.set_title(f"Antisymmetry on Ssym\nr={anti['r_dir_inv']:.3f}, bias δ={anti['bias_mean_sum']:+.3f}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    dist = report["results"]["model_ddg"]["within_protein_spearman"]
    zs = report["results"]["esm2_zeroshot"]["within_protein_spearman"]
    order = np.argsort(dist["values"])
    ys = np.arange(len(order))
    ax.barh(ys, np.array(dist["values"])[order], color=_RED, alpha=0.85, label="model_ddg")
    ax.axvline(dist["median"], color=_RED, ls="--", lw=1,
               label=f"median {dist['median']:.2f}")
    ax.axvline(zs["median"], color=_ORANGE, ls=":", lw=1.5,
               label=f"ESM-2 zero-shot median {zs['median']:.2f}")
    ax.set_yticks([]); ax.set_xlabel("within-protein Spearman ρ")
    ax.set_title(f"Per-protein ranking ({dist['n_proteins']} proteins with ≥5 muts)\n"
                 "deep but narrow: 3 proteins hold 72% of Ssym")
    ax.legend(fontsize=8); ax.grid(axis="x", alpha=0.3)

    ax = axes[2]
    ax.scatter(ddg_dir, true_dir, s=14, alpha=0.55, color=_BLUE)
    m = report["results"]["model_ddg"]["direct"]
    lo = float(min(ddg_dir.min(), true_dir.min()))
    hi = float(max(ddg_dir.max(), true_dir.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.9)
    _square_axes(ax, lo, hi)
    ax.set_xlabel("predicted ΔΔG"); ax.set_ylabel("experimental ΔΔG")
    ax.set_title(f"model_ddg on Ssym\nρ={m['spearman_rho']:.3f}  r={m['pearson_r']:.3f}  n={m['n']}")
    ax.grid(alpha=0.3)

    fig.suptitle("PyroForge ΔΔG — Ssym benchmark", fontsize=13)
    fig.tight_layout()
    fig.savefig(plots_dir / "ssym_benchmark.png", dpi=150)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/bench"))
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--plots-dir", type=Path, default=Path("checkpoints/plots"))
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    if args.fetch:
        fetch(args.data_dir)
    report = run(args.data_dir, args.out_dir, args.plots_dir)
    print(json.dumps(report["manifest"], indent=1))
    for name, res in report["results"].items():
        print(f"{name:16s} rho={res['direct'].get('spearman_rho', float('nan')):+.3f} "
              f"within={res['within_protein_spearman'].get('median', float('nan')):+.3f} "
              f"antisym={res['antisymmetry'].get('r_dir_inv', float('nan')):+.3f}")
    for key, cmp in report["paired_comparisons"].items():
        ci = cmp.get("ci95")
        print(f"{key}: Δ={cmp['delta']:+.3f}" + (f" CI[{ci[0]:+.3f},{ci[1]:+.3f}]" if ci else ""))


if __name__ == "__main__":
    main()
