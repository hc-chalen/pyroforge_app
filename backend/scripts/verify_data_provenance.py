#!/usr/bin/env python3
"""Establish what the ``esm2_*.pt`` label columns actually are, from the inside.

The export script that produced these files is not in this repo, and the files
carry no sequences — only embeddings and opaque integer ids. So the columns
cannot be matched row-by-row against the published source datasets. What *can*
be done is check the columns against each other: thermodynamic quantities from a
single experiment obey known relations, and a column that satisfies them is
almost certainly what its name says it is.

Four checks:

  1. ddG vs dG        — for a mutant and its wild-type anchor (same
                        ``source_sequence_ids``, 0 substitutions),
                        ddG should equal ±(dG_mut − dG_wt). Confirms the two
                        columns share an origin, and pins the sign convention.
  2. dG vs proteases  — Tsuboyama 2023 derives dG from trypsin and chymotrypsin
                        proteolysis. If dG is a function of those two columns,
                        dG is Tsuboyama-derived and the two ``*_ml`` columns are
                        explained.
  3. antisymmetry     — where both A→B and B→A exist, ddG should flip sign.
  4. ranges           — each column against the physical range its name implies.

Run:  uv run python scripts/verify_data_provenance.py
      uv run python scripts/verify_data_provenance.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]

# Physically sensible ranges, by what the column name claims to be.
EXPECTED_RANGES = {
    "ddg": (-25.0, 25.0, "kcal/mol"),
    "dg": (-15.0, 65.0, "kcal/mol"),
    "tm": (0.0, 140.0, "degC"),
    "dtm": (-60.0, 60.0, "degC"),
    "cm": (0.0, 15.0, "M"),
    "m": (-15.0, 15.0, "kcal/mol/M"),
    "dh": (-50.0, 2000.0, "kcal/mol"),
    "dhvh": (-150.0, 400.0, "kcal/mol"),
    "dcp": (-2.0, 70.0, "kcal/mol/K"),
    "domainome_ddg": (-1.0, 3.0, "a.u."),
    "domainome_fitness": (-2.0, 2.0, "a.u."),
    "chymotrypsin_ml": (-4.0, 3.0, "a.u."),
    "trypsin_ml": (-3.0, 3.0, "a.u."),
}


def _load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _np(data: dict, key: str) -> np.ndarray:
    return data[key].numpy().astype(np.float64)


def check_ddg_vs_dg(data: dict) -> dict[str, Any]:
    """ddG should be the dG difference between a mutant and its WT anchor."""
    sids = data["source_sequence_ids"].numpy()
    nsub = data["num_substitutions"].numpy()
    dg = _np(data, "dg")
    ddg = _np(data, "ddg")

    # WT anchor per source sequence: the 0-substitution row.
    wt_row: dict[int, int] = {}
    for i in np.flatnonzero(nsub == 0):
        wt_row.setdefault(int(sids[i]), int(i))

    rows, deltas, ddgs = [], [], []
    for i in np.flatnonzero(nsub > 0):
        w = wt_row.get(int(sids[i]))
        if w is None:
            continue
        if not (np.isfinite(dg[i]) and np.isfinite(dg[w]) and np.isfinite(ddg[i])):
            continue
        rows.append(int(i))
        deltas.append(dg[i] - dg[w])
        ddgs.append(ddg[i])

    out: dict[str, Any] = {"n_comparable": len(rows), "n_wt_anchors": len(wt_row)}
    if len(rows) < 50:
        out["verdict"] = "INSUFFICIENT"
        return out

    d = np.asarray(deltas)
    g = np.asarray(ddgs)
    r_pos = float(np.corrcoef(d, g)[0, 1])
    mae_pos = float(np.mean(np.abs(d - g)))
    mae_neg = float(np.mean(np.abs(-d - g)))
    # Which sign convention fits: ddG = dG_mut - dG_wt, or the negation?
    if mae_pos <= mae_neg:
        convention, mae, resid = "ddg = dg_mut - dg_wt", mae_pos, d - g
    else:
        convention, mae, resid = "ddg = dg_wt - dg_mut", mae_neg, -d - g

    out.update(
        {
            "pearson_r": r_pos,
            "convention": convention,
            "mae": mae,
            "median_abs_resid": float(np.median(np.abs(resid))),
            "frac_within_0.1": float(np.mean(np.abs(resid) < 0.1)),
            "frac_within_0.5": float(np.mean(np.abs(resid) < 0.5)),
        }
    )
    if abs(r_pos) > 0.95 and mae < 0.5:
        out["verdict"] = "CONSISTENT"
    elif abs(r_pos) > 0.7:
        out["verdict"] = "PARTIAL"
    else:
        out["verdict"] = "INCONSISTENT"
    return out


def check_dg_vs_proteases(data: dict) -> dict[str, Any]:
    """Tsuboyama derives dG from trypsin + chymotrypsin proteolysis."""
    dg = _np(data, "dg")
    out: dict[str, Any] = {}
    for name in ("trypsin_ml", "chymotrypsin_ml"):
        if name not in data:
            continue
        v = _np(data, name)
        m = np.isfinite(dg) & np.isfinite(v)
        out[name] = {
            "n_overlap": int(m.sum()),
            "pearson_r": float(np.corrcoef(dg[m], v[m])[0, 1]) if m.sum() > 50 else None,
        }
    # Is dG close to a simple combination of the two?
    if "trypsin_ml" in data and "chymotrypsin_ml" in data:
        t, c = _np(data, "trypsin_ml"), _np(data, "chymotrypsin_ml")
        m = np.isfinite(dg) & np.isfinite(t) & np.isfinite(c)
        if m.sum() > 50:
            A = np.column_stack([t[m], c[m], np.ones(int(m.sum()))])
            coef, *_ = np.linalg.lstsq(A, dg[m], rcond=None)
            pred = A @ coef
            ss_res = float(np.sum((dg[m] - pred) ** 2))
            ss_tot = float(np.sum((dg[m] - dg[m].mean()) ** 2))
            out["linear_fit"] = {
                "n": int(m.sum()),
                "coef_trypsin": float(coef[0]),
                "coef_chymotrypsin": float(coef[1]),
                "intercept": float(coef[2]),
                "r_squared": 1.0 - ss_res / ss_tot if ss_tot else None,
            }
    rs = [v["pearson_r"] for v in out.values() if isinstance(v, dict) and v.get("pearson_r")]
    out["verdict"] = "RELATED" if rs and max(abs(r) for r in rs) > 0.5 else "UNRELATED"
    return out


def check_antisymmetry(data: dict) -> dict[str, Any]:
    """Where both A->B and B->A exist at the same position, ddG should flip."""
    nsub = data["num_substitutions"].numpy()
    pos = data["mutation_positions"].numpy()
    src = data["mutation_source_aa"].numpy()
    tgt = data["mutation_target_aa"].numpy()
    pid = data["parent_protein_ids"].numpy() if "parent_protein_ids" in data else None
    ddg = _np(data, "ddg")

    single = np.flatnonzero((nsub == 1) & np.isfinite(ddg))
    fwd: dict[tuple, float] = {}
    for i in single:
        key = (int(pid[i]) if pid is not None else 0, int(pos[i, 0]), int(src[i, 0]), int(tgt[i, 0]))
        fwd.setdefault(key, float(ddg[i]))

    pairs = []
    for (p, position, a, b), v in fwd.items():
        rev = fwd.get((p, position, b, a))
        if rev is not None and (p, position, b, a) > (p, position, a, b):
            pairs.append((v, rev))

    out: dict[str, Any] = {"n_reverse_pairs": len(pairs)}
    if len(pairs) < 20:
        out["verdict"] = "INSUFFICIENT"
        return out
    f = np.array([p[0] for p in pairs])
    r = np.array([p[1] for p in pairs])
    out.update(
        {
            "pearson_r": float(np.corrcoef(f, r)[0, 1]),
            "mean_sum": float(np.mean(f + r)),
            "mae_of_sum": float(np.mean(np.abs(f + r))),
            "frac_opposite_sign": float(np.mean(np.sign(f) != np.sign(r))),
        }
    )
    out["verdict"] = "ANTISYMMETRIC" if out["pearson_r"] < -0.5 else "NOT_ANTISYMMETRIC"
    return out


def check_ranges(data: dict) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, (lo, hi, unit) in EXPECTED_RANGES.items():
        if key not in data:
            continue
        v = _np(data, key)
        m = np.isfinite(v)
        if not m.any():
            out[key] = {"verdict": "EMPTY"}
            continue
        vals = v[m]
        n_out = int(((vals < lo) | (vals > hi)).sum())
        out[key] = {
            "unit_assumed": unit,
            "n_finite": int(m.sum()),
            "observed": [float(vals.min()), float(vals.max())],
            "expected": [lo, hi],
            "n_outside": n_out,
            "verdict": "IN_RANGE" if n_out == 0 else "OUT_OF_RANGE",
        }
    return out


def verify(path: Path) -> dict[str, Any]:
    data = _load(path)
    report: dict[str, Any] = {
        "file": str(path),
        "row_count": int(data["embeddings"].shape[0]),
        "model_name": data.get("model_name"),
        "pooling": data.get("pooling"),
        "scope": data.get("scope"),
        "has_parent_protein_ids": "parent_protein_ids" in data,
    }
    report["ddg_vs_dg"] = check_ddg_vs_dg(data)
    report["dg_vs_proteases"] = check_dg_vs_proteases(data)
    report["antisymmetry"] = check_antisymmetry(data)
    report["ranges"] = check_ranges(data)
    del data
    return report


def print_report(r: dict[str, Any]) -> None:
    print("=" * 74)
    print(r["file"])
    print(f"  rows={r['row_count']:,}  model={r['model_name']}  pooling={r['pooling']}")
    print()

    c = r["ddg_vs_dg"]
    print(f"  [1] ddG vs dG  ->  {c['verdict']}")
    print(f"      WT anchors={c['n_wt_anchors']:,}  comparable rows={c['n_comparable']:,}")
    if "pearson_r" in c:
        print(f"      convention: {c['convention']}")
        print(f"      r={c['pearson_r']:+.4f}  MAE={c['mae']:.4f}  median|resid|={c['median_abs_resid']:.4f}")
        print(f"      within 0.1: {c['frac_within_0.1']:.1%}   within 0.5: {c['frac_within_0.5']:.1%}")
    print()

    p = r["dg_vs_proteases"]
    print(f"  [2] dG vs proteases  ->  {p['verdict']}")
    for k in ("trypsin_ml", "chymotrypsin_ml"):
        if k in p and p[k].get("pearson_r") is not None:
            print(f"      dG ~ {k:18s} r={p[k]['pearson_r']:+.4f}  (n={p[k]['n_overlap']:,})")
    if "linear_fit" in p:
        lf = p["linear_fit"]
        print(f"      dG ~ {lf['coef_trypsin']:+.3f}*tryp {lf['coef_chymotrypsin']:+.3f}*chym "
              f"{lf['intercept']:+.3f}   R2={lf['r_squared']:.4f}  (n={lf['n']:,})")
    print()

    a = r["antisymmetry"]
    print(f"  [3] ddG antisymmetry  ->  {a['verdict']}")
    print(f"      reverse pairs found={a['n_reverse_pairs']:,}")
    if "pearson_r" in a:
        print(f"      r(fwd,rev)={a['pearson_r']:+.4f}  mean(fwd+rev)={a['mean_sum']:+.4f}  "
              f"MAE|sum|={a['mae_of_sum']:.4f}  opposite sign={a['frac_opposite_sign']:.1%}")
    print()

    print("  [4] physical ranges")
    for key, v in r["ranges"].items():
        if v["verdict"] == "EMPTY":
            print(f"      {key:22s} EMPTY")
            continue
        flag = "ok " if v["verdict"] == "IN_RANGE" else "!! "
        print(f"      {flag}{key:22s} n={v['n_finite']:>7,}  "
              f"obs=[{v['observed'][0]:>8.3g},{v['observed'][1]:>8.3g}]  "
              f"exp=[{v['expected'][0]:g},{v['expected'][1]:g}] {v['unit_assumed']}"
              + (f"  outside={v['n_outside']:,}" if v["n_outside"] else ""))
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify esm2_*.pt label provenance internally")
    ap.add_argument("paths", nargs="*", help="default: esm2_test.pt then esm2_train.pt")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    paths = [Path(p) for p in args.paths] or [
        REPO_ROOT / "esm2_test.pt",
        REPO_ROOT / "esm2_train.pt",
    ]
    paths = [p for p in paths if p.exists()]
    if not paths:
        print("no input files found", file=sys.stderr)
        sys.exit(1)

    reports = [verify(p) for p in paths]
    if args.json:
        print(json.dumps(reports, indent=2, default=str))
    else:
        for r in reports:
            print_report(r)


if __name__ == "__main__":
    main()
