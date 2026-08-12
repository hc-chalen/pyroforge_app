#!/usr/bin/env python3
"""Write the archive-ready copies of the ``esm2_*.pt`` embedding datasets.

The working files carry 15 label columns inherited from an upstream export whose
script is not in this repo. ``scripts/verify_data_provenance.py`` established
what five of them are; the other ten remain unattributed and are unused by every
trainer and every served model. A public, permanent archive should only contain
columns whose origin can be stated, so this script copies the datasets keeping
the documented columns and dropping the rest.

The working files are never modified — they stay as the only copy of the
unattributed columns, which cannot be regenerated without the missing export
script.

Kept, with the reason each survives:

  embeddings              the features; ESM2-650M layer 33, mean-pooled
  ddg                     trains model_ddg.pt; verified = dg_mut - dg_wt
  dg                      trains model_thermo.pt; verified R2~0.91 on proteases
  tm                      read by train_tm_meltome.py as a transfer diagnostic
  trypsin_ml              raw proteolysis measurement behind dg (Tsuboyama)
  chymotrypsin_ml         raw proteolysis measurement behind dg (Tsuboyama)
  *_ids, num_*, mutation_*  row identity and mutation description

Dropped — zero code references, provenance not established:

  dtm  cm  m  dh  dhvh  dcp
  domainome_ddg  domainome_ddg_std  domainome_fitness  domainome_fitness_std

Dropping them saves ~22 MB of ~2.8 GB: the point is a record that can be
described accurately, not disk space.

Run:  uv run python scripts/export_public_dataset.py
      uv run python scripts/export_public_dataset.py --out-dir ../release
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]

STRUCTURE_KEYS = [
    "embeddings",
    "mutant_ids",
    "parent_protein_ids",
    "source_sequence_ids",
    "target_sequence_ids",
    "num_substitutions",
    "num_insertions",
    "num_deletions",
    "mutation_positions",
    "mutation_source_aa",
    "mutation_target_aa",
]
LABEL_KEYS = ["ddg", "dg", "tm", "trypsin_ml", "chymotrypsin_ml"]
META_KEYS = [
    "aa_vocab",
    "model_name",
    "pooling",
    "hidden_size",
    "row_count",
    "export_limit",
    "export_offset",
    "available_in_scope",
    "scope",
]
DROPPED = [
    "dtm",
    "cm",
    "m",
    "dh",
    "dhvh",
    "dcp",
    "domainome_ddg",
    "domainome_ddg_std",
    "domainome_fitness",
    "domainome_fitness_std",
]

PROVENANCE = {
    "embeddings": "ESM-2 650M (facebook/esm2_t33_650M_UR50D), layer 33, mean-pooled over residues",
    "ddg": "Tsuboyama et al. 2023 (doi:10.1038/s41586-023-06328-6); verified internally as dg_mut - dg_wt",
    "dg": "Tsuboyama et al. 2023 (doi:10.1038/s41586-023-06328-6); verified against the proteolysis columns",
    "tm": "upstream export of FireProtDB lineage; assay differs from Meltome, used only as a transfer diagnostic",
    "trypsin_ml": "Tsuboyama et al. 2023 trypsin proteolysis measurement",
    "chymotrypsin_ml": "Tsuboyama et al. 2023 chymotrypsin proteolysis measurement",
}


def sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def export_one(src: Path, dest: Path) -> dict[str, Any]:
    data = torch.load(src, map_location="cpu", weights_only=False)

    out: dict[str, Any] = {}
    kept_structure, kept_labels, missing = [], [], []

    for k in STRUCTURE_KEYS:
        if k in data:
            out[k] = data[k]
            kept_structure.append(k)
        else:
            missing.append(k)

    for k in LABEL_KEYS:
        if k in data:
            out[k] = data[k]
            kept_labels.append(k)
        else:
            missing.append(k)

    for k in META_KEYS:
        if k in data:
            out[k] = data[k]

    dropped_present = [k for k in DROPPED if k in data]
    unexpected = [
        k for k in data
        if k not in STRUCTURE_KEYS + LABEL_KEYS + META_KEYS + DROPPED
    ]

    # Provenance travels inside the file, so a downloaded copy is self-describing.
    out["column_provenance"] = {k: PROVENANCE[k] for k in kept_structure + kept_labels if k in PROVENANCE}
    out["dropped_columns"] = dropped_present
    out["dropped_reason"] = (
        "Unused by all PyroForge trainers and served models, and provenance not "
        "established (the upstream export script is not part of the project "
        "repository). Retained only in the authors' working copy."
    )
    out["source_file"] = src.name
    out["verification"] = (
        "See scripts/verify_data_provenance.py: ddg reproduces dg_mut - dg_wt "
        "(train r=0.974, test r=1.0000) and dg is a linear function of the two "
        "proteolysis columns (train R2=0.906, test R2=0.952)."
    )

    del data
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dest)
    n_rows = int(out["embeddings"].shape[0])
    del out

    return {
        "source": str(src),
        "dest": str(dest),
        "rows": n_rows,
        "src_bytes": src.stat().st_size,
        "dest_bytes": dest.stat().st_size,
        "kept_structure": kept_structure,
        "kept_labels": kept_labels,
        "dropped": dropped_present,
        "missing_expected": missing,
        "unexpected_left_behind": unexpected,
        "sha256": sha256(dest),
    }


# The second Zenodo record: trained weights plus the reports that state what
# each one scored. Copied, not regenerated — these are the served artifacts.
MODEL_RECORD = [
    (Path("checkpoints/model_ddg.pt"), "models/model_ddg.pt"),
    (Path("checkpoints/model_thermo.pt"), "models/model_thermo.pt"),
    (Path("checkpoints/improved/model_tm.joblib"), "models/model_tm.joblib"),
    (Path("checkpoints/improved/tm_report.json"), "models/tm_report.json"),
    (Path("checkpoints/plots/all_models_metrics.json"), "models/all_models_metrics.json"),
    (Path("checkpoints/plots/all_models_scorecard.md"), "models/all_models_scorecard.md"),
    (Path("checkpoints/autopytorch/autopytorch_report.json"), "models/autopytorch_report.json"),
    (Path("checkpoints/improved/meltome_tm.npz"), "meltome_tm.npz"),
]


def stage_models(backend_root: Path, out_dir: Path) -> list[dict[str, Any]]:
    """Copy the model record into the staging dir and checksum it."""
    import shutil

    staged = []
    for rel, dest_rel in MODEL_RECORD:
        src = backend_root / rel
        if not src.exists():
            staged.append({"dest": dest_rel, "missing": True, "source": str(src)})
            continue
        dest = out_dir / dest_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        staged.append(
            {
                "dest": dest_rel,
                "source": str(src),
                "bytes": dest.stat().st_size,
                "sha256": sha256(dest),
            }
        )
    return staged


def main() -> None:
    ap = argparse.ArgumentParser(description="Export archive-ready esm2 datasets")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "release")
    ap.add_argument(
        "--inputs",
        nargs="*",
        type=Path,
        default=[REPO_ROOT / "esm2_train.pt", REPO_ROOT / "esm2_test.pt"],
    )
    ap.add_argument(
        "--no-models",
        action="store_true",
        help="skip staging the trained-model record",
    )
    args = ap.parse_args()

    srcs = [p for p in args.inputs if p.exists()]
    if not srcs:
        print("no input files found", file=sys.stderr)
        sys.exit(1)

    results = []
    for src in srcs:
        dest = args.out_dir / f"{src.stem}_public.pt"
        print(f"-> {src.name} ... ", end="", flush=True)
        r = export_one(src, dest)
        print(f"{r['rows']:,} rows, {r['dest_bytes'] / 1e9:.2f} GB")
        results.append(r)

    models: list[dict[str, Any]] = []
    if not args.no_models:
        print("-> staging model record ... ", end="", flush=True)
        models = stage_models(Path(__file__).resolve().parents[1], args.out_dir)
        n_missing = sum(1 for m in models if m.get("missing"))
        print(f"{len(models) - n_missing}/{len(models)} files")

    manifest = args.out_dir / "MANIFEST.json"
    with open(manifest, "w") as fh:
        json.dump({"data_record": results, "model_record": models}, fh, indent=2)

    print()
    for r in results:
        print(f"{Path(r['dest']).name}")
        print(f"  rows        {r['rows']:,}")
        print(f"  size        {r['src_bytes'] / 1e9:.3f} GB -> {r['dest_bytes'] / 1e9:.3f} GB")
        print(f"  labels kept {', '.join(r['kept_labels'])}")
        print(f"  dropped     {', '.join(r['dropped']) or 'none'}")
        if r["missing_expected"]:
            print(f"  MISSING     {', '.join(r['missing_expected'])}")
        if r["unexpected_left_behind"]:
            print(f"  UNEXPECTED  {', '.join(r['unexpected_left_behind'])}")
        print(f"  sha256      {r['sha256']}")
        print()

    if models:
        print("model record")
        for m in models:
            if m.get("missing"):
                print(f"  MISSING  {m['dest']}  (expected at {m['source']})")
            else:
                print(f"  {m['dest']:34s} {m['bytes'] / 1e6:>8.2f} MB  {m['sha256'][:16]}…")
        print()

    print(f"manifest -> {manifest}")


if __name__ == "__main__":
    main()
