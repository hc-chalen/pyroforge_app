"""ESMFold smoke test + timing benchmark.

Run this on the GPU server BEFORE building the fold service. It answers the two
questions that decide the service's parameters:

  1. Does ESMFold run on this card at all, at the lengths PyroForge allows?
  2. How long does one fold take? (that number sets the queue policy)

Usage:

    CUDA_VISIBLE_DEVICES=2 python benchmark.py

The PDB written for the first sequence is a real ESMFold output — check that its
B-factor column carries pLDDT, because VisualizationEngine.approximate_rmsf()
reads exactly that column.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import EsmForProteinFolding

# PyroForge caps folding at 400 residues (ESMFOLD_MAX_LENGTH in
# app/engines/visualization.py). Sweep up to that so we know the worst case.
LENGTHS = [50, 100, 200, 300, 400]

# The sequence that triggered this whole investigation — 84 residues, valid,
# rejected by the public API with a 504.
REAL_SEQUENCE = (
    "MSGPNGDLGMPVEAGAEGEEDGFGEAEYAAINSMLDQINSCLDHLEEKND"
    "HLHARLQELLESNRQTRLEFQQQLGEAPSDASP"
)

# Filler used to build synthetic sequences of an exact length for the sweep.
# Composition barely affects runtime or memory; length dominates both.
_FILLER = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKR"


def synthetic(length: int) -> str:
    return (_FILLER * (length // len(_FILLER) + 1))[:length]


def load_model(chunk_size: int) -> EsmForProteinFolding:
    print("loading facebook/esmfold_v1 ... (first run downloads ~10 GB)", flush=True)
    t0 = time.perf_counter()
    model = EsmForProteinFolding.from_pretrained(
        "facebook/esmfold_v1",
        low_cpu_mem_usage=True,
    )
    model = model.cuda().eval()

    # fp16 on the ESM-2 3B stem roughly halves its resident footprint. Keep the
    # folding trunk in fp32 — it is numerically touchier and much smaller.
    model.esm = model.esm.half()

    # TF32 is Ampere+ only (3090 yes, 2080 Ti no). Harmless either way.
    torch.backends.cuda.matmul.allow_tf32 = True

    # Splits the trunk's pair-representation ops into chunks. Lower = less peak
    # memory, slower. Only matters for long sequences.
    model.trunk.set_chunk_size(chunk_size)

    print(f"loaded in {time.perf_counter() - t0:.1f}s", flush=True)
    return model


def fold_once(model: EsmForProteinFolding, seq: str) -> tuple[str, float, float]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    with torch.no_grad():
        pdb = model.infer_pdb(seq)
    torch.cuda.synchronize()

    elapsed = time.perf_counter() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    return pdb, elapsed, peak_gb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--out", type=Path, default=Path("benchmark_out"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device visible. Check CUDA_VISIBLE_DEVICES.")

    name = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"device: {name}  ({total_gb:.1f} GB)\n", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    model = load_model(args.chunk_size)
    results = []

    # Real sequence first: this is the one that must work, and its PDB is the
    # artifact worth eyeballing.
    pdb, elapsed, peak = fold_once(model, REAL_SEQUENCE)
    (args.out / "real_84.pdb").write_text(pdb)
    print(f"real sequence  len= 84  {elapsed:6.1f}s  peak {peak:5.2f} GB", flush=True)
    results.append({"label": "real", "length": 84, "seconds": round(elapsed, 1),
                    "peak_gb": round(peak, 2)})

    for length in LENGTHS:
        try:
            _, elapsed, peak = fold_once(model, synthetic(length))
        except torch.cuda.OutOfMemoryError:
            print(f"synthetic      len={length:4d}  OUT OF MEMORY", flush=True)
            results.append({"label": "synthetic", "length": length, "oom": True})
            torch.cuda.empty_cache()
            continue
        print(f"synthetic      len={length:4d}  {elapsed:6.1f}s  peak {peak:5.2f} GB",
              flush=True)
        results.append({"label": "synthetic", "length": length,
                        "seconds": round(elapsed, 1), "peak_gb": round(peak, 2)})

    summary = {
        "device": name,
        "total_gb": round(total_gb, 1),
        "chunk_size": args.chunk_size,
        "results": results,
    }
    (args.out / "benchmark.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out}/benchmark.json and {args.out}/real_84.pdb", flush=True)


if __name__ == "__main__":
    main()
