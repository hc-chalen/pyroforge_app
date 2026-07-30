"""Meltome Atlas ingestion — add thousands of proteins with absolute Tm labels.

Absolute Tm in PyroForge is data-limited: only ~237 distinct proteins carry Tm
labels, and a learning curve shows skill still climbing at 237 with no plateau,
so the fix is *more proteins*, not a bigger model. This module wires the FLIP
Meltome split (Jarzab et al. 2020, cross-species meltome) into the same pipeline:

  1. download the FASTA (sequence + TARGET=Tm + SET=train/test),
  2. filter (length, valid AAs), de-duplicate by sequence,
  3. embed with the SAME frozen ESM2-650M mean-pooled recipe as the training data
     (app.services.esm2.ESM2Service — layer 33, residues mean-pooled),
  4. cache an npz {emb, tm, split, source} that the Tm trainer consumes.

Meltome Tm (cell-lysate thermal proteome profiling) is measured differently from
the folding-based Tm already in the dataset, so downstream training carries a
per-source indicator to absorb the systematic offset (see train_tm_meltome).

Run:  uv run python -m app.training.ingest_meltome --max-proteins 4000 --max-len 400
      uv run python -m app.training.ingest_meltome --stats   # parse only, no embedding
"""
from __future__ import annotations

import argparse
import logging
import shutil
import urllib.request
from pathlib import Path

import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)

MELTOME_URL = "http://data.bioembeddings.com/public/FLIP/fasta/meltome/mixed_split.fasta"
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def download_meltome(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        logger.info("meltome FASTA already present at %s", dest)
        return dest
    logger.info("downloading Meltome FASTA -> %s", dest)
    # The host's bot filter returns HTTP 406 for both urllib's default
    # "Python-urllib/x" UA *and* browser-like "Mozilla/..." UAs; it only accepts
    # plain CLI-tool agents (curl/wget), so present a curl UA.
    req = urllib.request.Request(MELTOME_URL, headers={"User-Agent": "curl/8.4.0"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as fh:
        shutil.copyfileobj(resp, fh)
    return dest


def parse_fasta(path: Path) -> list[tuple[str, float, str]]:
    """Return [(sequence, Tm, split)] from the FLIP meltome FASTA headers."""
    records: list[tuple[str, float, str]] = []
    seq_parts: list[str] = []
    tm: float | None = None
    split = "train"

    def flush():
        if tm is not None and seq_parts:
            records.append(("".join(seq_parts), tm, split))

    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                flush()
                seq_parts = []
                tm = None
                split = "train"
                for tok in line[1:].split():
                    if tok.startswith("TARGET="):
                        tm = float(tok.split("=", 1)[1])
                    elif tok.startswith("SET="):
                        split = tok.split("=", 1)[1].lower()
            else:
                seq_parts.append(line)
    flush()
    return records


def select(records, min_len=20, max_len=400, max_proteins=4000, seed=42):
    """Filter by length/AA, de-duplicate by sequence, cap protein count.

    .. warning::
       De-duplication is by **exact sequence only**, and the split comes from FLIP's
       ``mixed_split``. Homologous (non-identical) sequences can therefore appear on
       both sides of the train/test boundary, so the resulting Meltome test metrics
       (r ≈ 0.83 / MAE ≈ 5.4 °C) are an optimistic estimate of generalization to
       genuinely novel proteins — and the served ±95% Tm band derived from them is
       correspondingly too narrow. A clustered split (e.g. MMseqs2 at 30% identity,
       whole clusters held out) is required before those numbers can be cited as
       novel-protein performance.
    """
    seen: set[str] = set()
    kept: list[tuple[str, float, str]] = []
    for seq, tm, split in records:
        seq = seq.strip().upper()
        if not (min_len <= len(seq) <= max_len):
            continue
        if set(seq) - STANDARD_AA:
            continue
        if seq in seen:
            continue
        seen.add(seq)
        kept.append((seq, tm, split))
    # keep the whole official test split; subsample only the train pool
    test = [r for r in kept if r[2] == "test"]
    train = [r for r in kept if r[2] != "test"]
    rng = np.random.default_rng(seed)
    if max_proteins and len(train) > max_proteins:
        idx = rng.choice(len(train), max_proteins, replace=False)
        train = [train[i] for i in idx]
    return train + test


def embed_sequences(seqs: list[str], batch_size: int = 8) -> np.ndarray:
    from app.services.esm2 import ESM2Service

    svc = ESM2Service.get_instance()
    out = np.empty((len(seqs), 1280), dtype=np.float32)
    for i in range(0, len(seqs), batch_size):
        batch = seqs[i : i + batch_size]
        emb = svc.batch_embed(batch, batch_size=batch_size).numpy()
        out[i : i + len(batch)] = emb
        if (i // batch_size) % 20 == 0:
            logger.info("embedded %d/%d", i + len(batch), len(seqs))
    return out


def build_and_cache(out_path: Path, max_proteins: int, max_len: int, batch_size: int = 8) -> dict:
    fasta = download_meltome(settings.checkpoint_dir / "improved" / "meltome_mixed.fasta")
    records = parse_fasta(fasta)
    logger.info("parsed %d meltome entries", len(records))
    sel = select(records, max_len=max_len, max_proteins=max_proteins)
    seqs = [s for s, _, _ in sel]
    tm = np.array([t for _, t, _ in sel], dtype=np.float32)
    split = np.array([sp for _, _, sp in sel])
    logger.info("embedding %d proteins (max_len=%d) ...", len(seqs), max_len)
    emb = embed_sequences(seqs, batch_size=batch_size)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, emb=emb, tm=tm, split=split, source=np.full(len(tm), "meltome"),
             # Recorded so downstream reports can state the caveat in ``select``.
             split_method=np.array("flip_mixed_split:exact_sequence_dedup_only"))
    logger.info("cached %d meltome proteins -> %s", len(tm), out_path)
    return {"n": len(tm), "n_test": int((split == "test").sum()), "path": str(out_path)}


def _stats(max_len: int):
    fasta = download_meltome(settings.checkpoint_dir / "improved" / "meltome_mixed.fasta")
    records = parse_fasta(fasta)
    lens = np.array([len(s) for s, _, _ in records])
    tms = np.array([t for _, t, _ in records])
    for L in [200, 300, 400, 500, 700]:
        n = int((lens <= L).sum())
        print(f"  len<= {L:>4}: {n:>6} entries")
    print(f"total={len(records)}  Tm mean={tms.mean():.1f} std={tms.std():.1f} "
          f"range=[{tms.min():.1f},{tms.max():.1f}]  median_len={np.median(lens):.0f}")


def main():
    ap = argparse.ArgumentParser(description="Ingest Meltome Atlas Tm data into PyroForge")
    ap.add_argument("--max-proteins", type=int, default=4000)
    ap.add_argument("--max-len", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--stats", action="store_true", help="parse + length stats only (no embedding)")
    ap.add_argument("--out", type=Path, default=settings.checkpoint_dir / "improved" / "meltome_tm.npz")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    if args.stats:
        _stats(args.max_len)
        return
    print(build_and_cache(args.out, args.max_proteins, args.max_len, args.batch_size))


if __name__ == "__main__":
    main()
