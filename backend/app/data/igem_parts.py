#!/usr/bin/env python3
"""Build the local iGEM parts snapshot the Registry search runs against.

The search is a *protein* homology search, but the Registry distributes *DNA*.
This script bridges the two: it takes a Registry export, keeps the parts that
carry a real coding sequence, translates them, and writes the protein FASTA plus
a manifest recording where the snapshot came from and when.

Two ways to get the export:

  # 1. Direct download, if the server can reach the Registry.
  uv run pyroforge-fetch-parts

  # 2. From a file you downloaded yourself. parts.igem.org sits behind a CDN
  #    that returns 403 to a lot of server traffic (including this repo's CI and
  #    at least one university network), so this is the reliable path: open
  #    https://parts.igem.org/fasta/parts/All_Parts in a browser, save it, then
  uv run pyroforge-fetch-parts --from-file ~/All_Parts.fasta

Both accept DNA or protein FASTA and detect which it is. Nothing here invents
sequences: if the export cannot be read the script fails and the app reports the
database as not provisioned, rather than answering queries against a stand-in.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx

from app.engines.registry_engine import PartsDatabase
from app.services.codon import CODON_TO_AA

# The Registry's bulk protein-coding export. Overridable because iGEM has moved
# this path before and a mirror is sometimes the only reachable copy.
DEFAULT_EXPORT_URL = "https://parts.igem.org/fasta/parts/All_Parts"

STOP = "*"
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
# Shorter than this is a promoter, RBS, terminator or scar — not something a
# protein homology search can say anything about.
MIN_PROTEIN_LEN = 30
MAX_PROTEIN_LEN = 2_000


def parse_fasta(text: str) -> list[tuple[str, str, str]]:
    """``[(part_id, description, sequence)]`` from FASTA text."""
    records: list[tuple[str, str, str]] = []
    header: str | None = None
    chunks: list[str] = []

    def flush() -> None:
        if header is None:
            return
        tokens = header.split(maxsplit=1)
        if not tokens:
            return
        seq = "".join(chunks).upper().replace(" ", "")
        if seq:
            records.append((tokens[0], tokens[1].strip() if len(tokens) > 1 else "", seq))

    for line in text.splitlines():
        if line.startswith(">"):
            flush()
            header = line[1:].strip()
            chunks = []
        elif header is not None:
            chunks.append(line.strip())
    flush()
    return records


def looks_like_dna(seq: str) -> bool:
    if not seq:
        return False
    return sum(c in "ACGTUN" for c in seq) / len(seq) > 0.9


def translate(dna: str) -> str:
    """Translate the longest ORF on the forward strand.

    Registry parts are deposited with arbitrary flanking sequence (prefix, suffix,
    RBS), so frame 0 from base 0 is usually wrong. Taking the longest ATG→stop ORF
    across the three forward frames recovers the actual CDS for the overwhelming
    majority of protein-coding BioBricks.
    """
    dna = dna.replace("U", "T")
    best = ""
    for frame in range(3):
        protein: list[str] = []
        started = False
        for i in range(frame, len(dna) - 2, 3):
            aa = CODON_TO_AA.get(dna[i : i + 3])
            if aa is None:
                protein, started = [], False
                continue
            if aa == STOP:
                if len("".join(protein)) > len(best):
                    best = "".join(protein)
                protein, started = [], False
                continue
            if not started:
                if aa != "M":
                    continue
                started = True
            protein.append(aa)
        if len("".join(protein)) > len(best):
            best = "".join(protein)
    return best


def to_protein_records(
    records: list[tuple[str, str, str]],
) -> tuple[list[tuple[str, str, str]], dict[str, int]]:
    out: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    stats = {"input": len(records), "translated": 0, "too_short": 0, "no_orf": 0, "duplicate": 0}

    for part_id, description, seq in records:
        protein = translate(seq) if looks_like_dna(seq) else seq
        if not protein:
            stats["no_orf"] += 1
            continue
        protein = "".join(c for c in protein if c in STANDARD_AA)
        if not (MIN_PROTEIN_LEN <= len(protein) <= MAX_PROTEIN_LEN):
            stats["too_short"] += 1
            continue
        # Identical sequences add search time and clutter the hit list with what
        # reads as independent evidence; keep the first part ID for each.
        if protein in seen:
            stats["duplicate"] += 1
            continue
        seen.add(protein)
        out.append((part_id, description, protein))
        stats["translated"] += 1
    return out, stats


def download(url: str) -> str:
    with httpx.Client(timeout=300.0, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
        return resp.text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from-file", type=Path, help="A Registry export saved locally")
    ap.add_argument("--url", default=DEFAULT_EXPORT_URL, help="Registry export URL")
    ap.add_argument("--out", type=Path, help="Override the output FASTA path")
    args = ap.parse_args()

    db = PartsDatabase(args.out)

    if args.from_file:
        source = f"local file: {args.from_file}"
        try:
            text = args.from_file.read_text(errors="replace")
        except OSError as exc:
            print(f"error: cannot read {args.from_file}: {exc}", file=sys.stderr)
            return 1
    else:
        source = args.url
        print(f"Downloading {args.url} …", file=sys.stderr)
        try:
            text = download(args.url)
        except httpx.HTTPError as exc:
            print(
                f"error: download failed ({exc}).\n"
                f"parts.igem.org is behind a CDN that blocks many server networks. "
                f"Open {args.url} in a browser, save the file, and re-run with "
                f"--from-file <path>.",
                file=sys.stderr,
            )
            return 1

    records = parse_fasta(text)
    if not records:
        print("error: no FASTA records found in the export", file=sys.stderr)
        return 1

    proteins, stats = to_protein_records(records)
    if not proteins:
        print("error: no protein-coding parts survived translation", file=sys.stderr)
        return 1

    detail = (
        f"{stats['input']:,} records in export; {stats['translated']:,} kept "
        f"({stats['no_orf']:,} no ORF, {stats['too_short']:,} outside "
        f"{MIN_PROTEIN_LEN}-{MAX_PROTEIN_LEN} aa, {stats['duplicate']:,} duplicate)"
    )
    n = db.write(proteins, source=source, detail=detail)
    print(f"Wrote {n:,} protein parts → {db.fasta_path}")
    print(f"Manifest → {db.manifest_path}")
    print(detail)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
