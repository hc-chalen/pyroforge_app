#!/usr/bin/env python3
"""Derive a codon usage table from a reference genome's annotated CDS.

Why this exists: the E. coli table in ``app/services/codon.py`` was transcribed by
hand from a Kazusa entry built on **14 CDS / 5,122 codons**. That is a real table,
but it is far too small to settle which codon a synonymous family prefers. Scoring
the margin between the top two codons in each family against Poisson noise, 7 of the
18 multi-codon families were unresolved at 2σ — Ala, Cys, Phe, His, Asn, Arg and Tyr
were effectively coin flips:

    Asn   AAC=125 vs AAT=112   0.8σ
    Tyr   TAT=86  vs TAC=75    0.9σ
    His   CAT=81  vs CAC=67    1.2σ

Hand-transcribing 64 numbers is also how the wrong table got in: before the Kazusa
one, this module shipped weights that marked AAG/GAG/GCC/GTG as E. coli's preferred
codons, which is the *human* preference. Deriving the counts from an annotated
genome removes both problems at once — the sample is the whole proteome, and nobody
types a number.

Usage:

    uv run python -m scripts.build_codon_table --organism ecoli
    uv run python -m scripts.build_codon_table --organism yeast
    uv run python -m scripts.build_codon_table --organism ecoli --from-file cds.fna.gz

It prints a ``CodonUsage(...)`` literal to stdout for pasting into
``app/services/codon.py``. The table is committed as source rather than loaded at
runtime so that serving needs no data file and the diff is reviewable — but it is
now *generated*, and this script is what makes the number reproducible.

Note on what the CAI means. These are genome-wide counts, so the CAI scored against
them is "adaptation to the genome-average gene". Classical CAI (Sharp & Li 1987,
NAR 15:1281-1295) uses a *highly-expressed* reference set instead, which is a
different and for codon-optimization purposes more useful target. The app labels
which one it reports; see ``CodonUsage.provenance``.
"""

from __future__ import annotations

import argparse
import gzip
import io
import re
import sys
from collections import Counter
from pathlib import Path

import httpx

from app.services.codon import GENETIC_CODE

# RefSeq "CDS from genomic" FASTA: one record per annotated coding sequence, which
# is exactly the population a codon usage table is defined over.
GENOMES = {
    "ecoli": {
        "label": "Escherichia coli K-12 MG1655",
        "accession": "NC_000913.3",
        "url": (
            "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/845/"
            "GCF_000005845.2_ASM584v2/GCF_000005845.2_ASM584v2_cds_from_genomic.fna.gz"
        ),
    },
    "yeast": {
        "label": "Saccharomyces cerevisiae S288C",
        "accession": "GCF_000146045.2",
        "url": (
            "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/146/045/"
            "GCF_000146045.2_R64_cds_from_genomic.fna.gz"
        ),
    },
}

CODONS = sorted({c for codons in GENETIC_CODE.values() for c in codons})


def iter_records(text: str):
    """``(header, sequence)`` pairs from FASTA text."""
    header: str | None = None
    chunks: list[str] = []
    for line in text.splitlines():
        if line.startswith(">"):
            if header is not None:
                yield header, "".join(chunks)
            header = line[1:]
            chunks = []
        elif header is not None:
            chunks.append(line.strip())
    if header is not None:
        yield header, "".join(chunks)


def count_codons(text: str) -> tuple[Counter, dict[str, int]]:
    """Count codons over every usable CDS.

    Rejections are counted rather than silently dropped, because "how much of the
    proteome went into this table" is part of the table's provenance:

      * ``pseudo=true`` — annotated pseudogenes. Their reading frames are broken, so
        their codons are not evidence of translational preference.
      * ``exception=`` — records with an annotated translational exception
        (ribosomal slippage, selenocysteine). Reading them in frame is wrong.
      * length not a multiple of 3, or a non-ACGT character (RefSeq uses N for
        ambiguity) anywhere in the record.
    """
    counts: Counter = Counter()
    stats = {"records": 0, "used": 0, "pseudo": 0, "exception": 0, "bad_length": 0, "ambiguous": 0}

    for header, seq in iter_records(text):
        stats["records"] += 1
        if "pseudo=true" in header:
            stats["pseudo"] += 1
            continue
        if "exception=" in header:
            stats["exception"] += 1
            continue
        seq = seq.upper()
        if not seq or len(seq) % 3:
            stats["bad_length"] += 1
            continue
        if re.search(r"[^ACGT]", seq):
            stats["ambiguous"] += 1
            continue
        # The terminal stop is part of the CDS and part of the codon census; CAI
        # excludes stop codons separately, at scoring time (see CAI_EXCLUDED_AA).
        counts.update(seq[i : i + 3] for i in range(0, len(seq), 3))
        stats["used"] += 1

    return counts, stats


def render(organism: str, label: str, accession: str, counts: Counter, n_cds: int) -> str:
    """Emit the CodonUsage literal, laid out four codons per line like the original."""
    total = sum(counts[c] for c in CODONS)
    lines = [
        f'    "{organism}": CodonUsage(',
        f'        label="{label}",',
        f'        source="NCBI RefSeq {accession}, all annotated CDS",',
        f"        n_cds={n_cds:_},",
        f"        n_codons={total:_},",
        "        counts={",
    ]
    for i in range(0, len(CODONS), 4):
        row = "".join(f'"{c}": {counts[c]:_}, ' for c in CODONS[i : i + 4]).rstrip()
        lines.append(f"            {row}")
    lines.append("        },")
    lines.append("    ),")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--organism", choices=sorted(GENOMES), required=True)
    ap.add_argument("--from-file", type=Path, help="A CDS FASTA saved locally (.fna or .fna.gz)")
    args = ap.parse_args()

    genome = GENOMES[args.organism]

    if args.from_file:
        raw = args.from_file.read_bytes()
    else:
        print(f"Downloading {genome['url']} …", file=sys.stderr)
        try:
            with httpx.Client(timeout=300.0, follow_redirects=True) as client:
                resp = client.get(genome["url"])
                resp.raise_for_status()
                raw = resp.content
        except httpx.HTTPError as exc:
            print(f"error: download failed ({exc})", file=sys.stderr)
            return 1

    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    text = io.BytesIO(raw).read().decode("ascii", "replace")

    counts, stats = count_codons(text)
    if not stats["used"]:
        print("error: no usable CDS found", file=sys.stderr)
        return 1

    print(
        f"{stats['records']:,} CDS records; {stats['used']:,} used "
        f"({stats['pseudo']:,} pseudogenes, {stats['exception']:,} translational "
        f"exceptions, {stats['bad_length']:,} not a multiple of 3, "
        f"{stats['ambiguous']:,} with ambiguous bases)",
        file=sys.stderr,
    )
    print(render(args.organism, genome["label"], genome["accession"], counts, stats["used"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
