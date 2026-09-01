#!/usr/bin/env python3
"""Build the local iGEM parts snapshot the Registry search runs against.

The search is a *protein* homology search, but the Registry distributes *DNA*.
This script bridges the two: it takes a Registry export, keeps the parts that
carry a real coding sequence, translates them, and writes the protein FASTA plus
a manifest recording where the snapshot came from and when.

Three ways to get the export, in the order worth trying:

  # 1. The MySQL dump of the Registry's `parts` table. THE WORKING PATH as of
  #    2026-09: the FASTA endpoint below now answers 403 from iGEM's own Apache
  #    (not the CDN), so it is gone rather than blocked. The dump is linked from
  #    registry.igem.org; download it in a browser, then:
  uv run pyroforge-fetch-parts --from-sql ~/sql_parts.sql.gz

  # 2. Direct download of the bulk FASTA, if the server can reach the Registry.
  uv run pyroforge-fetch-parts

  # 3. From a FASTA you downloaded yourself. parts.igem.org sits behind a CDN
  #    that returns 403 to a lot of server traffic (including this repo's CI and
  #    at least one university network), so this used to be the reliable path:
  #    open https://parts.igem.org/fasta/parts/All_Parts in a browser, save it, then
  uv run pyroforge-fetch-parts --from-file ~/All_Parts.fasta

All three converge on the same translation, length filter and de-duplication, so a
snapshot means the same thing whichever route built it — only the manifest's
``source`` differs. The FASTA routes accept DNA or protein and detect which it is.

Nothing here invents sequences: if the export cannot be read the script fails and
the app reports the database as not provisioned, rather than answering queries
against a stand-in.
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
    ap.add_argument("--from-file", type=Path, help="A Registry FASTA export saved locally")
    ap.add_argument(
        "--from-sql",
        type=Path,
        help="A mysqldump of the Registry's `parts` table (.sql or .sql.gz)",
    )
    ap.add_argument("--url", default=DEFAULT_EXPORT_URL, help="Registry export URL")
    ap.add_argument("--out", type=Path, help="Override the output FASTA path")
    args = ap.parse_args()

    db = PartsDatabase(args.out)

    if args.from_sql:
        # Imported here, not at module scope: the SQL reader is only needed on this
        # path and pulls in zlib/regex machinery the FASTA routes never touch.
        from app.data.igem_sql import iter_parts

        try:
            records, row_stats = iter_parts(args.from_sql)
        except (OSError, ValueError) as exc:
            print(f"error: cannot read {args.from_sql}: {exc}", file=sys.stderr)
            return 1
        source = f"iGEM parts SQL dump: {args.from_sql.name}"
        # Row-level rejections are reported separately from translation-level ones.
        # They answer different questions — "what was in the dump" vs "what could be
        # turned into a searchable protein" — and collapsing them hides which stage
        # a missing part was lost at.
        row_detail = (
            f"{row_stats['rows']:,} rows in dump; {row_stats['kept']:,} with a usable "
            f"sequence ({row_stats['deleted']:,} deleted/discontinued, "
            f"{row_stats['no_sequence']:,} empty sequence)"
        )
        print(row_detail, file=sys.stderr)
        return _write_snapshot(db, records, source, prefix=row_detail + "; ")

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

    return _write_snapshot(db, records, source)


def _write_snapshot(
    db: PartsDatabase,
    records: list[tuple[str, str, str]],
    source: str,
    prefix: str = "",
) -> int:
    """Translate, filter and write the snapshot. Shared by all three input routes.

    Keeping this common means a snapshot built from the SQL dump and one built from
    the FASTA export are filtered identically — the manifest's ``source`` is the only
    thing that differs, which is what makes the two comparable.
    """
    proteins, stats = to_protein_records(records)
    if not proteins:
        print("error: no protein-coding parts survived translation", file=sys.stderr)
        return 1

    detail = prefix + (
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
