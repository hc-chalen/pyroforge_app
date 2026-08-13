from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from Bio.Align import PairwiseAligner

from app.core.config import settings

logger = logging.getLogger(__name__)


class PartsDatabase:
    """The local iGEM parts snapshot the registry search runs against.

    Two files, side by side:

      * ``igem_parts.fasta``   — protein sequences, ``>PART_ID description``.
      * ``igem_parts.manifest.json`` — where the snapshot came from and when.

    Both are produced by ``backend/scripts/fetch_igem_parts.py``. If they are
    absent the database is **not provisioned**, and that is reported as such —
    it is never silently replaced by a handful of built-in sequences. The
    previous version did exactly that: it wrote three hard-coded fluorescent
    proteins and then answered every query against them, so a real protein came
    back "No matching parts found" while the UI claimed to have searched the
    iGEM Registry.
    """

    def __init__(self, fasta_path: Path | None = None) -> None:
        self.fasta_path = fasta_path or settings.igem_parts_fasta_path
        self.manifest_path = self.fasta_path.with_suffix(".manifest.json")
        self._entries: list[tuple[str, str]] | None = None
        self._entries_mtime: float | None = None

    @property
    def is_provisioned(self) -> bool:
        return self.fasta_path.exists() and self.fasta_path.stat().st_size > 0

    def manifest(self) -> dict:
        """Provenance of the snapshot, always safe to call."""
        if not self.is_provisioned:
            return {
                "status": "not_provisioned",
                "n_parts": 0,
                "source": None,
                "built_at": None,
                "detail": (
                    "No iGEM parts snapshot on this server. Build one with "
                    "`uv run pyroforge-fetch-parts` (see docs/OPERATIONS.md)."
                ),
            }
        meta: dict = {}
        if self.manifest_path.exists():
            try:
                meta = json.loads(self.manifest_path.read_text())
            except (OSError, json.JSONDecodeError):
                meta = {}
        return {
            "status": "ready",
            "n_parts": meta.get("n_parts", len(self.entries())),
            "source": meta.get("source"),
            "built_at": meta.get("built_at"),
            "detail": meta.get("detail"),
        }

    def entries(self) -> list[tuple[str, str]]:
        """``[(part_id, sequence)]``, cached until the FASTA changes on disk."""
        if not self.is_provisioned:
            return []
        mtime = self.fasta_path.stat().st_mtime
        if self._entries is None or self._entries_mtime != mtime:
            self._entries = self._parse_fasta()
            self._descriptions = {pid: d for pid, d in self._parse_descriptions()}
            self._entries_mtime = mtime
        return self._entries

    def description(self, part_id: str) -> str:
        self.entries()  # populate the description map alongside the sequences
        return getattr(self, "_descriptions", {}).get(part_id, "")

    def _iter_records(self):
        header = ""
        seq_parts: list[str] = []
        for line in self.fasta_path.read_text().splitlines():
            if line.startswith(">"):
                if header:
                    yield header, "".join(seq_parts)
                header = line[1:].strip()
                seq_parts = []
            else:
                seq_parts.append(line.strip())
        if header:
            yield header, "".join(seq_parts)

    def _parse_fasta(self) -> list[tuple[str, str]]:
        return [
            (h.split()[0], s.upper()) for h, s in self._iter_records() if h.split()
        ]

    def _parse_descriptions(self) -> list[tuple[str, str]]:
        out = []
        for header, _ in self._iter_records():
            tokens = header.split(maxsplit=1)
            if tokens:
                out.append((tokens[0], tokens[1].strip() if len(tokens) > 1 else ""))
        return out

    def write(self, records: list[tuple[str, str, str]], source: str, detail: str = "") -> int:
        """Write the snapshot: ``records`` is ``[(part_id, description, protein)]``."""
        self.fasta_path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for part_id, description, seq in records:
            header = f">{part_id} {description}".rstrip()
            lines.append(f"{header}\n{seq}")
        self.fasta_path.write_text("\n".join(lines) + "\n")
        self.manifest_path.write_text(
            json.dumps(
                {
                    "n_parts": len(records),
                    "source": source,
                    "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "detail": detail,
                },
                indent=2,
            )
            + "\n"
        )
        self._entries = None
        return len(records)


class MMseqsService:
    """Homology search against the local parts snapshot.

    Prefers the ``mmseqs`` binary; degrades to a pairwise BLOSUM62 alignment when
    it is missing *or when it fails*. The failure path matters: mmseqs ships one
    tarball per instruction set, and a mismatched build dies with SIGILL on the
    first AVX2 instruction. That used to propagate as a failed Celery job with an
    opaque ``CalledProcessError``; now the search still returns, flagged with the
    method that actually produced it.
    """

    # Below this alignment identity a fallback hit is not evidence of homology, so it
    # is not reported at all. With affine gap penalties an unrelated pair still scores
    # ~15-25%, which is noise, not a match.
    FALLBACK_MIN_IDENTITY = 30.0

    # A pairwise scan is O(database) full alignments, so it cannot stand in for
    # mmseqs on a real snapshot (tens of thousands of parts) — it would hold the
    # single-slot worker for minutes. Past this size the honest answer is that the
    # search cannot run, not a search that silently takes forever.
    FALLBACK_MAX_PARTS = 2_000

    def __init__(self, database: PartsDatabase | None = None) -> None:
        self.mmseqs_bin = settings.mmseqs_bin
        self.db = database or PartsDatabase()

    def _mmseqs_available(self) -> bool:
        return shutil.which(self.mmseqs_bin) is not None

    def _fallback_search(self, sequence: str, max_hits: int) -> list[dict]:
        """Pairwise-alignment fallback used when mmseqs2 is unavailable or failed.

        Uses BLOSUM62 with affine gap penalties. The PairwiseAligner defaults
        (match 1, mismatch 0, **no gap penalty**) make the global score essentially
        a longest-common-subsequence length and inflate "identity" for completely
        unrelated sequences. There is no E-value computable here at all, so it is
        reported as ``None`` rather than as ``0.0`` — the strongest possible value.
        """
        from Bio.Align import substitution_matrices

        aligner = PairwiseAligner()
        aligner.mode = "global"
        aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
        aligner.open_gap_score = -11
        aligner.extend_gap_score = -1

        hits: list[dict] = []
        for part_id, target_seq in self.db.entries():
            try:
                alignment = next(iter(aligner.align(sequence, target_seq)), None)
            except (ValueError, KeyError):
                # A residue outside BLOSUM62 (X, U, B, Z) in either sequence.
                continue
            if alignment is None:
                continue
            a, b = alignment[0], alignment[1]
            aligned = sum(1 for x, y in zip(a, b) if x != "-" and y != "-")
            matches = sum(1 for x, y in zip(a, b) if x == y and x != "-")
            identity = (matches / aligned * 100) if aligned else 0.0
            if identity < self.FALLBACK_MIN_IDENTITY:
                continue
            hits.append({
                "part_id": part_id,
                "identity": round(identity, 2),
                "evalue": None,
                "bits": float(alignment.score),
                "search_method": "pairwise_blosum62_fallback",
            })
        hits.sort(key=lambda h: h["bits"], reverse=True)
        return hits[:max_hits]

    @staticmethod
    def _normalize_identity(raw: float) -> float:
        """MMseqs2 reports ``pident`` as a 0–1 fraction on some builds and 0–100 on
        others. The UI renders it as a percentage, so a fraction would display as
        "0.9%" for a 92% match. Normalize to 0–100 here, once.
        """
        return raw * 100.0 if raw <= 1.0 else raw

    def _mmseqs_search(self, sequence: str, max_hits: int) -> list[dict]:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            query_fasta = tmp_path / "query.fasta"
            query_fasta.write_text(f">query\n{sequence}\n")
            result_m8 = tmp_path / "result.m8"
            subprocess.run(
                [
                    self.mmseqs_bin,
                    "easy-search",
                    str(query_fasta),
                    str(self.db.fasta_path),
                    str(result_m8),
                    str(tmp_path / "tmp"),
                    "--format-output",
                    "query,target,pident,evalue,bits",
                ],
                check=True,
                capture_output=True,
            )
            hits = []
            if result_m8.exists():
                for line in result_m8.read_text().splitlines()[:max_hits]:
                    parts = line.split("\t")
                    if len(parts) < 5:
                        continue
                    target = parts[1]
                    hits.append({
                        "part_id": target.split()[0] if target else target,
                        "identity": round(self._normalize_identity(float(parts[2])), 2),
                        "evalue": float(parts[3]),
                        "bits": float(parts[4]),
                        "search_method": "mmseqs2",
                    })
            return hits

    def search(self, sequence: str, max_hits: int = 10) -> tuple[list[dict], str | None]:
        """Returns ``(hits, unavailable_reason)``.

        ``unavailable_reason`` is non-None when no search could be run at all —
        the caller reports that instead of an empty result set, because "we could
        not search" and "we searched and found nothing" are different answers.
        """
        if not self.db.is_provisioned:
            return [], "no_database"

        if self._mmseqs_available():
            try:
                return self._mmseqs_search(sequence, max_hits), None
            except (subprocess.CalledProcessError, OSError) as exc:
                logger.warning(
                    "mmseqs easy-search failed (%s); falling back to pairwise alignment",
                    exc,
                )

        if len(self.db.entries()) > self.FALLBACK_MAX_PARTS:
            return [], "mmseqs_required"
        return self._fallback_search(sequence, max_hits), None


class RegistryEngine:
    """Homology search against a local iGEM parts snapshot.

    Part descriptions come from the snapshot itself, not from a live call to
    parts.igem.org. That endpoint sits behind a CDN that returns 403 to server
    traffic, so the per-hit fetch it used to do returned an empty description
    every time — after spending up to 15 s per hit on the ``fast`` Celery queue,
    which also serves predictions. Search is now fully offline.
    """

    UNAVAILABLE_DETAIL = {
        "no_database": (
            "No iGEM parts snapshot is installed on this server, so no search was "
            "run. Build one with `uv run pyroforge-fetch-parts`."
        ),
        "mmseqs_required": (
            "The parts snapshot is too large for the pairwise fallback and the "
            "mmseqs2 binary is unavailable, so no search was run."
        ),
    }

    def __init__(self, database: PartsDatabase | None = None) -> None:
        self.db = database or PartsDatabase()
        self.mmseqs = MMseqsService(self.db)

    @staticmethod
    def part_url(part_id: str) -> str:
        return f"https://parts.igem.org/Part:{part_id}"

    def search(self, sequence: str, max_hits: int = 10) -> dict:
        seq = sequence.strip().upper()
        raw_hits, unavailable = self.mmseqs.search(seq, max_hits=max_hits)
        matches = [
            {
                **hit,
                "description": self.db.description(hit["part_id"]),
                "biobrick_url": self.part_url(hit["part_id"]),
            }
            for hit in raw_hits
        ]
        return {
            "matches": matches,
            "query_length": len(seq),
            # Provenance travels with every result: how many parts were actually
            # searched, and where that snapshot came from. Without it "no matches"
            # is unreadable — it could mean the query has no homologs, or that the
            # database has three sequences in it.
            "database": self.db.manifest(),
            "unavailable_reason": unavailable,
            "unavailable_detail": self.UNAVAILABLE_DETAIL.get(unavailable or ""),
        }
