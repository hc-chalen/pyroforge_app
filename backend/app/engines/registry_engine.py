from __future__ import annotations

import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from Bio.Align import PairwiseAligner

from app.core.config import settings


class MMseqsService:
    def __init__(self) -> None:
        self.mmseqs_bin = settings.mmseqs_bin
        self.db_path = settings.mmseqs_db_path
        self.fasta_path = settings.igem_parts_fasta_path

    def _mmseqs_available(self) -> bool:
        return shutil.which(self.mmseqs_bin) is not None

    def ensure_database(self) -> None:
        # No _create_db call: `search` runs `mmseqs easy-search` against the FASTA
        # directly and never reads db_path, so building the DB was pure wasted work
        # on first request. Restore it only if search switches to `mmseqs search`.
        self.fasta_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.fasta_path.exists():
            self._bootstrap_fasta()

    def _bootstrap_fasta(self) -> None:
        """Fetch a representative set of iGEM protein parts."""
        # iGEM provides a parts export; fallback to curated seed sequences
        seed_parts = [
            (">BBa_K176000 GFP", "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"),
            (">BBa_K864100 mCherry", "MVSKGEEDNMAIIKEFMRFKVHMEGSVNGHEFEIEGEGEGRPYEGTQTAKLKVTKGGPLPFAWDILSPQFMYGSKAYVKHPADIPDYLKLSFPEGFKWERVMNFEDGGVVTVTQDSSLQDGEFIYKVKLRGTNFPSDGPVMQKKTMGWEASSERMYPEDGALKGEIKQRLKLKDGGHYDAEVKTTYKAKKPVQLPGAYNVNIKLDITSHNEDYTIVEQYERAEGRHSTGGMDELYK"),
            (">BBa_I746001 RFP", "MSELIKENMHMKLYMEGTVNNYFFSPMIYFAFLLFYGVTLYFLYGQYRFRKKMPSIEFDDSLAGEILELSLQIDDMIPAFLLSKVYTQHQRVSFFEFMNFLVHLSVCQSCGFEITVFECYFALCYRFLVSHYKVYLRCKKYLVCESCMEDVCKIEQQLMREVE"),
        ]
        self.fasta_path.write_text("\n".join(f"{h}\n{s}" for h, s in seed_parts) + "\n")
        # (An iGEM enrichment request used to run here and discard its response; it
        # was removed rather than left as a no-op network call on first request.)

    def _create_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [self.mmseqs_bin, "createdb", str(self.fasta_path), str(self.db_path)],
            check=True,
            capture_output=True,
        )

    def _parse_fasta(self) -> list[tuple[str, str]]:
        entries: list[tuple[str, str]] = []
        header = ""
        seq_parts: list[str] = []
        for line in self.fasta_path.read_text().splitlines():
            if line.startswith(">"):
                if header:
                    entries.append((header, "".join(seq_parts)))
                header = line[1:].split()[0]
                seq_parts = []
            else:
                seq_parts.append(line.strip())
        if header:
            entries.append((header, "".join(seq_parts)))
        return entries

    # Below this alignment identity a fallback hit is not evidence of homology, so it
    # is not reported at all. With affine gap penalties an unrelated pair still scores
    # ~15-25%, which is noise, not a match.
    FALLBACK_MIN_IDENTITY = 30.0

    def _fallback_search(self, sequence: str, max_hits: int) -> list[dict]:
        """Pairwise-alignment fallback used when the mmseqs2 binary is unavailable.

        Uses BLOSUM62 with affine gap penalties. The previous version used the
        PairwiseAligner defaults (match 1, mismatch 0, **no gap penalty**), which makes
        the global score essentially a longest-common-subsequence length and inflates
        "identity" for completely unrelated sequences. It also reported
        ``evalue: 0.0`` — the strongest possible E-value — on every hit. There is no
        E-value here at all, so it is now reported as ``None``.
        """
        from Bio.Align import substitution_matrices

        aligner = PairwiseAligner()
        aligner.mode = "global"
        aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
        aligner.open_gap_score = -11
        aligner.extend_gap_score = -1

        hits: list[dict] = []
        for part_id, target_seq in self._parse_fasta():
            alignment = next(iter(aligner.align(sequence, target_seq)), None)
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
                # No E-value is computable from a pairwise alignment without a
                # database model; 0.0 would read as a perfect hit.
                "evalue": None,
                "bits": float(alignment.score),
                "search_method": "pairwise_blosum62_fallback",
            })
        hits.sort(key=lambda h: h["bits"], reverse=True)
        return hits[:max_hits]

    def search(self, sequence: str, max_hits: int = 10) -> list[dict]:
        self.ensure_database()
        if not self._mmseqs_available():
            return self._fallback_search(sequence, max_hits)

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
                    str(self.fasta_path),
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
                    part_id = target.split()[0] if target else target
                    hits.append({
                        "part_id": part_id,
                        "identity": float(parts[2]),
                        "evalue": float(parts[3]),
                        "bits": float(parts[4]),
                    })
            return hits


class IGEMService:
    def fetch_part_metadata(self, part_id: str) -> dict:
        try:
            with httpx.Client(timeout=15.0) as client:
                resp = client.get(
                    f"{settings.igem_api_base}/xml/part.cgi",
                    params={"part": part_id},
                )
                resp.raise_for_status()
                text = resp.text
        except httpx.HTTPError:
            return {"part_id": part_id, "description": "", "biobrick_url": self._part_url(part_id)}

        description = ""
        if "<part_name>" in text:
            start = text.find("<part_name>") + len("<part_name>")
            end = text.find("</part_name>", start)
            description = text[start:end] if end > start else part_id

        return {
            "part_id": part_id,
            "description": description,
            "biobrick_url": self._part_url(part_id),
        }

    @staticmethod
    def _part_url(part_id: str) -> str:
        return f"https://parts.igem.org/Part:{part_id}"


class RegistryEngine:
    """MMseqs2 alignment against iGEM parts + registry metadata."""

    def __init__(self) -> None:
        self.mmseqs = MMseqsService()
        self.igem = IGEMService()

    def search(self, sequence: str, max_hits: int = 10) -> dict:
        seq = sequence.strip().upper()
        raw_hits = self.mmseqs.search(seq, max_hits=max_hits)
        # Fetch part metadata concurrently. Serially this was up to max_hits × 15 s of
        # blocking HTTP (150 s at the default) inside the single-slot Celery worker,
        # holding up every other user's job.
        matches: list[dict] = []
        if raw_hits:
            with ThreadPoolExecutor(max_workers=min(8, len(raw_hits))) as pool:
                metas = list(
                    pool.map(
                        lambda h: self.igem.fetch_part_metadata(h["part_id"]), raw_hits
                    )
                )
            matches = [{**hit, **meta} for hit, meta in zip(raw_hits, metas)]
        return {"matches": matches, "query_length": len(seq)}
