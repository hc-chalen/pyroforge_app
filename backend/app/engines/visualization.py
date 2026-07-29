from __future__ import annotations

import hashlib
import time
from pathlib import Path

import httpx

from app.core.config import settings

VALID_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
ESMFOLD_MAX_LENGTH = 400
ESMFOLD_MIN_LENGTH = 10
RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})
MAX_FOLD_RETRIES = 3
FOLD_READ_TIMEOUT_S = 300.0


class VisualizationEngine:
    """Structure folding and per-residue visualization data for frontend renderers."""

    def __init__(self) -> None:
        self.structure_cache = settings.cache_dir / "structures"
        self.structure_cache.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, sequence: str) -> Path:
        digest = hashlib.sha256(sequence.encode()).hexdigest()[:16]
        return self.structure_cache / f"{digest}.pdb"

    @staticmethod
    def validate_sequence(sequence: str) -> str:
        seq = sequence.strip().upper()
        if len(seq) < ESMFOLD_MIN_LENGTH:
            raise ValueError(
                f"Sequence must be at least {ESMFOLD_MIN_LENGTH} amino acids for folding.",
            )
        if len(seq) > ESMFOLD_MAX_LENGTH:
            raise ValueError(
                f"Sequence length {len(seq)} exceeds the ESMFold limit "
                f"({ESMFOLD_MAX_LENGTH} residues).",
            )
        invalid = sorted({char for char in seq if char not in VALID_AMINO_ACIDS})
        if invalid:
            raise ValueError(
                f"Invalid amino acid characters: {''.join(invalid)}",
            )
        return seq

    def _fetch_pdb_from_esmfold(self, seq: str) -> str:
        timeout = httpx.Timeout(
            connect=30.0,
            read=FOLD_READ_TIMEOUT_S,
            write=30.0,
            pool=30.0,
        )
        last_error: Exception | None = None

        for attempt in range(MAX_FOLD_RETRIES):
            try:
                with httpx.Client(timeout=timeout) as client:
                    response = client.post(
                        settings.esmfold_api_url,
                        data=seq,
                        headers={"Content-Type": "text/plain"},
                    )
                    if response.status_code in RETRYABLE_STATUS_CODES:
                        response.raise_for_status()
                    response.raise_for_status()
                    return response.text
            except httpx.TimeoutException as exc:
                last_error = exc
                if attempt == MAX_FOLD_RETRIES - 1:
                    raise ValueError(
                        "ESMFold API timed out. Try a shorter sequence or retry "
                        "in a few minutes.",
                    ) from exc
            except httpx.HTTPStatusError as exc:
                last_error = exc
                status = exc.response.status_code
                if status not in RETRYABLE_STATUS_CODES:
                    raise self._fold_http_error(exc, seq) from exc
                if attempt == MAX_FOLD_RETRIES - 1:
                    raise self._fold_http_error(exc, seq) from exc
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt == MAX_FOLD_RETRIES - 1:
                    raise ValueError(
                        "Could not reach the ESMFold API. Check your network and retry.",
                    ) from exc

            time.sleep(2**attempt)

        raise ValueError(
            "ESMFold structure prediction failed after multiple attempts.",
        ) from last_error

    @staticmethod
    def _fold_http_error(exc: httpx.HTTPStatusError, seq: str) -> ValueError:
        status = exc.response.status_code
        if status == 422:
            return ValueError(
                "ESMFold rejected this sequence. Use a valid protein sequence "
                f"between {ESMFOLD_MIN_LENGTH} and {ESMFOLD_MAX_LENGTH} residues.",
            )
        if status in RETRYABLE_STATUS_CODES:
            return ValueError(
                f"ESMFold API error ({status}). The service may be overloaded — "
                "please retry in a few minutes.",
            )
        return ValueError(
            f"ESMFold API returned HTTP {status} for a {len(seq)}-residue sequence.",
        )

    def fold_structure(self, sequence: str) -> dict:
        seq = self.validate_sequence(sequence)
        cache = self._cache_path(seq)
        if cache.exists():
            pdb_text = cache.read_text()
        else:
            pdb_text = self._fetch_pdb_from_esmfold(seq)
            cache.write_text(pdb_text)

        return {
            "pdb": pdb_text,
            "pdb_path": str(cache),
            "sequence_length": len(seq),
        }

    def build_heatmap(
        self,
        sequence: str,
        residue_scores: dict[int, float],
    ) -> list[dict]:
        if not residue_scores:
            return []
        values = list(residue_scores.values())
        vmin, vmax = min(values), max(values)
        span = vmax - vmin if vmax != vmin else 1.0
        result = []
        for pos, aa in enumerate(sequence):
            score = residue_scores.get(pos, 0.0)
            normalized = min(1.0, max(0.0, (score - vmin) / span))
            result.append({
                "position": pos + 1,
                "aa": aa,
                "score": score,
                "normalized": normalized,
                "color": self._score_to_color(normalized),
            })
        return result

    @staticmethod
    def _score_to_color(normalized: float) -> str:
        t = min(1.0, max(0.0, normalized))
        r = int(255 * t)
        b = int(255 * (1 - t))
        return f"rgb({r}, 80, {b})"

    @staticmethod
    def _plddt_from_bfactor(bfactor: float) -> float:
        """ESMFold emits pLDDT as 0–1 or 0–100 in the PDB B-factor column."""
        if bfactor <= 1.0:
            return max(0.0, min(1.0, bfactor))
        return max(0.0, min(1.0, bfactor / 100.0))

    def approximate_rmsf(self, pdb_path: str) -> list[dict]:
        """Extract per-residue pLDDT (B-factor) from PDB as rigidity proxy."""
        path = Path(pdb_path)
        if not path.exists():
            raise FileNotFoundError(f"PDB not found: {pdb_path}")

        residues: dict[tuple[str, int], float] = {}
        for line in path.read_text().splitlines():
            if not line.startswith("ATOM"):
                continue
            chain = line[21].strip() or "A"
            resnum = int(line[22:26])
            bfactor = float(line[60:66])
            plddt = self._plddt_from_bfactor(bfactor)
            key = (chain, resnum)
            residues[key] = max(residues.get(key, 0.0), plddt)

        return [
            {
                "chain": chain,
                "position": resnum,
                "rmsf_proxy": round(plddt * 100.0, 2),
                "rigidity": plddt,
            }
            for (chain, resnum), plddt in sorted(residues.items(), key=lambda x: x[0][1])
        ]
