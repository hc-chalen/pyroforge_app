"""S669 loader — the external blind benchmark (Pancotti et al., BiB 2022).

Why this dataset. It is the field's standard blind set for ΔΔG predictors: 669
single-point variants curated from ThermoMutDB, selected to be absent from the
S2648/VariBench training corpora that most published tools were fit on. Two
properties make it unusually valuable here:

  * it ships **both directions** of every variant (``Experimental_DDG_dir`` and
    ``..._inv``), so the antisymmetry that ``train_ddg`` enforces can finally be
    *measured* rather than merely trained for; and
  * it ships **published predictions from 20 tools** on these exact variants, so
    the comparison table needs no reimplementation of anyone else's method — in
    particular ``DDGun``/``DDGun3D``, the untrained baseline the benchmarking
    plan calls for.

Sequences. The distributed CSV indexes mutations two ways: ``PDB_Mut`` uses PDB
residue numbering and ``Seq_Mut`` uses 1-based sequence position (they differ for
333 of the 669 rows). We take sequences from the ATOM records of the bundled
wild-type structures, which reproduces ``Seq_Mut`` for 639/669 rows directly.
The residue-numbering convention is not quite uniform across the archive, so for
proteins that do not match we search for a single constant offset consistent with
*every* mutation of that protein — a per-protein offset that satisfies all its
rows is very unlikely to be coincidental. Rows still unresolved are dropped and
counted in the manifest rather than silently coerced.

Download (~44 MB, Zenodo record 7568094)::

    uv run python -m scripts.bench.data --fetch
"""

from __future__ import annotations

import csv
import logging
import re
import subprocess
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

S669_URL = "https://zenodo.org/records/7568094/files/S669.zip?download=1"

# Published tools whose predictions travel with the CSV. Suffixed ``_dir``/``_inv``.
# "-Seq" / "INPS-Seq" / "MuPro" / "SAAFEC-Seq" are sequence-only like model_ddg;
# the rest consume a structure and belong in a different comparison class.
PUBLISHED_TOOLS = [
    "ACDC-NN", "ACDC-NN-Seq", "DDGun", "DDGun3D", "DUET", "Dynamut", "FoldX",
    "I-Mutant3.0", "I-Mutant3.0-Seq", "INPS-Seq", "INPS3D", "MAESTRO", "mCSM",
    "MuPro", "PopMusic", "PremPS", "SAAFEC-Seq", "SDM", "ThermoNet",
]
SEQUENCE_ONLY_TOOLS = {
    "ACDC-NN-Seq", "DDGun", "I-Mutant3.0-Seq", "INPS-Seq", "MuPro", "SAAFEC-Seq",
}

_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M",
}

_MUT_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")

# ESM-2 has no length limit but attention is O(L^2); past ~1000 residues a single
# forward pass on MPS costs minutes and dominates the whole run. Sequences above
# this are excluded and reported in the coverage manifest, never silently cut —
# truncating would move the mutation site and produce a wrong answer quietly.
MAX_EMBED_LENGTH = 1022

# The regime model_ddg was trained on (Tsuboyama mega-scale is overwhelmingly
# small designed domains) and the API's own cap. Results are reported split on
# this, because a pooled number across both strata is not interpretable.
IN_REGIME_MAX_LENGTH = 400


@dataclass
class Variant:
    """One S669 row, resolved to sequences the served path can consume."""

    protein: str            # e.g. "1a0fA" (PDB id + chain)
    wt_seq: str
    mut_seq: str
    pos0: int               # 0-based index into wt_seq
    wt_aa: str
    mt_aa: str
    ddg_dir: float          # experimental, direct (wt -> mut). Sign: >0 stabilizing.
    ddg_inv: float          # experimental, reverse (mut -> wt)
    tools: dict = field(default_factory=dict)   # {"DDGun": (dir, inv), ...}

    @property
    def length(self) -> int:
        return len(self.wt_seq)

    @property
    def in_regime(self) -> bool:
        return self.length <= IN_REGIME_MAX_LENGTH


def fetch(data_dir: Path) -> None:
    """Download + extract the S669 archive into ``data_dir`` if absent."""
    data_dir.mkdir(parents=True, exist_ok=True)
    if (data_dir / "S669" / "S669.csv").exists():
        logger.info("S669 already present at %s", data_dir)
        return
    zip_path = data_dir / "S669.zip"
    if not zip_path.exists():
        logger.info("downloading S669 (~44 MB) ...")
        subprocess.run(["curl", "-sL", "-o", str(zip_path), S669_URL], check=True)
    subprocess.run(["unzip", "-qo", str(zip_path), "-d", str(data_dir)], check=True)
    logger.info("S669 extracted to %s", data_dir)


def _atom_sequence(pdb_path: Path, chain: str) -> str:
    """Chain sequence in ATOM-record order, one entry per residue id.

    Keyed on columns 22:27 (resSeq + iCode) so insertion codes do not collapse
    two residues into one.
    """
    residues: OrderedDict[str, str] = OrderedDict()
    with pdb_path.open() as handle:
        for line in handle:
            if line[:6] not in ("ATOM  ", "HETATM") or line[21] != chain:
                continue
            name = line[17:20].strip()
            if name in _THREE_TO_ONE:
                residues.setdefault(line[22:27], _THREE_TO_ONE[name])
    return "".join(residues.values())


def _resolve_offset(seq: str, mutations: list[tuple[str, int]]) -> int | None:
    """Smallest |offset| making *every* (wt_aa, pos) of a protein agree with ``seq``.

    Requiring one offset to satisfy all of a protein's mutations is what makes
    this safe: a wrong offset would have to coincidentally match every row.
    """
    for offset in sorted(range(-60, 61), key=abs):
        if all(
            0 <= pos - 1 + offset < len(seq) and seq[pos - 1 + offset] == wt
            for wt, pos in mutations
        ):
            return offset
    return None


def load_s669(data_dir: Path) -> tuple[list[Variant], dict]:
    """Return resolved variants plus a manifest describing what was dropped."""
    root = data_dir / "S669"
    rows = list(csv.DictReader((root / "S669.csv").open()))

    # Sequences are per (WT_PDB, Chain); cache so each structure is parsed once.
    seq_cache: dict[tuple[str, str], str] = {}
    by_protein: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_protein[row["Protein"]].append(row)

    variants: list[Variant] = []
    dropped_unresolved = 0
    for protein, prot_rows in by_protein.items():
        key = (prot_rows[0]["WT_PDB"], prot_rows[0]["Chain"])
        if key not in seq_cache:
            seq_cache[key] = _atom_sequence(root / "pdbs" / f"{key[0]}.pdb", key[1])
        seq = seq_cache[key]

        parsed = []
        for row in prot_rows:
            match = _MUT_RE.match(row["Seq_Mut"])
            if match:
                parsed.append((row, match.group(1), int(match.group(2)), match.group(3)))
        offset = _resolve_offset(seq, [(wt, pos) for _, wt, pos, _ in parsed])
        if offset is None:
            dropped_unresolved += len(prot_rows)
            logger.debug("no consistent offset for %s — dropping %d rows", protein, len(prot_rows))
            continue

        for row, wt_aa, pos, mt_aa in parsed:
            pos0 = pos - 1 + offset
            mut_seq = seq[:pos0] + mt_aa + seq[pos0 + 1:]
            tools = {}
            for tool in PUBLISHED_TOOLS:
                try:
                    tools[tool] = (float(row[f"{tool}_dir"]), float(row[f"{tool}_inv"]))
                except (KeyError, ValueError):
                    pass
            variants.append(Variant(
                protein=protein, wt_seq=seq, mut_seq=mut_seq, pos0=pos0,
                wt_aa=wt_aa, mt_aa=mt_aa,
                ddg_dir=float(row["Experimental_DDG_dir"]),
                ddg_inv=float(row["Experimental_DDG_inv"]),
                tools=tools,
            ))

    embeddable = [v for v in variants if v.length <= MAX_EMBED_LENGTH]
    manifest = {
        "dataset": "S669",
        "reference": "Pancotti et al., Briefings in Bioinformatics 23(2):bbab555 (2022)",
        "rows_in_csv": len(rows),
        "rows_resolved": len(variants),
        "rows_dropped_no_offset": dropped_unresolved,
        "rows_dropped_too_long": len(variants) - len(embeddable),
        "max_embed_length": MAX_EMBED_LENGTH,
        "rows_scored": len(embeddable),
        "proteins_scored": len({v.protein for v in embeddable}),
        "in_regime_rows": sum(1 for v in embeddable if v.in_regime),
        "in_regime_max_length": IN_REGIME_MAX_LENGTH,
        "out_of_regime_rows": sum(1 for v in embeddable if not v.in_regime),
    }
    return embeddable, manifest


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/bench"))
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    if args.fetch:
        fetch(args.data_dir)
    variants, manifest = load_s669(args.data_dir)
    for key, value in manifest.items():
        print(f"{key:28s} {value}")


if __name__ == "__main__":
    main()
