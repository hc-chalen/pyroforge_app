from __future__ import annotations

import math
import re
from dataclasses import dataclass

GENETIC_CODE = {
    "F": ["TTT", "TTC"], "L": ["TTA", "TTG", "CTT", "CTC", "CTA", "CTG"],
    "S": ["TCT", "TCC", "TCA", "TCG", "AGT", "AGC"], "Y": ["TAT", "TAC"],
    "*": ["TAA", "TAG", "TGA"], "C": ["TGT", "TGC"], "W": ["TGG"],
    "P": ["CCT", "CCC", "CCA", "CCG"], "H": ["CAT", "CAC"], "Q": ["CAA", "CAG"],
    "R": ["CGT", "CGC", "CGA", "CGG", "AGA", "AGG"], "I": ["ATT", "ATC", "ATA"],
    "M": ["ATG"], "T": ["ACT", "ACC", "ACA", "ACG"], "N": ["AAT", "AAC"],
    "K": ["AAA", "AAG"], "V": ["GTT", "GTC", "GTA", "GTG"],
    "A": ["GCT", "GCC", "GCA", "GCG"], "D": ["GAT", "GAC"],
    "E": ["GAA", "GAG"], "G": ["GGT", "GGC", "GGA", "GGG"],
}

CODON_TO_AA = {c: aa for aa, codons in GENETIC_CODE.items() for c in codons}

_COMPLEMENT = str.maketrans("ACGT", "TGCA")

# Runs the back-translator refuses to create: 5+ identical bases. Deliberately one
# stricter than the `(.)\1{5,}` (6+) run that raises the ``homopolymer_run``
# feasibility flag, so the optimizer leaves headroom rather than emitting sequence
# that sits exactly on the flag's boundary. Kept in sync with
# CodonOptimizer.MAX_HOMOPOLYMER.
_HOMOPOLYMER_RUN = re.compile(r"(.)\1{4,}")

# Amino acids whose family has exactly one codon. Their relative adaptiveness is
# 1.0 by construction, so they carry no information about codon choice and the
# standard CAI definition excludes them (along with stop codons).
SINGLE_CODON_AA = frozenset(aa for aa, codons in GENETIC_CODE.items() if len(codons) == 1)
CAI_EXCLUDED_AA = SINGLE_CODON_AA | {"*"}


@dataclass(frozen=True)
class CodonUsage:
    """A codon usage table plus the provenance needed to interpret its numbers.

    ``counts`` holds *raw observed codon counts*, not pre-computed weights, and
    the relative adaptiveness used for CAI is derived from them by
    :func:`relative_adaptiveness`. That direction matters: the previous tables
    stored hand-maintained weights in which several synonymous families
    (Ser, Arg, Gly, Pro) were never normalized to a family maximum of 1.0, so a
    perfectly codon-optimized gene scored CAI ≈ 0.82 instead of ≈ 1.0. Deriving
    the weights makes that class of error structurally impossible — see the
    self-check at the bottom of this module.
    """

    label: str
    source: str
    n_cds: int
    n_codons: int
    counts: dict[str, int]

    @property
    def provenance(self) -> str:
        return f"{self.label} — {self.source}, {self.n_cds:,} CDS / {self.n_codons:,} codons"


# ---------------------------------------------------------------------------
# Observed codon usage. Values transcribed from the Kazusa Codon Usage Database
# (https://www.kazusa.or.jp/codon/), which tabulates GenBank CDS codon counts.
#
# IMPORTANT — what these are and are not:
#
#   * They are *genome-wide* codon usage. Classical CAI (Sharp & Li 1987) scores
#     against a **highly-expressed gene** reference set, which is a materially
#     different (and for codon optimization, more useful) target. The CAI this
#     module reports is therefore "adaptation to the genome-average gene", and
#     the API labels it as such rather than implying Sharp & Li CAI.
#   * The E. coli K-12 entry in Kazusa is built from only 14 CDS / 5,122 codons.
#     It gets the large directional calls right — and notably corrects the table
#     that used to live here, which marked AAG/GAG/GCC/GTG as E. coli's preferred
#     codons when E. coli in fact prefers AAA over AAG ~2.7:1 and GAA over GAG
#     ~2.4:1 (that set of "preferred" codons is the human one). But the sample is
#     small and it should be replaced by a larger, highly-expressed reference set
#     before these sequences are ordered. Because the weights are derived, that
#     is a counts-only change.
# ---------------------------------------------------------------------------
CODON_USAGE: dict[str, CodonUsage] = {
    "ecoli": CodonUsage(
        label="Escherichia coli K-12",
        source="Kazusa Codon Usage Database (GenBank CDS)",
        n_cds=14,
        n_codons=5_122,
        counts={
            "TTT": 101, "TTC": 77, "TTA": 78, "TTG": 61,
            "TCT": 29, "TCC": 28, "TCA": 40, "TCG": 41,
            "TAT": 86, "TAC": 75, "TAA": 9, "TAG": 0,
            "TGT": 30, "TGC": 41, "TGA": 5, "TGG": 55,
            "CTT": 61, "CTC": 54, "CTA": 27, "CTG": 240,
            "CCT": 43, "CCC": 33, "CCA": 34, "CCG": 137,
            "CAT": 81, "CAC": 67, "CAA": 62, "CAG": 142,
            "CGT": 108, "CGC": 133, "CGA": 22, "CGG": 21,
            "ATT": 156, "ATC": 93, "ATA": 19, "ATG": 127,
            "ACT": 41, "ACC": 117, "ACA": 33, "ACG": 59,
            "AAT": 112, "AAC": 125, "AAA": 170, "AAG": 62,
            "AGT": 37, "AGC": 85, "AGA": 7, "AGG": 8,
            "GTT": 86, "GTC": 60, "GTA": 59, "GTG": 135,
            "GCT": 55, "GCC": 162, "GCA": 108, "GCG": 197,
            "GAT": 194, "GAC": 105, "GAA": 224, "GAG": 94,
            "GGT": 109, "GGC": 171, "GGA": 47, "GGG": 44,
        },
    ),
    "yeast": CodonUsage(
        label="Saccharomyces cerevisiae",
        source="Kazusa Codon Usage Database (GenBank CDS)",
        n_cds=14_411,
        n_codons=6_534_504,
        counts={
            "TTT": 170_666, "TTC": 120_510, "TTA": 170_884, "TTG": 177_573,
            "TCT": 153_557, "TCC": 92_923, "TCA": 122_028, "TCG": 55_951,
            "TAT": 122_728, "TAC": 96_596, "TAA": 6_913, "TAG": 3_312,
            "TGT": 52_903, "TGC": 31_095, "TGA": 4_447, "TGG": 67_789,
            "CTT": 80_076, "CTC": 35_545, "CTA": 87_619, "CTG": 68_494,
            "CCT": 88_263, "CCC": 44_309, "CCA": 119_641, "CCG": 34_597,
            "CAT": 89_007, "CAC": 50_785, "CAA": 178_251, "CAG": 79_121,
            "CGT": 41_791, "CGC": 16_993, "CGA": 19_562, "CGG": 11_351,
            "ATT": 196_893, "ATC": 112_176, "ATA": 116_254, "ATG": 136_805,
            "ACT": 132_522, "ACC": 83_207, "ACA": 116_084, "ACG": 52_045,
            "AAT": 233_124, "AAC": 162_199, "AAA": 273_618, "AAG": 201_361,
            "AGT": 92_466, "AGC": 63_726, "AGA": 139_081, "AGG": 60_289,
            "GTT": 144_243, "GTC": 76_947, "GTA": 76_927, "GTG": 70_337,
            "GCT": 138_358, "GCC": 82_357, "GCA": 105_910, "GCG": 40_358,
            "GAT": 245_641, "GAC": 132_048, "GAA": 297_944, "GAG": 125_717,
            "GGT": 156_109, "GGC": 63_903, "GGA": 71_216, "GGG": 39_359,
        },
    ),
}


def relative_adaptiveness(counts: dict[str, int]) -> dict[str, float]:
    """CAI relative adaptiveness ``w`` from raw codon counts.

    ``w(codon) = count(codon) / max(count of any codon in the same synonymous
    family)``, so every family's best codon is exactly 1.0. That normalization is
    the *definition* of relative adaptiveness; without it CAI is not on the 0–1
    scale where 1.0 means "perfectly adapted", which is how every consumer of the
    number reads it.

    An unobserved codon would make ``log w`` undefined, so zero counts take the
    conventional 0.5-count floor rather than 0.
    """
    weights: dict[str, float] = {}
    for codons in GENETIC_CODE.values():
        observed = [counts[c] for c in codons]
        peak = max(observed)
        if peak <= 0:  # pragma: no cover - a family with no observations at all
            weights.update({c: 1.0 for c in codons})
            continue
        for codon in codons:
            n = counts[codon]
            weights[codon] = (n if n > 0 else 0.5) / peak
    return weights


CODON_TABLES: dict[str, dict[str, float]] = {
    organism: relative_adaptiveness(usage.counts)
    for organism, usage in CODON_USAGE.items()
}


def _assert_normalized() -> None:
    """Guard the invariant the previous hand-maintained tables silently broke."""
    for organism, table in CODON_TABLES.items():
        missing = set(CODON_TO_AA) - set(table)
        assert not missing, f"{organism}: codons missing from table: {sorted(missing)}"
        for aa, codons in GENETIC_CODE.items():
            peak = max(table[c] for c in codons)
            assert abs(peak - 1.0) < 1e-9, (
                f"{organism}: synonymous family {aa!r} peaks at {peak:.3f}, not 1.0 — "
                f"relative adaptiveness must be normalized within each family."
            )


_assert_normalized()


class CodonOptimizer:
    @staticmethod
    def _table(organism: str) -> dict[str, float]:
        """Relative-adaptiveness table for ``organism``.

        Raises rather than falling back to E. coli: a silent fallback returns
        E. coli-optimized DNA while the response still echoes the requested
        organism, and scores its CAI against the wrong reference set.
        """
        try:
            return CODON_TABLES[organism]
        except KeyError:
            raise ValueError(
                f"Unsupported expression organism {organism!r}. "
                f"Supported: {', '.join(sorted(CODON_TABLES))}."
            ) from None

    @staticmethod
    def provenance(organism: str) -> str:
        """Human-readable description of the reference set ``organism`` scores against.

        Returned alongside the CAI so the number is never shown bare: a CAI is only
        meaningful relative to a stated reference set.
        """
        try:
            return CODON_USAGE[organism].provenance
        except KeyError:
            raise ValueError(
                f"Unsupported expression organism {organism!r}. "
                f"Supported: {', '.join(sorted(CODON_USAGE))}."
            ) from None

    # A codon this far below its family's best is a rare codon; substituting one in
    # to break up a repeat trades a synthesis problem for a translation problem.
    MIN_ACCEPTABLE_WEIGHT = 0.2
    # Manufacturability constraints, in the same units the feasibility flags use.
    MAX_HOMOPOLYMER = 4        # 5+ identical bases in a row is a synthesis flag
    HAIRPIN_STEM = 8
    HAIRPIN_MAX_LOOP = 12

    def optimize(self, protein: str, organism: str = "ecoli") -> str:
        """Back-translate ``protein`` into a synthesizable, codon-adapted CDS.

        Per residue this takes the highest-weight codon that does **not** create a
        homopolymer run or a hairpin stem, falling back to the highest-weight codon
        outright when every option is constrained.

        The previous version was a plain per-residue ``argmax``, which is wrong in
        two ways that both showed up on the dashboard. Because it always picked the
        family maximum, and relative adaptiveness is 1.0 at the family maximum by
        construction, **CAI was identically 1.000 for every input** — the number
        carried no information about the sequence at all. And using exactly one
        codon per amino acid produces heavy local repetition: on GFP it emitted 20
        distinct codons across 238 positions and tripped both ``homopolymer_run``
        and ``potential_hairpin``, so the feasibility flags fired on essentially
        every order and stopped meaning anything.
        """
        table = self._table(organism)
        out: list[str] = []
        dna = ""

        for i, aa in enumerate(protein):
            options = GENETIC_CODE.get(aa)
            if options is None:
                # Non-standard residue (X, B, Z, U). There is no codon for it; emit
                # the placeholder rather than guessing, and let analyze() see it.
                out.append("NNN")
                dna += "NNN"
                continue

            next_aa = protein[i + 1] if i + 1 < len(protein) else None
            chosen = self._choose_codon(dna, options, table, next_aa)
            out.append(chosen)
            dna += chosen
        return "".join(out)

    def _acceptable(self, options: list[str], table: dict[str, float]) -> list[str]:
        """Options ranked by adaptiveness, dropping rare codons.

        Substituting a rare codon in to break up a repeat trades a synthesis
        problem for a translation problem, so the family's poor performers are not
        candidates at all — unless the whole family is poor, in which case the best
        available one is still better than nothing.
        """
        ranked = sorted(options, key=lambda c: table[c], reverse=True)
        usable = [c for c in ranked if table[c] >= self.MIN_ACCEPTABLE_WEIGHT]
        return usable or ranked[:1]

    def _choose_codon(
        self,
        dna: str,
        options: list[str],
        table: dict[str, float],
        next_aa: str | None,
    ) -> str:
        """Pick this residue's codon, looking one residue ahead.

        Pure greedy paints itself into corners that a single step of lookahead
        avoids. On GFP's ``…D-F-F…`` the greedy choice GAT+TTT leaves four T's, and
        then *both* Phe codons start ``TT``, so the second F is forced to extend the
        run to seven regardless of what it picks. Choosing TTC for the first F costs
        a little adaptiveness and leaves the next residue free.

        When every option violates a constraint the run is unavoidable (poly-lysine
        is the honest example), so the pick is the one that keeps the damage
        smallest and the feasibility flag is left to fire.
        """
        ranked = self._acceptable(options, table)
        viable = [c for c in ranked if not self._violates_constraints(dna, c)]

        if viable and next_aa is not None:
            next_options = GENETIC_CODE.get(next_aa)
            if next_options is not None:
                next_ranked = self._acceptable(next_options, table)
                unblocking = [
                    c
                    for c in viable
                    if any(
                        not self._violates_constraints(dna + c, n) for n in next_ranked
                    )
                ]
                if unblocking:
                    return unblocking[0]
        if viable:
            return viable[0]
        # Nothing is clean: prefer the shortest resulting run, then adaptiveness.
        return min(ranked, key=lambda c: (self._longest_run(dna + c), -table[c]))

    @staticmethod
    def _longest_run(dna: str) -> int:
        tail = dna[-16:]
        return max((len(m.group()) for m in re.finditer(r"(.)\1*", tail)), default=0)

    def _violates_constraints(self, dna: str, codon: str) -> bool:
        """Would appending ``codon`` create a homopolymer run or a hairpin stem?

        Only the junction is examined — the prefix was already checked when it was
        built — so this stays O(1) per candidate instead of rescanning the whole
        CDS for every one of the ~6 options at every residue.
        """
        extended = dna + codon
        # The new run can end anywhere inside the appended codon, not only at the
        # last base: "…GAT" + "TTC" makes TTTTTT with a C after it. So the window
        # spans the codon plus the longest run that could precede it.
        window = extended[-(self.MAX_HOMOPOLYMER + len(codon)) :]
        if _HOMOPOLYMER_RUN.search(window):
            return True
        return self._closes_hairpin(extended)

    def _closes_hairpin(self, dna: str) -> bool:
        """True if a stem ending at the last base pairs with an upstream segment."""
        stem, loop = self.HAIRPIN_STEM, self.HAIRPIN_MAX_LOOP
        if len(dna) < 2 * stem:
            return False
        segment = dna[-stem:]
        rev_comp = segment.translate(_COMPLEMENT)[::-1]
        window_start = max(0, len(dna) - 2 * stem - loop)
        return rev_comp in dna[window_start : len(dna) - stem]

    def analyze(self, dna: str, organism: str = "ecoli") -> dict:
        dna = dna.upper()
        # Percentage (0–100) — the feasibility thresholds below are in the same unit.
        gc = (dna.count("G") + dna.count("C")) / len(dna) * 100 if dna else 0.0
        cai = self._compute_cai(dna, organism)
        flags = []
        if gc < 25 or gc > 65:
            flags.append("gc_content_extreme")
        if re.search(r"(.)\1{5,}", dna):
            flags.append("homopolymer_run")
        if self._has_hairpin(dna):
            flags.append("potential_hairpin")
        return {
            "cai": cai,
            "cai_reference": self.provenance(organism),
            "gc_content": round(gc, 2),
            "feasibility_flags": flags,
        }

    def _compute_cai(self, dna: str, organism: str = "ecoli") -> float:
        """Codon Adaptation Index: geometric mean of relative adaptiveness.

        Scored against ``organism``'s own table — scoring yeast-optimized DNA with
        E. coli weights is meaningless.

        Met, Trp and stop codons are excluded, per the standard definition: their
        families have a single codon (or no coding role), so ``w`` is 1.0 by
        construction and including them only dilutes the index toward 1 in
        proportion to how many of them the protein happens to contain.
        """
        table = self._table(organism)
        weights = []
        for i in range(0, len(dna) - 2, 3):
            codon = dna[i : i + 3]
            aa = CODON_TO_AA.get(codon)
            if aa is None:
                # Not a codon at all (e.g. the NNN placeholder for a non-standard
                # residue). Counting it as "maximally unadapted" would be a silent
                # judgement about unknown sequence, so skip it.
                continue
            if aa in CAI_EXCLUDED_AA:
                continue
            weights.append(table[codon])
        if not weights:
            return 0.0
        log_sum = sum(math.log(w) for w in weights) / len(weights)
        return round(math.exp(log_sum), 4)

    @staticmethod
    def _has_hairpin(dna: str, stem: int = 8, max_loop: int = 12) -> bool:
        """Detect a self-complementary stem that could form a hairpin.

        The previous implementation searched for a ``window//2``-long reverse
        complement inside a slice of exactly ``window//2`` characters, so ``in``
        degenerated into an equality test: it only matched a perfect 10 bp stem with a
        zero-length loop, and never fired on real sequences. This scans for a
        ``stem``-length reverse complement anywhere within ``max_loop`` bases
        downstream, which is what a hairpin actually is.
        """
        for i in range(len(dna) - stem):
            segment = dna[i : i + stem]
            if len(segment) < stem:
                break
            rev_comp = segment.translate(_COMPLEMENT)[::-1]
            search_from = i + stem
            if rev_comp in dna[search_from : search_from + max_loop + stem]:
                return True
        return False
