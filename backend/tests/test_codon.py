"""Fast, data-free tests for codon optimization and CAI.

These guard the property the previous hand-maintained weight tables broke: CAI
relative adaptiveness must be normalized *within each synonymous family*, so the
best codon of every family scores exactly 1.0. Without that, a perfectly
codon-optimized gene scored 0.82 rather than 1.0 — a number that then reached the
synthesis quote card as if it were a real CAI.

Run directly (``python tests/test_codon.py``) or with ``pytest``.
"""
from __future__ import annotations

from app.services.codon import (
    CAI_EXCLUDED_AA,
    CODON_TABLES,
    CODON_USAGE,
    GENETIC_CODE,
    CodonOptimizer,
    relative_adaptiveness,
)

# A sequence covering all 20 residues, including the S/R/G/P families whose
# weights were the ones left unnormalized.
PROTEIN = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQ"


def test_every_synonymous_family_peaks_at_one() -> None:
    for organism, table in CODON_TABLES.items():
        for aa, codons in GENETIC_CODE.items():
            peak = max(table[c] for c in codons)
            assert abs(peak - 1.0) < 1e-9, (
                f"{organism}: family {aa!r} peaks at {peak:.3f}; relative "
                f"adaptiveness must be normalized within each family"
            )


def test_fully_optimized_sequence_scores_cai_one() -> None:
    """A gene built from every family's best codon must score exactly 1.0.

    Built here from the table rather than from ``CodonOptimizer.optimize``: the
    optimizer no longer emits the family maximum unconditionally — it gives some
    adaptiveness back to avoid homopolymer runs and hairpins — so calling it here
    would test the back-translation policy instead of the normalization invariant
    this file exists to guard. See test_procurement for the optimizer's own
    contract.
    """
    opt = CodonOptimizer()
    for organism, table in CODON_TABLES.items():
        dna = "".join(
            max(GENETIC_CODE[aa], key=lambda c: table[c]) for aa in PROTEIN
        )
        cai = opt.analyze(dna, organism)["cai"]
        assert cai == 1.0, f"{organism}: best-codon sequence scored CAI {cai}, expected 1.0"


def test_cai_excludes_single_codon_families() -> None:
    """Met/Trp/stop carry no codon choice, so they must not move the index."""
    opt = CodonOptimizer()
    # A worst-codon gene, with and without a run of Met (whose w is 1.0).
    worst = "".join(
        min(GENETIC_CODE[aa], key=lambda c: CODON_TABLES["ecoli"][c]) for aa in "LSRP"
    )
    assert opt._compute_cai(worst, "ecoli") == opt._compute_cai(worst + "ATG" * 5, "ecoli")


def test_counts_match_published_totals() -> None:
    """The transcribed counts must sum to the totals the source reports."""
    for organism, usage in CODON_USAGE.items():
        total = sum(usage.counts.values())
        assert total == usage.n_codons, (
            f"{organism}: transcribed counts sum to {total:,}, but the source "
            f"reports {usage.n_codons:,} codons — a transcription error"
        )
        assert set(usage.counts) == {c for cs in GENETIC_CODE.values() for c in cs}


def test_unobserved_codon_gets_a_floor_not_zero() -> None:
    """log(0) is undefined, so a zero-count codon must not produce w = 0."""
    counts = {c: 10 for cs in GENETIC_CODE.values() for c in cs}
    counts["TTA"] = 0
    weights = relative_adaptiveness(counts)
    assert 0.0 < weights["TTA"] < 1.0


def test_cai_reference_is_reported_with_the_number() -> None:
    result = CodonOptimizer().analyze("ATGAAA", "ecoli")
    assert "Escherichia coli" in result["cai_reference"]
    assert "CDS" in result["cai_reference"]


def test_unsupported_organism_raises_rather_than_falling_back() -> None:
    opt = CodonOptimizer()
    for call in (lambda: opt.optimize("MK", "human"), lambda: opt.provenance("human")):
        try:
            call()
            raise AssertionError("expected ValueError for an unsupported organism")
        except ValueError as exc:
            assert "human" in str(exc)


def test_stop_and_met_are_the_excluded_families() -> None:
    assert CAI_EXCLUDED_AA == {"M", "W", "*"}


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
