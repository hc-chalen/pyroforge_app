"""Procurement contract: an estimate must not be shaped like a quote, and the
codon optimizer's reported metrics must actually vary with the input.
"""

from __future__ import annotations

import re

import pytest

from app.engines.procurement import ProcurementEngine
from app.services.codon import CODON_TO_AA, CodonOptimizer
from app.services.twist import TwistService

GFP = (
    "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDH"
    "MKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMA"
    "DKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITH"
    "GMDELYK"
)
LYSOZYME = (
    "MKALIVLGLVLLSVTVQGKVFERCELARTLKRLGMDGYRGISLANWMCLAKWESGYNTRATNYNAGDRSTDYGIFQI"
    "NSRYWCNDGKTPGAVNACHLSCSALLQDNIADAVACAKRVVRDPQGIRAWVAWRNRCQNRDVRQYVQGCGV"
)


def translate(dna: str) -> str:
    return "".join(CODON_TO_AA.get(dna[i : i + 3], "?") for i in range(0, len(dna), 3))


@pytest.mark.parametrize("protein", [GFP, LYSOZYME])
@pytest.mark.parametrize("organism", ["ecoli", "yeast"])
def test_optimized_cds_encodes_the_input_protein(protein, organism) -> None:
    dna = CodonOptimizer().optimize(protein, organism)
    assert translate(dna) == protein


@pytest.mark.parametrize("organism", ["ecoli", "yeast"])
def test_cai_is_not_a_constant(organism) -> None:
    """The previous per-residue argmax always picked each family's maximum, whose
    relative adaptiveness is 1.0 by definition — so CAI was identically 1.000 for
    every input and told the user nothing about their sequence."""
    opt = CodonOptimizer()
    scores = {
        name: opt.analyze(opt.optimize(seq, organism), organism)["cai"]
        for name, seq in (("gfp", GFP), ("lysozyme", LYSOZYME), ("polyk", "M" + "K" * 60))
    }
    assert len(set(scores.values())) == len(scores), scores
    assert all(0.0 < v <= 1.0 for v in scores.values())


@pytest.mark.parametrize("protein", [GFP, LYSOZYME])
@pytest.mark.parametrize("organism", ["ecoli", "yeast"])
def test_real_proteins_do_not_trip_every_feasibility_flag(protein, organism) -> None:
    """One codon per amino acid produced heavy local repetition: GFP used 20
    distinct codons over 238 positions and tripped both homopolymer_run and
    potential_hairpin, so the flags fired on essentially every order."""
    opt = CodonOptimizer()
    flags = opt.analyze(opt.optimize(protein, organism), organism)["feasibility_flags"]
    assert flags == []


def test_optimizer_avoids_long_homopolymers() -> None:
    dna = CodonOptimizer().optimize(GFP, "ecoli")
    assert re.search(r"(.)\1{4,}", dna) is None


def test_offline_price_is_marked_as_an_estimate() -> None:
    quote = TwistService._offline_estimate("ATG" * 200)
    assert quote["price_is_estimate"] is True
    assert quote["twist_status"] == "estimated_offline"
    # The basis has to say Twist was never contacted — this is the number that
    # used to be rendered identically to a real quote.
    assert "never contacted" in quote["price_basis"]
    assert quote["order_url"].startswith("https://")


def test_quote_rejects_non_protein_input() -> None:
    """The route used to accept anything: a non-standard residue became an NNN
    codon that still got a CAI and a price."""
    with pytest.raises(ValueError, match="Invalid amino acids"):
        ProcurementEngine().quote("MKAXLIVUGLZ" * 3)


def test_quote_carries_the_order_route(monkeypatch) -> None:
    monkeypatch.setattr("app.services.twist.settings.twist_api_key", "", raising=False)
    result = ProcurementEngine().quote(LYSOZYME)
    assert result["order_url"]
    assert result["price_is_estimate"] is True
    assert result["cai_reference"]
    assert "NNN" not in result["dna_sequence"]
