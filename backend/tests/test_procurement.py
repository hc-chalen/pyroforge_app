"""Procurement contract: every number on the card is computed from the sequence,
and the codon optimizer's reported metrics actually vary with the input.

The card used to carry a price and a lead time. Both were invented — a hard-coded
per-bp band table and `10 if length < 1500 else 15` — so they were removed rather
than re-labelled, and `test_quote_states_no_price_or_lead_time` is what keeps them
from coming back.
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


def test_order_route_is_a_link_not_a_quote() -> None:
    route = TwistService().order_route()
    assert route["order_url"].startswith("https://")
    assert route["pricing_url"].startswith("https://")
    assert route["vendor"] == "Twist Bioscience"
    # Nothing commercial is asserted by this app.
    assert "price_usd" not in route
    assert "lead_time_days" not in route


@pytest.mark.parametrize("protein", [GFP, LYSOZYME])
def test_quote_states_no_price_or_lead_time(protein) -> None:
    """The regression guard. A price band table and `10 if length < 1500 else 15`
    used to ship as `price_usd` and `lead_time_days`; the lead time carried no
    estimate flag at all, so a made-up figure sat beside a labelled-estimate price
    and read as fact. Neither field may return."""
    result = ProcurementEngine().quote(protein)

    for invented in (
        "price_usd",
        "lead_time_days",
        "price_is_estimate",
        "price_basis",
        "synthesis_product",
        "twist_status",
    ):
        assert invented not in result, f"{invented} is not something this app knows"
    # What it does know still travels.
    assert result["dna_sequence"]
    assert result["cai_reference"]
    assert result["order_url"]


def test_quote_rejects_non_protein_input() -> None:
    """The route used to accept anything: a non-standard residue became an NNN
    codon that still got a CAI and a price."""
    with pytest.raises(ValueError, match="Invalid amino acids"):
        ProcurementEngine().quote("MKAXLIVUGLZ" * 3)


def test_quote_carries_the_order_route() -> None:
    result = ProcurementEngine().quote(LYSOZYME)
    assert result["order_url"]
    assert result["vendor"] == "Twist Bioscience"
    assert result["cai_reference"]
    assert "NNN" not in result["dna_sequence"]
