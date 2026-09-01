from __future__ import annotations

from app.services.codon import CodonOptimizer
from app.services.esm2 import ESM2Service
from app.services.twist import TwistService


class ProcurementEngine:
    """Codon optimization and Twist Bioscience synthesis quoting."""

    def __init__(self) -> None:
        self.codon = CodonOptimizer()
        self.twist = TwistService()

    def quote(
        self,
        protein_sequence: str,
        organism: str = "ecoli",
    ) -> dict:
        # Validated with the same rules as every other endpoint, server-side. This
        # route used to accept anything: a non-standard residue became an ``NNN``
        # codon that still got a CAI and a price, and a pasted DNA sequence was
        # silently read as protein (A, T, G and C are all valid residue letters).
        # The browser blocked both, so the gap only opened for direct API callers —
        # which on an unauthenticated deployment is anyone.
        seq = ESM2Service.validate_sequence(protein_sequence)
        dna = self.codon.optimize(seq, organism)
        analysis = self.codon.analyze(dna, organism)

        # Everything returned here is computed from the sequence. The vendor part of
        # the payload is a link, not a price — see TwistService for why quoting was
        # removed rather than better-labelled.
        return {
            "protein_sequence": seq,
            "dna_sequence": dna,
            "organism": organism,
            "cai": analysis["cai"],
            # Never surface a CAI without the reference set it was scored against.
            "cai_reference": analysis["cai_reference"],
            "gc_content": analysis["gc_content"],
            "feasibility_flags": analysis["feasibility_flags"],
            **self.twist.order_route(),
        }
