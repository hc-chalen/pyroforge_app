from __future__ import annotations

from app.services.codon import CodonOptimizer
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
        seq = protein_sequence.strip().upper()
        dna = self.codon.optimize(seq, organism)
        analysis = self.codon.analyze(dna, organism)
        twist_quote = self.twist.get_quote(dna)

        return {
            "protein_sequence": seq,
            "dna_sequence": dna,
            "organism": organism,
            "cai": analysis["cai"],
            # Never surface a CAI without the reference set it was scored against.
            "cai_reference": analysis["cai_reference"],
            "gc_content": analysis["gc_content"],
            "feasibility_flags": analysis["feasibility_flags"],
            **twist_quote,
        }
