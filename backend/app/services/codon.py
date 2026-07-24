from __future__ import annotations

import math
import re
from collections import Counter

# Relative adaptiveness (CAI weights) for E. coli K-12 and S. cerevisiae
CODON_TABLES: dict[str, dict[str, float]] = {
    "ecoli": {
        "TTT": 0.58, "TTC": 1.00, "TTA": 0.14, "TTG": 0.13,
        "TCT": 0.17, "TCC": 0.15, "TCA": 0.14, "TCG": 0.15,
        "TAT": 0.59, "TAC": 1.00, "TAA": 0.61, "TAG": 0.02,
        "TGT": 0.46, "TGC": 1.00, "TGA": 0.52, "TGG": 1.00,
        "CTT": 0.12, "CTC": 0.10, "CTA": 0.04, "CTG": 1.00,
        "CCT": 0.18, "CCC": 0.13, "CCA": 0.20, "CCG": 0.49,
        "CAT": 0.57, "CAC": 1.00, "CAA": 0.34, "CAG": 1.00,
        "CGT": 0.36, "CGC": 0.36, "CGA": 0.07, "CGG": 0.11,
        "ATT": 0.43, "ATC": 1.00, "ATA": 0.03, "ATG": 1.00,
        "ACT": 0.27, "ACC": 1.00, "ACA": 0.17, "ACG": 0.41,
        "AAT": 0.49, "AAC": 1.00, "AAA": 0.74, "AAG": 1.00,
        "AGT": 0.16, "AGC": 0.26, "AGA": 0.07, "AGG": 0.04,
        "GTT": 0.39, "GTC": 0.43, "GTA": 0.17, "GTG": 1.00,
        "GCT": 0.33, "GCC": 1.00, "GCA": 0.21, "GCG": 0.34,
        "GAT": 0.63, "GAC": 1.00, "GAA": 0.68, "GAG": 1.00,
        "GGT": 0.34, "GGC": 0.41, "GGA": 0.10, "GGG": 0.10,
    },
    "yeast": {
        "TTT": 0.56, "TTC": 1.00, "TTA": 0.28, "TTG": 0.27,
        "TCT": 0.26, "TCC": 0.18, "TCA": 0.20, "TCG": 0.10,
        "TAT": 0.59, "TAC": 1.00, "TAA": 1.00, "TAG": 0.09,
        "TGT": 0.46, "TGC": 1.00, "TGA": 0.30, "TGG": 1.00,
        "CTT": 0.13, "CTC": 0.06, "CTA": 0.14, "CTG": 1.00,
        "CCT": 0.31, "CCC": 0.12, "CCA": 0.40, "CCG": 0.11,
        "CAT": 0.65, "CAC": 1.00, "CAA": 0.65, "CAG": 1.00,
        "CGT": 0.13, "CGC": 0.06, "CGA": 0.04, "CGG": 0.04,
        "ATT": 0.46, "ATC": 1.00, "ATA": 0.30, "ATG": 1.00,
        "ACT": 0.35, "ACC": 1.00, "ACA": 0.30, "ACG": 0.17,
        "AAT": 0.59, "AAC": 1.00, "AAA": 0.70, "AAG": 1.00,
        "AGT": 0.16, "AGC": 0.11, "AGA": 0.21, "AGG": 0.04,
        "GTT": 0.39, "GTC": 0.21, "GTA": 0.20, "GTG": 1.00,
        "GCT": 0.34, "GCC": 1.00, "GCA": 0.40, "GCG": 0.11,
        "GAT": 0.65, "GAC": 1.00, "GAA": 0.71, "GAG": 1.00,
        "GGT": 0.47, "GGC": 0.19, "GGA": 0.22, "GGG": 0.12,
    },
}

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


class CodonOptimizer:
    def optimize(self, protein: str, organism: str = "ecoli") -> str:
        table = CODON_TABLES.get(organism, CODON_TABLES["ecoli"])
        codons = []
        for aa in protein:
            options = GENETIC_CODE.get(aa, ["NNN"])
            best = max(options, key=lambda c: table.get(c, 0.01))
            codons.append(best)
        return "".join(codons)

    def analyze(self, dna: str) -> dict:
        dna = dna.upper()
        gc = (dna.count("G") + dna.count("C")) / len(dna) * 100 if dna else 0.0
        cai = self._compute_cai(dna)
        flags = []
        if gc < 25 or gc > 65:
            flags.append("gc_content_extreme")
        if re.search(r"(.)\1{5,}", dna):
            flags.append("homopolymer_run")
        if self._has_hairpin(dna):
            flags.append("potential_hairpin")
        return {"cai": cai, "gc_content": round(gc, 2), "feasibility_flags": flags}

    def _compute_cai(self, dna: str) -> float:
        table = CODON_TABLES["ecoli"]
        weights = []
        for i in range(0, len(dna) - 2, 3):
            codon = dna[i : i + 3]
            weights.append(table.get(codon, 0.01))
        if not weights:
            return 0.0
        log_sum = sum(math.log(w) for w in weights) / len(weights)
        return round(math.exp(log_sum), 4)

    @staticmethod
    def _has_hairpin(dna: str, window: int = 20) -> bool:
        complement = str.maketrans("ACGT", "TGCA")
        for i in range(len(dna) - window):
            segment = dna[i : i + window // 2]
            rev_comp = segment.translate(complement)[::-1]
            if rev_comp in dna[i + window // 2 : i + window]:
                return True
        return False
