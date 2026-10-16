/**
 * DNA → protein translation for user input.
 *
 * Mirrors backend `app.data.igem_parts.translate`: longest ATG→stop ORF across
 * the three forward frames. Pasted CDS often includes flanks, so frame 0 from
 * base 0 is usually wrong.
 */

const CODON_TO_AA: Record<string, string> = {
  TTT: "F", TTC: "F", TTA: "L", TTG: "L",
  CTT: "L", CTC: "L", CTA: "L", CTG: "L",
  ATT: "I", ATC: "I", ATA: "I", ATG: "M",
  GTT: "V", GTC: "V", GTA: "V", GTG: "V",
  TCT: "S", TCC: "S", TCA: "S", TCG: "S", AGT: "S", AGC: "S",
  CCT: "P", CCC: "P", CCA: "P", CCG: "P",
  ACT: "T", ACC: "T", ACA: "T", ACG: "T",
  GCT: "A", GCC: "A", GCA: "A", GCG: "A",
  TAT: "Y", TAC: "Y", TAA: "*", TAG: "*", TGA: "*",
  CAT: "H", CAC: "H", CAA: "Q", CAG: "Q",
  AAT: "N", AAC: "N", AAA: "K", AAG: "K",
  GAT: "D", GAC: "D", GAA: "E", GAG: "E",
  TGT: "C", TGC: "C", TGG: "W",
  CGT: "R", CGC: "R", CGA: "R", CGG: "R", AGA: "R", AGG: "R",
  GGT: "G", GGC: "G", GGA: "G", GGG: "G",
};

const STOP = "*";

/** Strip whitespace; upper-case; U→T. Does not remove non-ACGT letters. */
export function normalizeDna(raw: string): string {
  return raw.toUpperCase().replace(/\s+/g, "").replace(/U/g, "T");
}

/** Longest forward-strand ORF starting at ATG. Empty string if none found. */
export function translateDna(dna: string): string {
  const seq = normalizeDna(dna);
  if (!seq) return "";

  let best = "";
  for (let frame = 0; frame < 3; frame++) {
    const protein: string[] = [];
    let started = false;
    for (let i = frame; i < seq.length - 2; i += 3) {
      const aa = CODON_TO_AA[seq.slice(i, i + 3)];
      if (aa == null) {
        protein.length = 0;
        started = false;
        continue;
      }
      if (aa === STOP) {
        const joined = protein.join("");
        if (joined.length > best.length) best = joined;
        protein.length = 0;
        started = false;
        continue;
      }
      if (!started) {
        if (aa !== "M") continue;
        started = true;
      }
      protein.push(aa);
    }
    const joined = protein.join("");
    if (joined.length > best.length) best = joined;
  }
  return best;
}
