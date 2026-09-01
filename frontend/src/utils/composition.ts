/**
 * Composition descriptors for pyrolysis reasoning.
 *
 * Everything here is arithmetic on the sequence itself — atom counting and
 * residue tallies. None of it is a trained prediction, and none of it has been
 * validated against a pyrolysis dataset; the UI must keep labelling these as
 * descriptors.
 */

/** Residue (peptide-bonded, i.e. amino acid minus water) atom counts. */
const RESIDUE_ATOMS: Record<string, [C: number, H: number, N: number, O: number, S: number]> = {
  A: [3, 5, 1, 1, 0],
  R: [6, 12, 4, 1, 0],
  N: [4, 6, 2, 2, 0],
  D: [4, 5, 1, 3, 0],
  C: [3, 5, 1, 1, 1],
  E: [5, 7, 1, 3, 0],
  Q: [5, 8, 2, 2, 0],
  G: [2, 3, 1, 1, 0],
  H: [6, 7, 3, 1, 0],
  I: [6, 11, 1, 1, 0],
  L: [6, 11, 1, 1, 0],
  K: [6, 12, 2, 1, 0],
  M: [5, 9, 1, 1, 1],
  F: [9, 9, 1, 1, 0],
  P: [5, 7, 1, 1, 0],
  S: [3, 5, 1, 2, 0],
  T: [4, 7, 1, 2, 0],
  W: [11, 10, 2, 1, 0],
  Y: [9, 9, 1, 2, 0],
  V: [5, 9, 1, 1, 0],
};

/** Amide/amine-bearing and heteroaromatic-N residues. */
const N_RICH = new Set(["R", "K", "H", "Q", "N", "W"]);
/** Aromatic rings (+ His imidazole) — the ring systems that survive to char. */
const CHAR_PRECURSOR = new Set(["F", "Y", "W", "H"]);

// REMOVED: an `hcnRisk` classification with thresholds at N/C 0.30 and 0.34.
//
// The atom counts below are arithmetic and checkable. Those two numbers were
// neither: they had no source, and they drove a user-visible "HCN risk: elevated /
// high" badge — a safety-adjacent claim (HCN is hydrogen cyanide) resting on a
// figure nobody could cite. They also fired on ordinary proteins. Average protein
// N/C is ~0.275, but hen lysozyme is 0.315 and tripped "elevated" with nothing
// unusual about it.
//
// The ratios are still reported. Turning one of them into a risk grade is a
// judgement this repo cannot support, so it is left to the reader — pin the
// thresholds to a pyrolysis reference and this can come back.

export interface CompositionDescriptors {
  /** Residues actually counted (unknown letters are skipped). */
  counted: number;
  ratios: { nc: number; oc: number; hc: number; sc: number; pc: number };
  nRichFraction: number;
  charPrecursorFraction: number;
}

export function computeCompositionDescriptors(
  sequence: string,
): CompositionDescriptors | null {
  const residues = sequence.toUpperCase().replace(/[^A-Z]/g, "");
  let c = 0;
  let h = 0;
  let n = 0;
  let o = 0;
  let s = 0;
  let counted = 0;
  let nRich = 0;
  let charPrecursor = 0;

  for (const aa of residues) {
    const atoms = RESIDUE_ATOMS[aa];
    if (!atoms) continue;
    c += atoms[0];
    h += atoms[1];
    n += atoms[2];
    o += atoms[3];
    s += atoms[4];
    counted += 1;
    if (N_RICH.has(aa)) nRich += 1;
    if (CHAR_PRECURSOR.has(aa)) charPrecursor += 1;
  }

  if (counted === 0 || c === 0) return null;

  // The chain carries one extra water across its two termini.
  h += 2;
  o += 1;

  return {
    counted,
    // Canonical residues contain no phosphorus, so P/C is structurally 0.
    ratios: { nc: n / c, oc: o / c, hc: h / c, sc: s / c, pc: 0 },
    nRichFraction: nRich / counted,
    charPrecursorFraction: charPrecursor / counted,
  };
}
