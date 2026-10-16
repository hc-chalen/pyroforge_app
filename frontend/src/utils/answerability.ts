/**
 * Honest-refusal policy for the UI.
 *
 * Every number this app shows comes from a model with a known domain of
 * validity. When an input falls outside that domain the honest answer is "we
 * don't know", not a number carrying a paragraph of apology next to it — a
 * caveat printed beside a confident-looking figure is read as the figure.
 *
 * So the rule is: **withhold first, reveal on request.** Each check below
 * returns `null` when the question is answerable, or a {@link Withheld} saying
 * in one line what is being refused and why. Where a number does exist but is
 * an extrapolation, `extrapolated` is set and the UI offers an explicit
 * "show anyway" — the user can still see it, but never by accident.
 */

import type { InputFormat, PredictResponse } from "@/types/api";
import { normalizeDna, translateDna } from "@/utils/translateDna";

/** Shared input ceiling. Past this length folding, Tm, and the mutation scan
 *  are out of regime, so the sequence is refused at the input rather than
 *  producing a number that looks precise. */
export const SEQ_MIN_LENGTH = 10;
export const SEQ_MAX_LENGTH = 400;
/** ESMFold ceiling, and the site-saturation scan ceiling (settings.optimize_max_length). */
export const FOLD_MAX_LENGTH = SEQ_MAX_LENGTH;
export const OPTIMIZE_MAX_LENGTH = SEQ_MAX_LENGTH;

/** DNA paste ceiling (nt). Allows modest flanks around a ≤400-aa CDS. */
export const DNA_MIN_LENGTH = 30;
export const DNA_MAX_LENGTH = 3_600;

const STANDARD_AA = new Set("ACDEFGHIKLMNPQRSTVWY");
const DNA_BASES = new Set("ACGT");

export interface Withheld {
  /** Short headline for the refusal, e.g. "Outside training range". */
  headline: string;
  /** One sentence: what can't be answered, and why. */
  reason: string;
  /**
   * "input" — the sequence itself is unusable, so every card is blocked and only
   * the input card states it (repeating it six times is the noise this UI is
   * trying to lose). "step" — this particular step can't run on an otherwise
   * valid sequence, so the card must say so itself.
   */
  scope?: "input" | "step";
  /**
   * True when a number was produced but lies outside the model's training
   * domain. The UI keeps it hidden behind an explicit reveal rather than
   * dropping it, so the extrapolation stays available but never implicit.
   */
  extrapolated?: boolean;
}

/** Protein alphabet/length gate used by every downstream card. */
export function withheldProteinSequence(sequence: string): Withheld | null {
  const seq = sequence.trim().toUpperCase();
  if (!seq) return null; // empty input is "nothing asked yet", not a refusal

  if (seq.startsWith(">")) {
    return {
      headline: "FASTA header found",
      reason: "Paste the residues only — delete the “>” description line.",
      scope: "input",
    };
  }
  const invalid = [...new Set([...seq].filter((c) => !STANDARD_AA.has(c)))];
  if (invalid.length > 0) {
    return {
      headline: "Not a protein sequence",
      reason: `Contains ${invalid
        .slice(0, 6)
        .map((c) => (c.trim() === "" ? "whitespace" : `“${c}”`))
        .join(", ")}${invalid.length > 6 ? " and others" : ""}. Only the 20 standard amino acids are accepted — gaps and ambiguity codes must be removed.`,
      scope: "input",
    };
  }
  if (seq.length < SEQ_MIN_LENGTH) {
    return {
      headline: "Sequence too short",
      reason: `${seq.length} residues; the models require at least ${SEQ_MIN_LENGTH}.`,
      scope: "input",
    };
  }
  if (seq.length > SEQ_MAX_LENGTH) {
    return {
      headline: "Sequence too long",
      reason: `${seq.length} residues; results are only reliable up to ${SEQ_MAX_LENGTH} amino acids.`,
      scope: "input",
    };
  }
  return null;
}

/** DNA paste gate: alphabet, length, then translate + protein checks. */
export function withheldDnaSequence(dna: string): Withheld | null {
  const raw = dna.trim();
  if (!raw) return null;

  if (raw.startsWith(">")) {
    return {
      headline: "FASTA header found",
      reason: "Paste the nucleotides only — delete the “>” description line.",
      scope: "input",
    };
  }

  const seq = normalizeDna(raw);
  const invalid = [...new Set([...seq].filter((c) => !DNA_BASES.has(c)))];
  if (invalid.length > 0) {
    return {
      headline: "Not a DNA sequence",
      reason: `Contains ${invalid
        .slice(0, 6)
        .map((c) => `“${c}”`)
        .join(", ")}${invalid.length > 6 ? " and others" : ""}. Only A, C, G, T/U are accepted.`,
      scope: "input",
    };
  }
  if (seq.length < DNA_MIN_LENGTH) {
    return {
      headline: "DNA too short",
      reason: `${seq.length} nt; paste at least ${DNA_MIN_LENGTH} bases (enough for a short CDS).`,
      scope: "input",
    };
  }
  if (seq.length > DNA_MAX_LENGTH) {
    return {
      headline: "DNA too long",
      reason: `${seq.length} nt; paste at most ${DNA_MAX_LENGTH} bases.`,
      scope: "input",
    };
  }

  const protein = translateDna(seq);
  if (!protein) {
    return {
      headline: "No coding sequence found",
      reason:
        "No ATG→stop ORF was found on the forward strand. Paste a protein-coding DNA sequence.",
      scope: "input",
    };
  }
  return withheldProteinSequence(protein);
}

/**
 * Input gate for the Sequence Input card.
 * Downstream cards should keep calling this with protein only (default `"aa"`).
 */
export function withheldSequence(
  sequence: string,
  format: InputFormat = "aa",
): Withheld | null {
  return format === "dna"
    ? withheldDnaSequence(sequence)
    : withheldProteinSequence(sequence);
}

/** Folding gate: ESMFold's own length ceiling. */
export function withheldFold(sequence: string): Withheld | null {
  const base = withheldSequence(sequence);
  if (base) return base;
  const len = sequence.trim().length;
  if (len > FOLD_MAX_LENGTH) {
    return {
      headline: "Too long to fold",
      reason: `${len} residues; ESMFold accepts at most ${FOLD_MAX_LENGTH}.`,
      scope: "step",
    };
  }
  return null;
}

/** Optimization gate: the scan is O(L × 20) ESM-2 passes, so it is capped too. */
export function withheldOptimize(sequence: string): Withheld | null {
  const base = withheldSequence(sequence);
  if (base) return base;
  const len = sequence.trim().length;
  if (len > OPTIMIZE_MAX_LENGTH) {
    return {
      headline: "Too long to scan",
      reason: `${len} residues; the mutation scan accepts at most ${OPTIMIZE_MAX_LENGTH}.`,
      scope: "step",
    };
  }
  return null;
}

/** Absolute Tm: withheld when no checkpoint exists, or the length is out of range. */
export function withheldTm(prediction?: PredictResponse): Withheld | null {
  if (!prediction) return null;
  if (prediction.tm_confidence === "unavailable" || prediction.tm == null) {
    return {
      headline: "No Tm model",
      reason: "No model_tm checkpoint is installed, so absolute Tm cannot be predicted.",
      scope: "step",
    };
  }
  if (prediction.tm_confidence === "low") {
    return {
      headline: "Outside Tm training range",
      reason:
        prediction.tm_confidence_reason ??
        "This sequence is outside the length window model_tm was trained on.",
      scope: "step",
      extrapolated: true,
    };
  }
  return null;
}

/**
 * ΔG and ΔΔG share one domain: the Tsuboyama small-designed-domain assay. A
 * full-length enzyme is extrapolation for both, so both are withheld together.
 */
export function withheldStability(prediction?: PredictResponse): Withheld | null {
  if (!prediction) return null;
  if (prediction.dg_confidence === "low") {
    return {
      headline: "Outside ΔG/ΔΔG training range",
      reason:
        prediction.dg_confidence_reason ??
        "This sequence is longer than the small designed domains ΔG and ΔΔG were trained on.",
      scope: "step",
      extrapolated: true,
    };
  }
  return null;
}
