import type { Analysis } from "@/store/analysisStore";
import type { FoldResult } from "@/types/api";

/** A fold result with the PDB body removed — see {@link stripPdbForPersist}. */
function withoutPdb(fold: FoldResult): FoldResult {
  return {
    pdb_path: fold.pdb_path,
    sequence_length: fold.sequence_length,
    rmsf: fold.rmsf ?? [],
    pdb: "",
  };
}

/**
 * Drop every PDB body before an analysis is written to localStorage.
 *
 * A 400-residue ESMFold PDB is 300–500 kB of text and localStorage holds ~5 MB
 * in total, so persisting even a handful of folds overflows the quota and makes
 * `setItem` throw — which loses the *entire* persisted state, not just the
 * structures. The PDB text lives in `foldPdbCache` (in-memory, plus a
 * size-capped sessionStorage copy) instead.
 *
 * Both copies must be stripped: the deprecated top-level `fold` *and* every
 * `variantResults[variant].fold`. Stripping only the former (which is what this
 * used to do) left the full PDB in the per-variant copy, so the quota blew
 * anyway — the sanitizer looked correct while doing nothing.
 */
export function stripPdbForPersist(analysis: Analysis): Analysis {
  const variantResults = analysis.variantResults
    ? (Object.fromEntries(
        Object.entries(analysis.variantResults).map(([variant, results]) => [
          variant,
          results?.fold ? { ...results, fold: withoutPdb(results.fold) } : results,
        ]),
      ) as Analysis["variantResults"])
    : undefined;

  return {
    ...analysis,
    fold: analysis.fold ? withoutPdb(analysis.fold) : undefined,
    variantResults,
  };
}
