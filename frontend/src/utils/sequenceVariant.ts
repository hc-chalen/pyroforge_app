import type { OptimizationResult, SequenceVariant } from "@/types/api";

/**
 * The "original" variant is the sequence currently in the input box.
 *
 * It previously preferred ``optimization.wild_type_sequence``, which meant that once
 * an optimization existed, edits to the textarea were silently ignored by every tool
 * — the UI showed the new sequence while the backend kept receiving the old one. The
 * optimization's own wild type is now only a fallback for an empty input, and a
 * mismatch between the two is surfaced via {@link isOptimizationStale}.
 */
export function getOriginalSequence(
  sequence: string,
  optimization?: OptimizationResult,
): string {
  return (sequence.trim() || optimization?.wild_type_sequence || "").trim();
}

/** True when the input has been edited since this optimization was computed. */
export function isOptimizationStale(
  sequence: string,
  optimization?: OptimizationResult,
): boolean {
  const wt = optimization?.wild_type_sequence?.trim();
  const current = sequence.trim();
  return Boolean(wt && current && wt !== current);
}

export function getOptimizedSequence(
  optimization?: OptimizationResult,
): string | undefined {
  const opt = optimization?.optimized_sequence?.trim();
  return opt || undefined;
}

export function resolveActiveSequence(
  sequence: string,
  variant: SequenceVariant,
  optimization?: OptimizationResult,
): string {
  if (variant === "optimized") {
    return getOptimizedSequence(optimization) ?? getOriginalSequence(sequence, optimization);
  }
  return getOriginalSequence(sequence, optimization);
}

export function hasOptimizedVariant(optimization?: OptimizationResult): boolean {
  return Boolean(getOptimizedSequence(optimization));
}

export function formatSimilarityPercent(similarity: number): string {
  return `${(similarity * 100).toFixed(1)}%`;
}

export function similarityColor(similarity: number): string {
  if (similarity >= 0.95) return "green";
  if (similarity >= 0.85) return "teal";
  if (similarity >= 0.7) return "yellow";
  return "orange";
}
