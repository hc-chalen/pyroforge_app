import type { OptimizationResult, SequenceVariant } from "@/types/api";

export function getOriginalSequence(
  sequence: string,
  optimization?: OptimizationResult,
): string {
  return (optimization?.wild_type_sequence ?? sequence).trim();
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
