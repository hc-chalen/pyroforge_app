import { useAnalysisStore } from "@/store/analysisStore";
import type { OptimizationResult, SequenceVariant } from "@/types/api";
import { resolveActiveSequence } from "@/utils/sequenceVariant";

export function useSequenceVariant(
  analysisId: string,
  sequence: string,
  optimization?: OptimizationResult,
) {
  const variant =
    useAnalysisStore(
      (s) => s.analyses.find((a) => a.id === analysisId)?.sequenceVariant,
    ) ?? "original";

  const setSequenceVariant = useAnalysisStore((s) => s.setSequenceVariant);

  const activeSequence = resolveActiveSequence(sequence, variant, optimization);

  function setVariant(next: SequenceVariant) {
    setSequenceVariant(analysisId, next);
  }

  return { variant, activeSequence, setVariant };
}

export type StaleReason = "variant" | "sequence";

/**
 * Fetch the result stored for ``variant``, and report whether it is stale.
 *
 * Two independent staleness causes:
 *  - ``"variant"`` — the stored result was computed for the other variant. (Results
 *    are keyed by variant, so this can only arise from legacy persisted state.)
 *  - ``"sequence"`` — the input sequence has been edited since the result was
 *    computed. Without this check, editing the textarea leaves the previous numbers
 *    on screen as though they described the new sequence.
 */
export function useVariantResult<T>(
  analysisId: string,
  variant: SequenceVariant,
  key: "prediction" | "fold" | "registry" | "procurement",
  activeSequence?: string,
): {
  data?: T;
  resultVariant?: SequenceVariant;
  isStale: boolean;
  staleReason?: StaleReason;
} {
  const analysis = useAnalysisStore((s) =>
    s.analyses.find((a) => a.id === analysisId),
  );
  const vr = analysis?.variantResults?.[variant];
  const variantKey = `${key}Variant` as const;
  const sequenceKey = `${key}Sequence` as const;
  let data = vr?.[key] as T | undefined;
  let resultVariant = vr?.[variantKey];
  const resultSequence = vr?.[sequenceKey];

  // Legacy fallback for analyses saved before per-variant storage
  if (!data && variant === "original" && analysis) {
    data = analysis[key] as T | undefined;
    if (data) resultVariant = "original";
  }

  let staleReason: StaleReason | undefined;
  if (data) {
    if (resultVariant && resultVariant !== variant) {
      staleReason = "variant";
    } else if (
      activeSequence != null &&
      resultSequence != null &&
      resultSequence !== activeSequence
    ) {
      staleReason = "sequence";
    }
  }

  return { data, resultVariant, isStale: Boolean(staleReason), staleReason };
}
