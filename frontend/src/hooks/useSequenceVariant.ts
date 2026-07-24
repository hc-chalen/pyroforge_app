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

export function useVariantResult<T>(
  analysisId: string,
  variant: SequenceVariant,
  key: "prediction" | "fold" | "registry" | "procurement",
): { data?: T; resultVariant?: SequenceVariant; isStale: boolean } {
  const analysis = useAnalysisStore((s) =>
    s.analyses.find((a) => a.id === analysisId),
  );
  const vr = analysis?.variantResults?.[variant];
  const variantKey = `${key}Variant` as const;
  let data = vr?.[key] as T | undefined;
  let resultVariant = vr?.[variantKey];

  // Legacy fallback for analyses saved before per-variant storage
  if (!data && variant === "original" && analysis) {
    data = analysis[key] as T | undefined;
    if (data) resultVariant = "original";
  }

  const isStale = Boolean(data && resultVariant && resultVariant !== variant);
  return { data, resultVariant, isStale };
}
