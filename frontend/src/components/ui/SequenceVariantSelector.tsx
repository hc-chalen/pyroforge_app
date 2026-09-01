import { Group, SegmentedControl } from "@mantine/core";
import { InfoNote } from "@/components/ui/InfoNote";
import type { OptimizationResult, SequenceVariant } from "@/types/api";

interface SequenceVariantSelectorProps {
  sequence: string;
  optimization?: OptimizationResult;
  variant: SequenceVariant;
  onVariantChange: (variant: SequenceVariant) => void;
}

export function SequenceVariantSelector({
  optimization,
  variant,
  onVariantChange,
}: SequenceVariantSelectorProps) {
  const hasOptimized = Boolean(optimization?.optimized_sequence?.trim());

  if (!hasOptimized) return null;

  return (
    <Group gap="sm" align="flex-start" wrap="wrap" w="100%">
      <SegmentedControl
        value={variant}
        onChange={(v) => onVariantChange(v as SequenceVariant)}
        data={[
          { label: "Original", value: "original" },
          { label: "Optimized", value: "optimized" },
        ]}
        size="xs"
      />
    </Group>
  );
}

interface StaleVariantAlertProps {
  resultVariant?: SequenceVariant;
  activeVariant: SequenceVariant;
  /** Why the result is stale. When omitted, only the variant mismatch is reported. */
  staleReason?: "variant" | "sequence";
  onRerun?: () => void;
}

export function StaleVariantAlert({
  resultVariant,
  activeVariant,
  staleReason,
}: StaleVariantAlertProps) {
  if (staleReason === "sequence") {
    return (
      <InfoNote
        tone="warning"
        summary="The sequence was edited after these results — they describe the previous one. Re-run."
      />
    );
  }
  if (!resultVariant || resultVariant === activeVariant) return null;

  return (
    <InfoNote
      tone="warning"
      summary={`These results are from the ${resultVariant} sequence. Switch variant or re-run.`}
    />
  );
}
