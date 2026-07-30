import { Alert, Group, SegmentedControl, Text } from "@mantine/core";
import type { SequenceVariant } from "@/types/api";
import { resolveActiveSequence } from "@/utils/sequenceVariant";
import type { OptimizationResult } from "@/types/api";

interface SequenceVariantSelectorProps {
  sequence: string;
  optimization?: OptimizationResult;
  variant: SequenceVariant;
  onVariantChange: (variant: SequenceVariant) => void;
  /** Show compact label only (no sequence preview) */
  compact?: boolean;
}

export function SequenceVariantSelector({
  sequence,
  optimization,
  variant,
  onVariantChange,
  compact = false,
}: SequenceVariantSelectorProps) {
  const hasOptimized = Boolean(optimization?.optimized_sequence?.trim());
  const activeSequence = resolveActiveSequence(sequence, variant, optimization);

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
      {!compact && (
        <Text size="xs" c="dimmed" ff="monospace" lineClamp={1} style={{ flex: 1, minWidth: 0 }}>
          {activeSequence.length} aa · {activeSequence.slice(0, 48)}
          {activeSequence.length > 48 ? "…" : ""}
        </Text>
      )}
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
      <Alert color="yellow" title="Results are out of date">
        The sequence has been edited since these results were computed, so they
        describe the <Text span fw={600}>previous</Text> sequence. Re-run to refresh.
      </Alert>
    );
  }
  if (!resultVariant || resultVariant === activeVariant) return null;

  return (
    <Alert color="yellow" title="Results from another variant">
      These results were computed for the{" "}
      <Text span fw={600}>
        {resultVariant === "optimized" ? "optimized" : "original"}
      </Text>{" "}
      sequence. Switch variant or re-run to refresh.
    </Alert>
  );
}
