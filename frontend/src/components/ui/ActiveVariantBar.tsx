import { Paper, Stack, Text } from "@mantine/core";
import {
  SequenceVariantSelector,
} from "@/components/ui/SequenceVariantSelector";
import { useSequenceVariant } from "@/hooks/useSequenceVariant";
import type { OptimizationResult } from "@/types/api";
import { hasOptimizedVariant } from "@/utils/sequenceVariant";

interface ActiveVariantBarProps {
  analysisId: string;
  sequence: string;
  optimization?: OptimizationResult;
}

export function ActiveVariantBar({
  analysisId,
  sequence,
  optimization,
}: ActiveVariantBarProps) {
  const { variant, setVariant } = useSequenceVariant(
    analysisId,
    sequence,
    optimization,
  );

  if (!hasOptimizedVariant(optimization)) return null;

  return (
    <Paper p="md" bg="var(--pyro-card-nested-bg)">
      <Stack gap="sm">
        <Text fw={600} style={{ fontSize: "0.875rem" }}>
          Active sequence variant
        </Text>
        <Text size="sm" c="dimmed">
          Choose which protein sequence downstream tools (visualization, registry,
          procurement) operate on.
        </Text>
        <SequenceVariantSelector
          sequence={sequence}
          optimization={optimization}
          variant={variant}
          onVariantChange={setVariant}
        />
      </Stack>
    </Paper>
  );
}
