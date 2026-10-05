import {
  Alert,
  Badge,
  Button,
  Code,
  CopyButton,
  Group,
  Paper,
  SimpleGrid,
  Stack,
  Text,
  Tooltip,
} from "@mantine/core";
import { useState } from "react";
import { CardHeader } from "@/components/ui/CardHeader";
import { InfoNote } from "@/components/ui/InfoNote";
import { MetricLabel } from "@/components/ui/MetricLabel";
import { WithheldNotice } from "@/components/ui/Withheld";
import {
  SequenceVariantSelector,
  StaleVariantAlert,
} from "@/components/ui/SequenceVariantSelector";
import { StatPaper } from "@/components/ui/StatPaper";
import { useSequenceVariant, useVariantResult } from "@/hooks/useSequenceVariant";
import { procureQuote } from "@/services/procure";
import { useAnalysisStore } from "@/store/analysisStore";
import { useSettingsStore } from "@/store/settingsStore";
import type { OptimizationResult, ProcureResponse } from "@/types/api";
import { withheldSequence } from "@/utils/answerability";
import { hasOptimizedVariant } from "@/utils/sequenceVariant";

interface ProcurementCardProps {
  analysisId: string;
  sequence: string;
  optimization?: OptimizationResult;
}

export function ProcurementCard({
  analysisId,
  sequence,
  optimization,
}: ProcurementCardProps) {
  const [loading, setLoading] = useState(false);
  const [localError, setLocalError] = useState<string | undefined>();
  const organism = useSettingsStore((s) => s.organism);
  const setProcurement = useAnalysisStore((s) => s.setProcurement);
  const setStatus = useAnalysisStore((s) => s.setStatus);
  const setError = useAnalysisStore((s) => s.setError);

  const { variant, activeSequence, setVariant } = useSequenceVariant(
    analysisId,
    sequence,
    optimization,
  );
  const { data: procurement, resultVariant, staleReason } =
    useVariantResult<ProcureResponse>(
      analysisId,
      variant,
      "procurement",
      activeSequence,
    );
  const mutationCount = optimization?.mutations.length ?? 0;
  const inputRefusal = withheldSequence(activeSequence);

  async function handleRun() {
    if (!activeSequence) return;
    setLoading(true);
    setLocalError(undefined);
    setStatus(analysisId, "running");
    setError(analysisId, undefined);
    try {
      const result = await procureQuote({
        protein_sequence: activeSequence,
        organism,
      });
      setProcurement(analysisId, result, variant, activeSequence);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Procurement quote failed";
      setLocalError(message);
      setError(analysisId, message);
    } finally {
      setLoading(false);
    }
  }

  const displayError = localError;
  const vendor = procurement?.vendor ?? "Twist";

  return (
    <Paper>
      <Stack gap="sm">
        <CardHeader
          eyebrow="Synthesis"
          title="Procurement Pipeline"
          status={
            <Badge color={procurement ? "green" : "gray"}>
              {procurement ? "complete" : "idle"}
            </Badge>
          }
        />
        <Text size="sm" c="dimmed">
          Codon-optimized CDS for Twist ordering.
        </Text>
        {hasOptimizedVariant(optimization) && (
          <SequenceVariantSelector
            sequence={sequence}
            optimization={optimization}
            variant={variant}
            onVariantChange={setVariant}
          />
        )}
        <StaleVariantAlert
          resultVariant={resultVariant}
          activeVariant={variant}
          staleReason={staleReason}
        />
        {inputRefusal?.scope === "step" && (
          <WithheldNotice withheld={inputRefusal} />
        )}
        {variant === "optimized" && mutationCount > 0 && (
          <InfoNote
            tone="warning"
            summary={`Quote is for ${mutationCount} unvalidated predicted mutation${
              mutationCount === 1 ? "" : "s"
            }. Order the wild type as a control.`}
          >
            The optimizer returns a ranked shortlist for screening, not a
            confirmed design.
          </InfoNote>
        )}
        {displayError && (
          <Alert color="red" title="Error">
            {displayError}
          </Alert>
        )}
        {procurement && (
          <Stack gap="sm">
            <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="xs">
              <StatPaper>
                <MetricLabel
                  label="CAI"
                  method="derived"
                  tooltip={
                    procurement.cai_reference
                      ? `Scored vs ${procurement.cai_reference}`
                      : "Codon Adaptation Index for the expression host."
                  }
                />
                <Text fw={700} size="lg" ff="monospace" mt={4}>
                  {procurement.cai.toFixed(3)}
                </Text>
              </StatPaper>
              <StatPaper>
                <MetricLabel
                  label="GC content"
                  method="derived"
                  tooltip="GC fraction of the codon-optimized CDS."
                />
                <Text fw={700} size="lg" ff="monospace" mt={4}>
                  {procurement.gc_content.toFixed(1)}%
                </Text>
              </StatPaper>
            </SimpleGrid>
            {procurement.feasibility_flags.length > 0 && (
              <Text size="xs" c="dimmed">
                Flags: {procurement.feasibility_flags.join(", ")}
              </Text>
            )}
            <Code block ff="monospace" style={{ maxHeight: 88, overflow: "auto" }}>
              {procurement.dna_sequence}
            </Code>
            <Group gap="xs" wrap="wrap">
              <CopyButton value={procurement.dna_sequence} timeout={2000}>
                {({ copied, copy }) => (
                  <Button variant="light" onClick={copy}>
                    {copied ? "CDS copied" : "Copy CDS"}
                  </Button>
                )}
              </CopyButton>
              {procurement.order_url && (
                <CopyButton value={procurement.dna_sequence} timeout={2000}>
                  {({ copied, copy }) => (
                    <Tooltip
                      label={`Opens ${vendor} with the CDS on the clipboard. Paste it into their upload.`}
                      multiline
                      maw={260}
                    >
                      <Button
                        component="a"
                        href={procurement.order_url ?? undefined}
                        target="_blank"
                        rel="noopener noreferrer"
                        onClick={copy}
                        variant="outline"
                        color="brand"
                      >
                        {copied ? "CDS copied ↗" : `Order at ${vendor} ↗`}
                      </Button>
                    </Tooltip>
                  )}
                </CopyButton>
              )}
            </Group>
          </Stack>
        )}
        <Group>
          <Button
            onClick={handleRun}
            loading={loading}
            disabled={!activeSequence || loading || inputRefusal != null}
            color="brand"
          >
            Get Quote ({variant})
          </Button>
        </Group>
      </Stack>
    </Paper>
  );
}
