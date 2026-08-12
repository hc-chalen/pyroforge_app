import {
  Alert,
  Badge,
  Button,
  Code,
  Group,
  Paper,
  SimpleGrid,
  Stack,
  Text,
} from "@mantine/core";
import { useState } from "react";
import { CardHeader } from "@/components/ui/CardHeader";
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
          Codon optimization and Twist Bioscience synthesis quote.
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
        {/* This card ends in a real order. The optimized variant is a model
            shortlist whose top-5 precision is ~74% on held-out proteins and as low
            as 0% on individual ones, so it must not reach a synthesis quote
            without saying so. */}
        {variant === "optimized" && mutationCount > 0 && (
          <Alert color="orange" variant="light" title="You are quoting a predicted variant" p="xs">
            <Text size="xs">
              This sequence carries {mutationCount} unvalidated model-predicted
              mutation{mutationCount === 1 ? "" : "s"}. The optimizer returns a
              ranked shortlist for screening — on held-out proteins roughly a
              quarter of top-5 entries were not stabilizing, and for some proteins
              none were. Order the wild type alongside it as a control, and treat
              the quote as costing an experiment rather than confirming a result.
            </Text>
          </Alert>
        )}
        {displayError && (
          <Alert color="red" title="Error">
            {displayError}
          </Alert>
        )}
        {procurement && (
          <Stack gap="sm">
            <SimpleGrid cols={{ base: 1, sm: 2, md: 4 }}>
              <StatPaper>
                <Text size="xs" c="dimmed">
                  CAI
                </Text>
                <Text fw={700} ff="monospace">
                  {procurement.cai.toFixed(3)}
                </Text>
                {/* A CAI is meaningless without the reference set it was scored
                    against, so the two are never shown apart. */}
                {procurement.cai_reference && (
                  <Text size="xs" c="dimmed">
                    vs {procurement.cai_reference}
                  </Text>
                )}
              </StatPaper>
              <StatPaper>
                <Text size="xs" c="dimmed">
                  GC content
                </Text>
                <Text fw={700} ff="monospace">
                  {/* Backend already returns a percentage (0–100). */}
                  {procurement.gc_content.toFixed(1)}%
                </Text>
              </StatPaper>
              {procurement.price_usd != null && (
                <StatPaper>
                  <Text size="xs" c="dimmed">
                    Price (USD)
                  </Text>
                  <Text fw={700} ff="monospace">
                    ${procurement.price_usd.toFixed(2)}
                  </Text>
                </StatPaper>
              )}
              {procurement.lead_time_days != null && (
                <StatPaper>
                  <Text size="xs" c="dimmed">
                    Lead time
                  </Text>
                  <Text fw={700} ff="monospace">
                    {procurement.lead_time_days} days
                  </Text>
                </StatPaper>
              )}
            </SimpleGrid>
            <Text size="xs" c="dimmed">
              Status: {procurement.twist_status} · Organism:{" "}
              {procurement.organism} · variant: {variant}
            </Text>
            {procurement.feasibility_flags.length > 0 && (
              <Text size="xs">
                Flags: {procurement.feasibility_flags.join(", ")}
              </Text>
            )}
            <Code block ff="monospace" style={{ maxHeight: 120, overflow: "auto" }}>
              {procurement.dna_sequence}
            </Code>
          </Stack>
        )}
        <Group>
          <Button
            onClick={handleRun}
            loading={loading}
            disabled={!activeSequence || loading}
            color="brand"
          >
            Get Quote ({variant})
          </Button>
        </Group>
      </Stack>
    </Paper>
  );
}
