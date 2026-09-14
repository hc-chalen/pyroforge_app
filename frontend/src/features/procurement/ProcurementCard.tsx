import {
  Alert,
  Anchor,
  Badge,
  Button,
  Code,
  CopyButton,
  Group,
  Paper,
  SimpleGrid,
  Stack,
  Text,
} from "@mantine/core";
import { useState } from "react";
import { CardHeader } from "@/components/ui/CardHeader";
import { InfoNote } from "@/components/ui/InfoNote";
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
          Codon optimization, then a hand-off to Twist Bioscience for ordering.
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
        {/* This card ends in a real order. The optimized variant is a model
            shortlist whose top-5 precision is ~74% on held-out proteins and as low
            as 0% on individual ones, so it must not reach an order page
            without saying so. */}
        {variant === "optimized" && mutationCount > 0 && (
          <InfoNote
            tone="warning"
            summary={`This quote is for ${mutationCount} unvalidated predicted mutation${
              mutationCount === 1 ? "" : "s"
            }, it is suggested to order the wild type as a control for comparison.`}
          >
            The optimizer returns a ranked shortlist for screening: on held-out
            proteins roughly a quarter of top-5 entries were not stabilizing, and
            for some proteins none were. The quote buys an experiment, it does not
            confirm a result.
          </InfoNote>
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
            </SimpleGrid>
            <Text size="xs" c="dimmed">
              Organism: {procurement.organism} · variant: {variant}
            </Text>
            {procurement.feasibility_flags.length > 0 && (
              <Text size="xs">
                Flags: {procurement.feasibility_flags.join(", ")}
              </Text>
            )}
            <Code block ff="monospace" style={{ maxHeight: 120, overflow: "auto" }}>
              {procurement.dna_sequence}
            </Code>
            {/* The card's actual endpoint. Twist has no documented deep link that
                pre-fills a sequence, so the honest handoff is: put the CDS on the
                clipboard and open their order page. */}
            <Group gap="xs">
              <CopyButton value={procurement.dna_sequence} timeout={2000}>
                {({ copied, copy }) => (
                  <Button variant="light" onClick={copy}>
                    {copied ? "CDS copied" : "Copy CDS"}
                  </Button>
                )}
              </CopyButton>
              {procurement.order_url && (
                /* Copies on the way out: Twist cannot be deep-linked with a
                   sequence, so arriving there without the CDS on the clipboard
                   means going back for it. */
                <CopyButton value={procurement.dna_sequence} timeout={2000}>
                  {({ copied, copy }) => (
                    <Button
                      component="a"
                      href={procurement.order_url ?? undefined}
                      target="_blank"
                      rel="noopener noreferrer"
                      onClick={copy}
                      variant="outline"
                      color="brand"
                    >
                      {copied
                        ? "CDS copied ↗"
                        : `Order at ${procurement.vendor ?? "Twist"} ↗`}
                    </Button>
                  )}
                </CopyButton>
              )}
              {procurement.pricing_url && (
                <Anchor
                  href={procurement.pricing_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  size="xs"
                >
                  Twist pricing ↗
                </Anchor>
              )}
            </Group>
            <Text size="xs" c="dimmed">
              Ordering opens {procurement.vendor ?? "Twist"} and needs an account
              there. The CDS is copied to your clipboard on the way — paste it
              into their sequence upload.
            </Text>
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
