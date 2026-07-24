import {
  Alert,
  Anchor,
  Badge,
  Button,
  Group,
  Paper,
  Progress,
  Stack,
  Table,
  Text,
} from "@mantine/core";
import { useState } from "react";
import { CardHeader } from "@/components/ui/CardHeader";
import {
  SequenceVariantSelector,
  StaleVariantAlert,
} from "@/components/ui/SequenceVariantSelector";
import { useAsyncJob } from "@/hooks/useAsyncJob";
import { useSequenceVariant, useVariantResult } from "@/hooks/useSequenceVariant";
import { searchRegistry } from "@/services/registry";
import { useAnalysisStore } from "@/store/analysisStore";
import { useSettingsStore } from "@/store/settingsStore";
import type { OptimizationResult, RegistrySearchResponse } from "@/types/api";
import { hasOptimizedVariant } from "@/utils/sequenceVariant";

interface RegistryCardProps {
  analysisId: string;
  sequence: string;
  optimization?: OptimizationResult;
}

export function RegistryCard({
  analysisId,
  sequence,
  optimization,
}: RegistryCardProps) {
  const [localError, setLocalError] = useState<string | undefined>();
  const { loading, progress, runJob } = useAsyncJob();
  const maxHits = useSettingsStore((s) => s.maxHits);
  const setRegistry = useAnalysisStore((s) => s.setRegistry);
  const setStatus = useAnalysisStore((s) => s.setStatus);
  const setError = useAnalysisStore((s) => s.setError);

  const { variant, activeSequence, setVariant } = useSequenceVariant(
    analysisId,
    sequence,
    optimization,
  );
  const { data: registry, resultVariant } = useVariantResult<RegistrySearchResponse>(
    analysisId,
    variant,
    "registry",
  );

  async function handleRun() {
    if (!activeSequence) return;
    setLocalError(undefined);
    setStatus(analysisId, "running");
    setError(analysisId, undefined);
    try {
      const { job_id } = await searchRegistry({
        sequence: activeSequence,
        max_hits: maxHits,
      });
      const job = await runJob(job_id, "Registry search");
      setRegistry(analysisId, job.result as RegistrySearchResponse, variant);
      setStatus(analysisId, "draft");
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Registry search failed";
      setLocalError(message);
      setError(analysisId, message);
    }
  }

  const displayError = localError;

  return (
    <Paper>
      <Stack gap="sm">
        <CardHeader
          eyebrow="Interoperability"
          title="Registry Interoperability"
          status={
            <Badge color={registry ? "green" : loading ? "blue" : "gray"}>
              {loading ? "running" : registry ? "complete" : "idle"}
            </Badge>
          }
        />
        <Text size="sm" c="dimmed">
          Search iGEM Parts Registry for homologous BioBricks via MMseqs2.
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
        />
        {loading && (
          <Progress value={progress * 100} animated striped size="sm" />
        )}
        {displayError && (
          <Alert color="red" title="Error">
            {displayError}
          </Alert>
        )}
        {registry && (
          <Stack gap="sm">
            {registry.matches.length === 0 ? (
              <Text size="sm" c="dimmed">
                No matching parts found for this sequence.
              </Text>
            ) : (
              <Table striped highlightOnHover withTableBorder>
                <Table.Thead>
                  <Table.Tr>
                    <Table.Th>Part</Table.Th>
                    <Table.Th>Identity</Table.Th>
                    <Table.Th>Description</Table.Th>
                  </Table.Tr>
                </Table.Thead>
                <Table.Tbody>
                  {registry.matches.map((m) => (
                    <Table.Tr key={m.part_id}>
                      <Table.Td>
                        <Anchor href={m.biobrick_url} target="_blank" size="sm">
                          {m.part_id}
                        </Anchor>
                      </Table.Td>
                      <Table.Td>{m.identity.toFixed(1)}%</Table.Td>
                      <Table.Td>{m.description}</Table.Td>
                    </Table.Tr>
                  ))}
                </Table.Tbody>
              </Table>
            )}
          </Stack>
        )}
        <Group>
          <Button
            onClick={handleRun}
            loading={loading}
            disabled={!activeSequence || loading}
            color="brand"
          >
            Search Registry ({variant})
          </Button>
        </Group>
      </Stack>
    </Paper>
  );
}
