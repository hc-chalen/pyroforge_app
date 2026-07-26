import {
  Alert,
  Badge,
  Button,
  Group,
  Paper,
  Progress,
  ScrollArea,
  SimpleGrid,
  Stack,
  Table,
  Text,
} from "@mantine/core";
import { useState } from "react";
import { CardHeader } from "@/components/ui/CardHeader";
import { MetricLabel } from "@/components/ui/MetricLabel";
import { StatPaper } from "@/components/ui/StatPaper";
import { useAsyncJob } from "@/hooks/useAsyncJob";
import { useHealth } from "@/hooks/useHealth";
import { optimize } from "@/services/optimize";
import { StructureCompareCard } from "@/features/optimization/StructureCompareCard";
import { useAnalysisStore } from "@/store/analysisStore";
import { useSettingsStore } from "@/store/settingsStore";
import type { OptimizationResult } from "@/types/api";

interface OptimizationCardProps {
  analysisId: string;
  sequence: string;
  optimization?: OptimizationResult;
}

export function OptimizationCard({
  analysisId,
  sequence,
  optimization,
}: OptimizationCardProps) {
  const [localError, setLocalError] = useState<string | undefined>();
  const { loading, progress, runJob } = useAsyncJob();
  const { modelsReady, loading: healthLoading } = useHealth();
  const maxMutations = useSettingsStore((s) => s.maxMutations);
  const perplexityThreshold = useSettingsStore((s) => s.perplexityThreshold);
  const setOptimization = useAnalysisStore((s) => s.setOptimization);
  const setStructureComparison = useAnalysisStore((s) => s.setStructureComparison);
  const structureComparison = useAnalysisStore(
    (s) => s.analyses.find((a) => a.id === analysisId)?.structureComparison,
  );
  const setStatus = useAnalysisStore((s) => s.setStatus);
  const setError = useAnalysisStore((s) => s.setError);

  const disabled = !sequence.trim() || loading;

  async function handleRun() {
    setLocalError(undefined);
    setStatus(analysisId, "running");
    setError(analysisId, undefined);
    try {
      const { job_id } = await optimize({
        sequence: sequence.trim(),
        max_mutations: maxMutations,
        perplexity_threshold: perplexityThreshold,
      });
      const job = await runJob(job_id, "Optimization");
      const result = job.result as OptimizationResult;
      setOptimization(analysisId, result);
      if (result.structure_comparison) {
        setStructureComparison(analysisId, result.structure_comparison);
      }
      setStatus(analysisId, "draft");
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Optimization failed";
      setLocalError(message);
      setError(analysisId, message);
    }
  }

  const displayError = localError;

  return (
    <Paper>
      <Stack gap="sm">
        <CardHeader
          eyebrow="Mutation scan"
          title="Optimization"
          status={
            <Badge color={optimization ? "green" : loading ? "blue" : "gray"}>
              {loading ? "running" : optimization ? "complete" : "idle"}
            </Badge>
          }
        />
        <Text size="sm" c="dimmed">
          Site-saturation scan ranked by ΔΔG (difference-embedding model), with
          perplexity guardrail and greedy multi-mutant selection.
        </Text>
        {!modelsReady && !healthLoading && (
          <Alert color="orange" title="Models not ready">
            Optimization will ask the backend to load checkpoints. If checkpoint
            files are missing, the backend will return an error.
          </Alert>
        )}
        {loading && (
          <Progress value={progress * 100} animated striped size="sm" />
        )}
        {displayError && (
          <Alert color="red" title="Error">
            {displayError}
          </Alert>
        )}
        {optimization && (
          <Stack gap="sm">
            <SimpleGrid cols={{ base: 1, sm: 2 }}>
              <StatPaper>
                <MetricLabel
                  label="Baseline Tm (°C)"
                  method="absolute"
                  tooltip="Absolute Tm of the input protein from the Meltome-trained model_tm. A few point mutations barely move the mean-pooled embedding, so the variant's Tm stays ≈ this value — use ΔΔG for the stabilization effect."
                />
                <Text fw={700} size="lg" ff="monospace">
                  {optimization.baseline_tm != null
                    ? optimization.baseline_tm.toFixed(2)
                    : "—"}
                </Text>
              </StatPaper>
              <StatPaper>
                <MetricLabel
                  label="ΔΔG (kcal/mol)"
                  method="derived"
                  tooltip="Total ΔΔG of the selected mutations from difference-embedding model_ddg. Positive = stabilizing → predicted more thermostable. This is the optimization objective."
                />
                <Text fw={700} size="lg" ff="monospace">
                  {optimization.ddg != null ? optimization.ddg.toFixed(2) : "—"}
                </Text>
              </StatPaper>
            </SimpleGrid>
            <Text size="xs" c="dimmed" ff="monospace" lineClamp={2}>
              {optimization.optimized_sequence}
            </Text>
            {optimization.mutations.length > 0 && (
              <ScrollArea type="auto">
                <Table striped highlightOnHover withTableBorder miw={240}>
                <Table.Thead>
                  <Table.Tr>
                    <Table.Th>Position</Table.Th>
                    <Table.Th>Change</Table.Th>
                  </Table.Tr>
                </Table.Thead>
                <Table.Tbody>
                  {optimization.mutations.map((m) => (
                    <Table.Tr key={`${m.position}-${m.target_aa}`}>
                      <Table.Td>{m.position + 1}</Table.Td>
                      <Table.Td ff="monospace">
                        {m.source_aa}→{m.target_aa}
                      </Table.Td>
                    </Table.Tr>
                  ))}
                </Table.Tbody>
              </Table>
              </ScrollArea>
            )}
            <StructureCompareCard
              sequence={sequence}
              optimization={optimization}
              comparison={structureComparison ?? optimization.structure_comparison}
              onComparison={(c) => setStructureComparison(analysisId, c)}
            />
          </Stack>
        )}
        <Group>
          <Button
            onClick={handleRun}
            loading={loading}
            disabled={disabled}
            color="brand"
          >
            Run Optimization
          </Button>
        </Group>
      </Stack>
    </Paper>
  );
}
