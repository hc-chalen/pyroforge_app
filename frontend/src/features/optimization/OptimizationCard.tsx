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
import { SequenceViewer } from "@/components/ui/SequenceViewer";
import { StatPaper } from "@/components/ui/StatPaper";
import { useAsyncJob } from "@/hooks/useAsyncJob";
import { useHealth } from "@/hooks/useHealth";
import { optimize } from "@/services/optimize";
import { StructureCompareCard } from "@/features/optimization/StructureCompareCard";
import { useAnalysisStore } from "@/store/analysisStore";
import { useSettingsStore } from "@/store/settingsStore";
import type { OptimizationResult, RankingQuality } from "@/types/api";
import { isOptimizationStale } from "@/utils/sequenceVariant";

function pct(value?: number): string {
  return value == null ? "—" : `${Math.round(value * 100)}%`;
}

/**
 * States what the shortlist ordering is measured to be worth, and where it fails.
 *
 * This replaces the previous "cleared a 1.96σ significance gate" copy, which read
 * as a statistical guarantee about each individual mutation that the model cannot
 * support. What held-out data does support is the *ranking*, so that is what is
 * quoted — including the per-protein spread, because the mean alone hides the
 * proteins where the ranking fails outright.
 */
function RankingQualityAlert({ quality }: { quality?: RankingQuality | null }) {
  if (!quality) return null;
  const k = quality.headline_k ?? 5;
  const precision = quality.top_k_precision?.[String(k)];
  const baseRate = quality.base_rate_stabilizing;
  if (precision == null || baseRate == null || !baseRate) return null;
  const ci = quality.headline_precision_ci95;
  const spearman = quality.median_within_protein_spearman;

  return (
    <Alert color="gray" variant="light" title="What this ranking is measured to be worth" p="xs">
      <Text size="xs">
        On {quality.n_proteins ?? "?"} held-out proteins,{" "}
        <strong>{pct(precision)}</strong> of a top-{k} shortlist was truly stabilizing
        {ci?.length === 2 ? ` (95% CI ${pct(ci[0])}–${pct(ci[1])})` : ""}, against a{" "}
        {pct(baseRate)} base rate — a {(precision / baseRate).toFixed(1)}×
        enrichment
        {spearman != null
          ? `, with a median within-protein rank correlation of ${spearman.toFixed(2)}`
          : ""}
        .{" "}
        <strong>It is not uniform.</strong> Per protein that precision ranged from{" "}
        {pct(quality.headline_precision_min)} to {pct(quality.headline_precision_max)}
        {quality.proteins_below_half_precision != null && quality.n_proteins
          ? `, and fewer than half the shortlist was stabilizing for ${quality.proteins_below_half_precision} of the ${quality.n_proteins} proteins`
          : ""}
        . Screen the shortlist as a whole; do not commit to a single entry on this
        evidence.
      </Text>
    </Alert>
  );
}

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
          Site-saturation scan ranked by ΔΔG (difference-embedding model), with an
          ESM-2 plausibility guardrail and greedy multi-mutant selection. The result
          is a <strong>ranked shortlist to screen</strong>, not a set of verified
          stabilizing mutations.
        </Text>
        <RankingQualityAlert quality={optimization?.ranking_quality} />
        <Alert color="gray" variant="light" title="What ΔΔG does and doesn't say" p="xs">
          <Text size="xs">
            ΔΔG is a change in <strong>folding free energy</strong> (Tsuboyama
            small-domain assay). Tm is a <strong>different measurement</strong>
            {" "}(Meltome whole-protein assay), and the two are only weakly related —
            corr(ΔG, Tm) is +0.24 row-level and +0.17 per-protein across the 21
            proteins carrying both labels. Read a positive ΔΔG as “predicted more
            resistant to unfolding”, <strong>not</strong> as a predicted Tm gain.
            {(optimization?.mutations.length ?? 0) > 1 && (
              <>
                {" "}
                The combined ΔΔG for a {optimization!.mutations.length}-mutation
                variant is also an <strong>extrapolation</strong>: model_ddg was
                trained on single substitutions, and selection assumes their effects
                add without epistasis.
              </>
            )}
          </Text>
        </Alert>
        {!modelsReady && !healthLoading && (
          <Alert color="orange" title="Models not ready">
            Optimization will ask the backend to load checkpoints. If checkpoint
            files are missing, the backend will return an error.
          </Alert>
        )}
        {isOptimizationStale(sequence, optimization) && (
          <Alert color="yellow" title="Optimization is out of date">
            <Text size="xs">
              The input sequence has been edited since this scan ran, so these
              mutations and their positions refer to the previous sequence. Re-run the
              optimization.
            </Text>
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
                  tooltip="Total ΔΔG of the shortlisted mutations from difference-embedding model_ddg. Positive = predicted more resistant to unfolding. It is a folding-stability change, not a predicted Tm shift. Read it as a rank-ordering signal: on held-out data the model's absolute level is biased low, so the magnitude is less trustworthy than the ordering."
                />
                <Text fw={700} size="lg" ff="monospace">
                  {optimization.ddg != null ? optimization.ddg.toFixed(2) : "—"}
                </Text>
                {optimization.ddg_ci95 != null && (
                  <Text size="xs" c="dimmed" ff="monospace">
                    ± {optimization.ddg_ci95.toFixed(2)} kcal/mol (95%)
                  </Text>
                )}
              </StatPaper>
            </SimpleGrid>
            {optimization.mutations.length === 0 ? (
              <Alert
                color="blue"
                variant="light"
                title="No stabilizing candidate found — sequence returned unchanged"
              >
                <Text size="xs">
                  {(optimization.scan_summary?.total_candidates ?? 0) === 0 ? (
                    <>
                      Every substitution was rejected by the ESM-2 plausibility
                      guardrail (threshold{" "}
                      {optimization.scan_summary?.perplexity_threshold ??
                        perplexityThreshold}
                      ), so nothing reached the ΔΔG model. Lower the threshold in
                      settings to widen the scan.
                    </>
                  ) : (
                    <>
                      The scan scored{" "}
                      {optimization.scan_summary?.total_candidates ?? 0} plausible
                      substitutions and none of them was predicted to be stabilizing
                      (ΔΔG &gt; 0).
                    </>
                  )}{" "}
                  That is why the optimized sequence below is byte-for-byte the input
                  — it is a result, not a failure. Note the model's absolute level is
                  biased low, so a bare “nothing found” is weaker evidence than it
                  looks.
                </Text>
              </Alert>
            ) : (
              optimization.scan_summary?.candidates_ddg_positive != null && (
                <Text size="xs" c="dimmed">
                  Shortlisted the top {optimization.mutations.length} of{" "}
                  {optimization.scan_summary.candidates_ddg_positive} candidate
                  {optimization.scan_summary.candidates_ddg_positive === 1 ? "" : "s"}{" "}
                  predicted stabilizing, one per position, out of{" "}
                  {optimization.scan_summary.total_candidates ?? 0} scored.
                </Text>
              )
            )}
            <SequenceViewer
              sequence={optimization.optimized_sequence}
              label="Optimized sequence"
              fastaName={`optimized_${optimization.mutations
                .map((m) => `${m.source_aa}${m.position + 1}${m.target_aa}`)
                .join("_") || "wild_type"}`}
              highlights={optimization.mutations.map((m) => m.position)}
              meta={
                optimization.optimized_sequence ===
                optimization.wild_type_sequence ? (
                  <Badge size="xs" variant="light" color="gray">
                    identical to input
                  </Badge>
                ) : undefined
              }
            />
            {optimization.mutations.length > 0 && (
              <ScrollArea type="auto">
                <Table striped highlightOnHover withTableBorder miw={240}>
                <Table.Thead>
                  <Table.Tr>
                    <Table.Th>Rank</Table.Th>
                    <Table.Th>Position</Table.Th>
                    <Table.Th>Change</Table.Th>
                    <Table.Th>ΔΔG</Table.Th>
                    <Table.Th>ESM-2 P</Table.Th>
                  </Table.Tr>
                </Table.Thead>
                <Table.Tbody>
                  {optimization.mutations.map((m, i) => {
                    // Per-entry evidence, when the backend supplied it. A shortlist
                    // shown without its ranking reads as a set of equals.
                    const detail = optimization.mutation_details?.find(
                      (d) => d.position === m.position && d.target_aa === m.target_aa,
                    );
                    return (
                      <Table.Tr key={`${m.position}-${m.target_aa}`}>
                        <Table.Td>{detail?.rank ?? i + 1}</Table.Td>
                        <Table.Td>{m.position + 1}</Table.Td>
                        <Table.Td ff="monospace">
                          {m.source_aa}→{m.target_aa}
                        </Table.Td>
                        <Table.Td ff="monospace">
                          {detail ? `${detail.ddg > 0 ? "+" : ""}${detail.ddg.toFixed(2)}` : "—"}
                        </Table.Td>
                        <Table.Td ff="monospace">
                          {detail ? detail.residue_prob.toFixed(2) : "—"}
                        </Table.Td>
                      </Table.Tr>
                    );
                  })}
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
