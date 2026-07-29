import {
  Alert,
  Badge,
  Button,
  Group,
  Paper,
  SimpleGrid,
  Stack,
  Text,
} from "@mantine/core";
import { useState } from "react";
import { CardHeader } from "@/components/ui/CardHeader";
import { MetricLabel } from "@/components/ui/MetricLabel";
import { StatPaper } from "@/components/ui/StatPaper";
import {
  SequenceVariantSelector,
} from "@/components/ui/SequenceVariantSelector";
import { useHealth } from "@/hooks/useHealth";
import { useSequenceVariant } from "@/hooks/useSequenceVariant";
import { predict } from "@/services/predict";
import { useAnalysisStore } from "@/store/analysisStore";
import type {
  OptimizationResult,
  PredictResponse,
  SequenceVariant,
} from "@/types/api";
import { hasOptimizedVariant } from "@/utils/sequenceVariant";

interface PredictionCardProps {
  analysisId: string;
  sequence: string;
  prediction?: PredictResponse;
  optimization?: OptimizationResult;
}

export function PredictionCard({
  analysisId,
  sequence,
  prediction,
  optimization,
}: PredictionCardProps) {
  const [loading, setLoading] = useState(false);
  const [localError, setLocalError] = useState<string | undefined>();
  const { modelsReady, loading: healthLoading } = useHealth();
  const setPrediction = useAnalysisStore((s) => s.setPrediction);
  const setStatus = useAnalysisStore((s) => s.setStatus);
  const setError = useAnalysisStore((s) => s.setError);

  const { variant, activeSequence, setVariant } = useSequenceVariant(
    analysisId,
    sequence,
    optimization,
  );
  const variantResults = useAnalysisStore(
    (s) => s.analyses.find((a) => a.id === analysisId)?.variantResults,
  );

  const disabled = !activeSequence || loading;
  const hasMutations =
    variant === "optimized" && (optimization?.mutations.length ?? 0) > 0;

  // Prediction is variant-aware: prefer the result computed for the selected
  // variant. If none exists yet, fall back to the other variant's result shown
  // dimmed + flagged stale (rather than silently displaying the wrong numbers).
  const otherVariant: SequenceVariant =
    variant === "original" ? "optimized" : "original";
  const ownPrediction =
    variantResults?.[variant]?.prediction ??
    (variant === "original" ? prediction : undefined);
  const otherPrediction =
    variantResults?.[otherVariant]?.prediction ??
    (otherVariant === "original" ? prediction : undefined);
  const displayPrediction = ownPrediction ?? otherPrediction;
  const isStale = !ownPrediction && Boolean(otherPrediction);

  // For the optimized variant the mutation effect is ΔΔG (from optimization);
  // model_tm barely moves for point mutations, so Tm is shown as ≈ baseline.
  const showBaselineTm = variant === "optimized" && optimization != null;
  const ddgValue =
    (variant === "optimized"
      ? (optimization?.ddg ?? displayPrediction?.ddg)
      : displayPrediction?.ddg) ?? null;

  async function handleRun() {
    setLoading(true);
    setStatus(analysisId, "running");
    setError(analysisId, undefined);
    setLocalError(undefined);
    try {
      const wtSequence = (
        optimization?.wild_type_sequence ?? sequence
      ).trim();
      const mutations = optimization?.mutations ?? [];
      const result =
        variant === "optimized" && mutations.length > 0
          ? await predict({ sequence: wtSequence, mutations })
          : await predict({ sequence: activeSequence });
      setPrediction(analysisId, result, variant);
      setStatus(analysisId, "draft");
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Prediction failed";
      setLocalError(message);
      setError(analysisId, message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <Paper>
      <Stack gap="sm">
        <CardHeader
          eyebrow="Model output"
          title="Thermodynamic Prediction"
          status={
            <Badge color={ownPrediction ? "green" : isStale ? "yellow" : "gray"}>
              {ownPrediction ? "complete" : isStale ? "stale" : "idle"}
            </Badge>
          }
        />
        <Text size="sm" c="dimmed">
          Absolute Tm (Meltome-trained model_tm) and ΔG on the sequence; ΔΔG from
          the difference-embedding model ranks the mutation effect.
          {hasOptimizedVariant(optimization)
            ? " Select original or optimized variant below."
            : hasMutations
              ? " Optimization mutations will be included to compute ΔΔG."
              : " Run optimization first to compute the stability change (ΔΔG)."}
        </Text>
        {hasOptimizedVariant(optimization) && (
          <SequenceVariantSelector
            sequence={sequence}
            optimization={optimization}
            variant={variant}
            onVariantChange={setVariant}
          />
        )}
        {!modelsReady && !healthLoading && (
          <Alert color="orange" title="Models not ready">
            Prediction will ask the backend to load checkpoints. If checkpoint
            files are missing, the backend will return an error.
          </Alert>
        )}
        {localError && (
          <Alert color="red" title="Error">
            {localError}
          </Alert>
        )}
        {displayPrediction && (
          <Stack gap="sm">
            {isStale && (
              <Alert color="yellow" variant="light" p="xs">
                <Text size="xs">
                  Showing the {otherVariant} result — prediction hasn't been run
                  for the {variant} variant yet. Click “Run Prediction ({variant}
                  )” to compute it.
                </Text>
              </Alert>
            )}
            <div style={{ opacity: isStale ? 0.5 : 1 }}>
              <SimpleGrid cols={{ base: 1, sm: 2, md: 3 }}>
                <StatPaper>
                  <MetricLabel
                    label="Tm (°C)"
                    method="absolute"
                    tooltip={
                      showBaselineTm
                        ? "Whole-protein Tm from model_tm. A few point mutations barely move the mean-pooled embedding, so the variant's Tm stays ≈ the wild-type baseline — the mutation effect is captured by ΔΔG, not Tm."
                        : "Absolute melting temperature from the Meltome-trained model_tm, run on this sequence."
                    }
                  />
                  {showBaselineTm ? (
                    <>
                      <Text fw={700} size="lg" ff="monospace">
                        ≈ {optimization!.baseline_tm != null
                          ? optimization!.baseline_tm.toFixed(2)
                          : "—"}
                      </Text>
                      <Text size="xs" c="dimmed">
                        Point mutations don't shift Tm — see ΔΔG.
                      </Text>
                    </>
                  ) : (
                    <Text fw={700} size="lg" ff="monospace">
                      {displayPrediction.tm != null
                        ? displayPrediction.tm.toFixed(2)
                        : "—"}
                    </Text>
                  )}
                </StatPaper>
                {displayPrediction.dg != null && (
                  <StatPaper>
                    <MetricLabel
                      label="ΔG (kcal/mol)"
                      method="absolute"
                      tooltip="Absolute folding free energy (ΔG) from the thermo model."
                    />
                    <Text fw={700} size="lg" ff="monospace">
                      {displayPrediction.dg.toFixed(2)}
                    </Text>
                  </StatPaper>
                )}
                {ddgValue != null && (
                  <StatPaper>
                    <MetricLabel
                      label="ΔΔG (kcal/mol)"
                      method="derived"
                      tooltip="Difference-embedding model_ddg. Positive = stabilizing (more thermostable). The trustworthy mutation-effect signal."
                    />
                    <Text
                      fw={700}
                      size="lg"
                      ff="monospace"
                      c={ddgValue > 0 ? "teal" : ddgValue < 0 ? "red" : undefined}
                    >
                      {ddgValue > 0 ? "+" : ""}
                      {ddgValue.toFixed(2)}
                    </Text>
                    <Text size="xs" c="dimmed">
                      {ddgValue > 0
                        ? "Stabilizing"
                        : ddgValue < 0
                          ? "Destabilizing"
                          : "Neutral"}
                    </Text>
                  </StatPaper>
                )}
              </SimpleGrid>
            </div>
          </Stack>
        )}
        <Group>
          <Button
            onClick={handleRun}
            loading={loading}
            disabled={disabled}
            color="brand"
          >
            Run Prediction ({variant})
          </Button>
        </Group>
      </Stack>
    </Paper>
  );
}
