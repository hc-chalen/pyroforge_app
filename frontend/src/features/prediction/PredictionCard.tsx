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
import { InfoNote } from "@/components/ui/InfoNote";
import { MetricLabel } from "@/components/ui/MetricLabel";
import { StatPaper } from "@/components/ui/StatPaper";
import { WithheldNotice, WithheldValue } from "@/components/ui/Withheld";
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
import {
  withheldSequence,
  withheldStability,
  withheldTm,
} from "@/utils/answerability";
import { hasOptimizedVariant } from "@/utils/sequenceVariant";

/** ±95% interval under a stat value. Renders nothing when the model has no estimate. */
function CiHint({ ci95, unit }: { ci95?: number | null; unit: string }) {
  if (ci95 == null) return null;
  return (
    <Text size="xs" c="dimmed" ff="monospace">
      ± {ci95.toFixed(1)} {unit} (95%)
    </Text>
  );
}

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

  const inputRefusal = withheldSequence(activeSequence);
  const disabled = !activeSequence || loading || inputRefusal != null;

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

  // Honest-refusal gates. Tm and ΔG/ΔΔG have different training domains, so a
  // sequence can be answerable for one and not the other.
  const tmRefusal = withheldTm(displayPrediction);
  const stabilityRefusal = withheldStability(displayPrediction);

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
      setPrediction(
        analysisId,
        result,
        variant,
        variant === "optimized" && mutations.length > 0 ? wtSequence : activeSequence,
      );
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
          Melting temperature and folding stability for the current sequence.
        </Text>
        {hasOptimizedVariant(optimization) && (
          <SequenceVariantSelector
            sequence={sequence}
            optimization={optimization}
            variant={variant}
            onVariantChange={setVariant}
          />
        )}
        {inputRefusal?.scope === "step" && (
          <WithheldNotice withheld={inputRefusal} />
        )}
        {!modelsReady && !healthLoading && (
          <InfoNote tone="warning" summary="Model checkpoints are not loaded yet.">
            The first run asks the backend to load them; if the checkpoint files
            are missing it will return an error instead of a prediction.
          </InfoNote>
        )}
        {localError && (
          <Alert color="red" title="Error">
            {localError}
          </Alert>
        )}
        {displayPrediction && (
          <Stack gap="sm">
            {isStale && (
              <InfoNote
                tone="warning"
                summary={`Showing the ${otherVariant} result — not yet run for ${variant}.`}
              />
            )}
            <div style={{ opacity: isStale ? 0.5 : 1 }}>
              <SimpleGrid cols={{ base: 1, sm: 2, md: 3 }}>
                <StatPaper>
                  <MetricLabel
                    label="Tm (°C)"
                    method="absolute"
                    tooltip={
                      showBaselineTm
                        ? "Whole-protein Tm from model_tm. Point mutations barely move the mean-pooled embedding, so the variant sits at ≈ the wild-type baseline — the mutation effect lives in ΔΔG."
                        : "Absolute melting temperature from the Meltome-trained model_tm."
                    }
                  />
                  {tmRefusal ? (
                    <WithheldValue
                      withheld={tmRefusal}
                      value={
                        showBaselineTm
                          ? optimization?.baseline_tm?.toFixed(2)
                          : displayPrediction.tm?.toFixed(2)
                      }
                    />
                  ) : showBaselineTm ? (
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
                    <>
                      <Text fw={700} size="lg" ff="monospace">
                        {displayPrediction.tm != null
                          ? displayPrediction.tm.toFixed(2)
                          : "—"}
                      </Text>
                      {displayPrediction.tm != null && (
                        <CiHint ci95={displayPrediction.tm_ci95} unit="°C" />
                      )}
                    </>
                  )}
                </StatPaper>
                {displayPrediction.dg != null && (
                  <StatPaper>
                    <MetricLabel
                      label="ΔG (kcal/mol)"
                      method="absolute"
                      tooltip="Absolute folding free energy from the thermo model."
                    />
                    {stabilityRefusal ? (
                      <WithheldValue
                        withheld={stabilityRefusal}
                        value={displayPrediction.dg?.toFixed(2)}
                      />
                    ) : (
                      <>
                        <Text fw={700} size="lg" ff="monospace">
                          {displayPrediction.dg!.toFixed(2)}
                        </Text>
                        <CiHint ci95={displayPrediction.dg_ci95} unit="kcal/mol" />
                      </>
                    )}
                  </StatPaper>
                )}
                {ddgValue != null && (
                  <StatPaper>
                    <MetricLabel
                      label="ΔΔG (kcal/mol)"
                      method="derived"
                      tooltip="Difference-embedding model_ddg. Positive = stabilizing. This is the mutation-effect signal to read."
                    />
                    {stabilityRefusal ? (
                      <WithheldValue
                        withheld={stabilityRefusal}
                        value={ddgValue?.toFixed(2)}
                      />
                    ) : (
                      <>
                        <Text
                          fw={700}
                          size="lg"
                          ff="monospace"
                          c={
                            ddgValue! > 0
                              ? "teal"
                              : ddgValue! < 0
                                ? "red"
                                : undefined
                          }
                        >
                          {ddgValue! > 0 ? "+" : ""}
                          {ddgValue!.toFixed(2)}
                        </Text>
                        <Text size="xs" c="dimmed">
                          {ddgValue! > 0
                            ? "Stabilizing"
                            : ddgValue! < 0
                              ? "Destabilizing"
                              : "Neutral"}
                        </Text>
                        <CiHint ci95={displayPrediction.ddg_ci95} unit="kcal/mol" />
                      </>
                    )}
                  </StatPaper>
                )}
              </SimpleGrid>
            </div>
          </Stack>
        )}
        {!displayPrediction && !inputRefusal && (
          <InfoNote
            summary={
              hasOptimizedVariant(optimization)
                ? "Runs on the selected variant."
                : "ΔΔG needs a mutation — run optimization first."
            }
          >
            Tm comes from the Meltome-trained <code>model_tm</code>; ΔG from the
            thermo model; ΔΔG from the difference-embedding model, which scores a
            mutation against its wild type. Each is reported only for sequences
            inside the length regime it was trained on.
          </InfoNote>
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
