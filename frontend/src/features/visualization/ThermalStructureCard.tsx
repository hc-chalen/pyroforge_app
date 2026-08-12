import {
  Alert,
  Badge,
  Box,
  Button,
  Group,
  Paper,
  Progress,
  Slider,
  Stack,
  Text,
  useComputedColorScheme,
} from "@mantine/core";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createViewer, type GLViewer } from "3dmol";
import { CardHeader } from "@/components/ui/CardHeader";
import { InfoNote } from "@/components/ui/InfoNote";
import { WithheldNotice } from "@/components/ui/Withheld";
import {
  SequenceVariantSelector,
  StaleVariantAlert,
} from "@/components/ui/SequenceVariantSelector";
import { useAsyncJob } from "@/hooks/useAsyncJob";
import { useSequenceVariant, useVariantResult } from "@/hooks/useSequenceVariant";
import { foldStructure } from "@/services/visualize";
import { useAnalysisStore } from "@/store/analysisStore";
import { getFoldPdb, setFoldPdb } from "@/store/foldPdbCache";
import type { FoldResult, OptimizationResult, PredictResponse } from "@/types/api";
import {
  applyThermalStylesToViewer,
  downloadPdb,
} from "@/utils/structureViewer";
import {
  FALLBACK_BASELINE_TM,
  baselineMeltingTemperature,
  computeResidueThermalStates,
  hasPredictedBaselineTm,
  temperatureRange,
} from "@/utils/thermalColors";
import { normalizeRmsfEntries } from "@/utils/normalizeRmsf";
import { withheldFold } from "@/utils/answerability";
import { hasOptimizedVariant } from "@/utils/sequenceVariant";

interface ThermalStructureCardProps {
  analysisId: string;
  sequence: string;
  prediction?: PredictResponse;
  optimization?: OptimizationResult;
}

export function ThermalStructureCard({
  analysisId,
  sequence,
  prediction,
  optimization,
}: ThermalStructureCardProps) {
  const viewerRef = useRef<HTMLDivElement>(null);
  const viewerInstanceRef = useRef<GLViewer | null>(null);
  const [localError, setLocalError] = useState<string | undefined>();
  const [viewerError, setViewerError] = useState<string | undefined>();
  const { loading, progress, runJob } = useAsyncJob();
  const setFold = useAnalysisStore((s) => s.setFold);
  const setStatus = useAnalysisStore((s) => s.setStatus);
  const setError = useAnalysisStore((s) => s.setError);
  const colorScheme = useComputedColorScheme("light");
  const viewerBackground = colorScheme === "dark" ? "#1a2029" : "#e7ebef";

  const { variant, activeSequence, setVariant } = useSequenceVariant(
    analysisId,
    sequence,
    optimization,
  );
  const { data: fold, resultVariant, staleReason } = useVariantResult<FoldResult>(
    analysisId,
    variant,
    "fold",
    activeSequence,
  );

  const pdbText = useMemo(() => {
    if (!fold) return undefined;
    return getFoldPdb(analysisId, variant) ?? fold.pdb ?? undefined;
  }, [analysisId, fold, variant]);

  const baselineTm = useMemo(() => {
    // model_tm is point-mutation-insensitive, so the reliable whole-protein Tm to
    // anchor the thermal view on is the wild-type baseline for either variant.
    if (variant === "optimized" && optimization?.baseline_tm != null) {
      return optimization.baseline_tm;
    }
    return baselineMeltingTemperature(
      prediction?.tm ?? undefined,
      optimization?.baseline_tm ?? undefined,
    );
  }, [variant, prediction?.tm, optimization?.baseline_tm]);

  // Whether baselineTm came from a model or from the fallback reference — the
  // fallback must never be labelled "predicted Tm".
  const tmIsPredicted = hasPredictedBaselineTm(
    prediction?.tm,
    optimization?.baseline_tm,
  );

  const { min: tempMin, max: tempMax, default: tempDefault } = useMemo(
    () => temperatureRange(baselineTm),
    [baselineTm],
  );

  const [temperature, setTemperature] = useState(tempDefault);

  useEffect(() => {
    setTemperature(tempDefault);
  }, [pdbText, tempDefault]);

  const thermalResidues = useMemo(() => {
    if (!fold || !activeSequence) return [];
    return computeResidueThermalStates(
      activeSequence,
      normalizeRmsfEntries(fold.rmsf ?? []),
      baselineTm,
      temperature,
    );
  }, [fold, activeSequence, baselineTm, temperature]);

  const initViewer = useCallback(() => {
    if (!pdbText || !viewerRef.current) return;

    try {
      setViewerError(undefined);
      viewerInstanceRef.current?.clear?.();
      viewerInstanceRef.current = null;

      const viewer = createViewer(viewerRef.current, {
        backgroundColor: viewerBackground,
      });
      viewerInstanceRef.current = viewer;
      viewer.addModel(pdbText, "pdb");
      viewer.zoomTo();
      viewer.setStyle({}, { cartoon: { color: "#b0b8c4" } });
      viewer.render();
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to initialize 3D viewer";
      setViewerError(message);
      console.error("3Dmol init error:", err);
    }
  }, [pdbText, viewerBackground]);

  useEffect(() => {
    initViewer();
    return () => {
      viewerInstanceRef.current?.clear?.();
      viewerInstanceRef.current = null;
    };
  }, [initViewer]);

  useEffect(() => {
    const viewer = viewerInstanceRef.current;
    if (!viewer || thermalResidues.length === 0) return;
    try {
      applyThermalStylesToViewer(viewer, thermalResidues);
    } catch (err) {
      console.error("3Dmol style update error:", err);
    }
  }, [thermalResidues]);

  async function handleRun() {
    if (!activeSequence) return;
    setLocalError(undefined);
    setViewerError(undefined);
    setStatus(analysisId, "running");
    setError(analysisId, undefined);
    try {
      const { job_id } = await foldStructure({ sequence: activeSequence });
      const job = await runJob(job_id, "Structure visualization");
      const result = job.result as FoldResult;
      if (result.pdb) {
        setFoldPdb(analysisId, result.pdb, variant);
      }
      setFold(analysisId, result, variant, activeSequence);
      setStatus(analysisId, "draft");
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Visualization failed";
      setLocalError(message);
      setError(analysisId, message);
    }
  }

  const labileCount = thermalResidues.filter((r) => r.labileFraction > 0.5).length;
  const foldRefusal = withheldFold(activeSequence);

  const tmMark = Math.round(baselineTm);
  const sliderMarks = useMemo(() => {
    const marks = [
      { value: tempMin, label: `${tempMin}°` },
      { value: tempMax, label: `${tempMax}°` },
    ];
    if (tmMark > tempMin && tmMark < tempMax) {
      marks.splice(1, 0, {
        value: tmMark,
        label: tmIsPredicted ? `Tm ${tmMark}°` : `ref ${tmMark}°`,
      });
    }
    return marks;
  }, [tempMin, tempMax, tmMark, tmIsPredicted]);

  return (
    <Paper>
      <Stack gap="sm">
        <CardHeader
          eyebrow="Structure"
          title="3D Structure & Thermal Heatmap"
          status={
            <Badge color={fold ? "green" : loading ? "blue" : "gray"}>
              {loading ? "running" : fold ? "ready" : "idle"}
            </Badge>
          }
        />
        <Text size="sm" c="dimmed">
          ESMFold structure, coloured by per-residue thermal vulnerability.
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
        {foldRefusal?.scope === "step" ? (
          <WithheldNotice withheld={foldRefusal} />
        ) : (
          <InfoNote summary="The colours rank fragile regions; they are not predicted temperatures.">
            Vulnerability is derived from pLDDT, a structure-confidence score used
            here as a flexibility proxy. Read the map as a relative ordering of
            fragile regions — the per-residue melt onsets on the slider are
            illustrative only.
          </InfoNote>
        )}
        {loading && (
          <Progress value={progress * 100} animated striped size="sm" />
        )}
        {localError && (
          <Alert color="red" title="Error">
            {localError}
          </Alert>
        )}
        {fold && (
          <Stack gap="sm">
            <Paper
              p={0}
              radius="md"
              style={{
                overflow: "hidden",
                border: "1px solid var(--mantine-color-default-border)",
              }}
            >
              {viewerError ? (
                <Alert color="orange" m="md" title="3D viewer unavailable">
                  {viewerError}
                </Alert>
              ) : (
                <div
                  ref={viewerRef}
                  style={{
                    width: "100%",
                    height: 400,
                    position: "relative",
                  }}
                />
              )}
              <Box
                p="md"
                style={{
                  borderTop: "1px solid var(--mantine-color-default-border)",
                  backgroundColor: "var(--pyro-card-nested-bg)",
                }}
              >
                <Stack gap="sm">
                  <Group justify="space-between" wrap="nowrap">
                    <Text size="sm" fw={500}>
                      System temperature
                    </Text>
                    <Text size="sm" fw={700} c="brand">
                      {temperature}°C
                    </Text>
                  </Group>
                  <Group justify="space-between" wrap="nowrap" gap="md">
                    <Group gap={6} wrap="nowrap">
                      <Box
                        w={14}
                        h={14}
                        style={{
                          borderRadius: 3,
                          backgroundColor: "#2f6fb0",
                          flexShrink: 0,
                        }}
                      />
                      <Text size="xs" c="dimmed">
                        Stable
                      </Text>
                    </Group>
                    <Group gap={6} wrap="nowrap">
                      <Box
                        w={14}
                        h={14}
                        style={{
                          borderRadius: 3,
                          backgroundColor: "#c23a2c",
                          flexShrink: 0,
                        }}
                      />
                      <Text size="xs" c="dimmed">
                        Labile
                      </Text>
                    </Group>
                  </Group>
                  <Box pb="lg" style={{ paddingInline: 4 }}>
                    <Slider
                      aria-label="System temperature"
                      value={temperature}
                      onChange={setTemperature}
                      min={tempMin}
                      max={tempMax}
                      step={1}
                      color="brand"
                      marks={sliderMarks}
                      disabled={!pdbText || Boolean(viewerError)}
                    />
                  </Box>
                  <Text size="xs" c="dimmed" ta="center">
                    {labileCount} labile hotspot
                    {labileCount === 1 ? "" : "s"} at {temperature}°C
                  </Text>
                </Stack>
              </Box>
            </Paper>

            {thermalResidues.length > 0 && (
              <Group gap={2} wrap="wrap" aria-label="Sequence thermal map">
                {thermalResidues.map((r) => (
                  <div
                    key={r.position}
                    title={`${r.aa}${r.position} · vulnerability ${(r.vulnerability * 100).toFixed(0)}% · illustrative onset ~${r.meltOnset.toFixed(0)}°C`}
                    style={{
                      width: 18,
                      height: 18,
                      backgroundColor: r.color,
                      borderRadius: 2,
                      fontSize: 9,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      color: r.labileFraction > 0.5 ? "#fff" : "#1b2530",
                      fontFamily: "'IBM Plex Mono', monospace",
                    }}
                  >
                    {r.aa}
                  </div>
                ))}
              </Group>
            )}

            <Group>
              <Button
                variant="subtle"
                size="compact-sm"
                disabled={!pdbText}
                onClick={() => pdbText && downloadPdb(pdbText)}
              >
                Download PDB
              </Button>
              <Text size="xs" c="dimmed">
                {fold.sequence_length} residues · {variant} ·{" "}
                {tmIsPredicted
                  ? `Tm ${baselineTm.toFixed(1)}°C`
                  : `no Tm — colours use a ${FALLBACK_BASELINE_TM}°C reference`}
              </Text>
            </Group>
          </Stack>
        )}
        <Group>
          <Button
            onClick={handleRun}
            loading={loading}
            disabled={!activeSequence || loading || foldRefusal != null}
            color="brand"
          >
            Fold & Visualize ({variant})
          </Button>
        </Group>
      </Stack>
    </Paper>
  );
}
