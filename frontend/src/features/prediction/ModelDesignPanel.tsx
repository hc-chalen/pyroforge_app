import {
  Accordion,
  Badge,
  Group,
  List,
  Stack,
  Text,
} from "@mantine/core";
import { useEffect, useState } from "react";
import { fetchNasReport } from "@/services/nasReport";
import type { NasModelSummary, NasReportResponse } from "@/types/api";

const ALGO_LABELS: Array<{ key: keyof NasModelSummary["algorithm_names"]; label: string }> = [
  { key: "search", label: "Search" },
  { key: "architecture", label: "Network" },
  { key: "activation", label: "Activation" },
  { key: "optimizer", label: "Optimizer" },
  { key: "normalization", label: "Norm" },
  { key: "loss", label: "Loss" },
];

function AlgorithmBadges({ algos }: { algos: NasModelSummary["algorithm_names"] }) {
  return (
    <Group gap={6}>
      {ALGO_LABELS.map(({ key, label }) => (
        <Badge key={key} size="md" variant="light" color="teal">
          {label}: {algos[key]}
        </Badge>
      ))}
    </Group>
  );
}

function formatShare(share: number | undefined): string {
  if (share == null) return "—";
  return `${Math.round(share * 100)}%`;
}

function NasModelSection({ model }: { model: NasModelSummary }) {
  const w = model.weighting;
  return (
    <Stack gap={6}>
      <Text size="sm" fw={600}>
        {model.label}
      </Text>
      <AlgorithmBadges algos={model.algorithm_names} />
      <Group gap={6}>
        <Badge size="sm" variant="light" color="grape">
          Weight decay: {w.weight_decay_label}
        </Badge>
        <Badge size="sm" variant="light" color="grape">
          NAS search time: {formatShare(w.search_time_share)}
        </Badge>
        <Badge size="sm" variant="light" color="gray">
          Loss: {w.loss_task_weights}
        </Badge>
      </Group>
      <Text size="xs" c="dimmed" ff="monospace">
        {model.architecture}
      </Text>
    </Stack>
  );
}

function WeightingPanel({ report }: { report: NasReportResponse }) {
  const prod = report.production_weighting;
  const budget = report.search_budget_weights;

  return (
    <Stack gap={6}>
      <Text size="sm" fw={600}>
        Model weighting
      </Text>
      {prod && (
        <Stack gap={4}>
          <Text size="xs" c="dimmed">
            Production PyTorch (inference)
          </Text>
          <Group gap={6}>
            <Badge size="sm" variant="light" color="indigo">
              ΔG loss: {prod.thermo_dg?.dg ?? 1}
            </Badge>
            <Badge size="sm" variant="light" color="indigo">
              Optimizer decay: {prod.adamw_weight_decay ?? 0}
            </Badge>
            <Badge size="sm" variant="light" color="indigo">
              Optimization: {prod.optimization_score?.formula ?? "-ΔΔG"}
            </Badge>
          </Group>
        </Stack>
      )}
      {budget && (
        <Stack gap={4}>
          <Text size="xs" c="dimmed">
            NAS search time budget (what autoPyTorch weighted)
          </Text>
          <Group gap={6}>
            <Badge size="sm" variant="light" color="grape">
              ΔG: {formatShare(budget.thermo_dg)}
            </Badge>
          </Group>
        </Stack>
      )}
    </Stack>
  );
}

function NasOptimizedPanel({ report }: { report: NasReportResponse }) {
  const first = report.models[0];
  const sharedAlgos = first
    ? {
        search: first.algorithm_names.search,
        activation: first.algorithm_names.activation,
        optimizer: first.algorithm_names.optimizer,
        normalization: first.algorithm_names.normalization,
        loss: first.algorithm_names.loss,
      }
    : null;

  return (
    <Stack gap="md">
      <Text size="sm" c="dimmed">
        Best algorithm stack selected by autoPyTorch-style NAS. Layer widths
        differ per target; activation, optimizer, normalization, and loss are
        shared across all three models.
      </Text>
      {sharedAlgos && (
        <Stack gap={6}>
          <Text size="sm" fw={600}>
            Shared algorithms
          </Text>
          <Group gap={6}>
            <Badge size="md" variant="filled" color="teal">
              Search: {sharedAlgos.search}
            </Badge>
            <Badge size="md" variant="filled" color="teal">
              Activation: {sharedAlgos.activation}
            </Badge>
            <Badge size="md" variant="filled" color="teal">
              Optimizer: {sharedAlgos.optimizer}
            </Badge>
            <Badge size="md" variant="filled" color="teal">
              Norm: {sharedAlgos.normalization}
            </Badge>
            <Badge size="md" variant="filled" color="teal">
              Loss: {sharedAlgos.loss}
            </Badge>
          </Group>
        </Stack>
      )}
      <WeightingPanel report={report} />
      <Stack gap="sm">
        <Text size="sm" fw={600}>
          Best network shape per target
        </Text>
        {report.models.map((model) => (
          <NasModelSection key={model.key} model={model} />
        ))}
      </Stack>
    </Stack>
  );
}

export function ModelDesignPanel() {
  const [nasReport, setNasReport] = useState<NasReportResponse | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetchNasReport()
      .then((report) => {
        if (!cancelled) setNasReport(report);
      })
      .catch(() => {
        if (!cancelled) setNasReport({ available: false, models: [] });
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <Accordion variant="contained" radius="md">
      <Accordion.Item value="design">
        <Accordion.Control>
          <Text size="sm" fw={600}>
            Hybrid thermo model design
          </Text>
        </Accordion.Control>
        <Accordion.Panel>
          <Text size="sm" c="dimmed" mb="xs">
            PyroForge runs three small models on frozen ESM2-650M embeddings:
          </Text>
          <List size="sm" spacing="xs" c="dimmed">
            <List.Item>
              <strong>Tm</strong> — absolute melting temperature from the
              Meltome-trained <Text span ff="monospace" size="sm">model_tm</Text>{" "}
              (PCA → Ridge + gradient boosting), run on the wild-type or mutant
              sequence embedding.
            </List.Item>
            <List.Item>
              <strong>ΔG</strong> — absolute folding free energy from{" "}
              <Text span ff="monospace" size="sm">model_thermo</Text>, an averaged
              ensemble of MLP towers.
            </List.Item>
            <List.Item>
              <strong>ΔΔG</strong> — from{" "}
              <Text span ff="monospace" size="sm">model_ddg</Text>, an MLP on the{" "}
              <Text span ff="monospace" size="sm">
                difference embedding Δemb = emb(mut) − emb(wt)
              </Text>
              . This is the trustworthy mutation-effect signal and what
              optimization ranks on.
            </List.Item>
            <List.Item>
              <strong>Optimization</strong> — ranks mutations by{" "}
              <Text span ff="monospace" size="sm">
                ΔΔG
              </Text>{" "}
              (higher ΔΔG = more stabilizing = higher score), behind an ESM-2
              residue-plausibility guardrail.
            </List.Item>
          </List>
        </Accordion.Panel>
      </Accordion.Item>
      {nasReport?.available && (
        <Accordion.Item value="nas">
          <Accordion.Control>
            <Group gap="xs">
              <Text size="sm" fw={600}>
                Best algorithms (autoPyTorch NAS)
              </Text>
              {nasReport.models[0] && (
                <Badge size="xs" variant="light" color="teal">
                  {nasReport.models[0].algorithm_names.activation} ·{" "}
                  {nasReport.models[0].algorithm_names.optimizer}
                </Badge>
              )}
            </Group>
          </Accordion.Control>
          <Accordion.Panel>
            <NasOptimizedPanel report={nasReport} />
          </Accordion.Panel>
        </Accordion.Item>
      )}
    </Accordion>
  );
}
