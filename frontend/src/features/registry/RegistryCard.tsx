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
import { useEffect, useState } from "react";
import { CardHeader } from "@/components/ui/CardHeader";
import { InfoNote } from "@/components/ui/InfoNote";
import {
  SequenceVariantSelector,
  StaleVariantAlert,
} from "@/components/ui/SequenceVariantSelector";
import { useAsyncJob } from "@/hooks/useAsyncJob";
import { useSequenceVariant, useVariantResult } from "@/hooks/useSequenceVariant";
import { getRegistryStatus, searchRegistry } from "@/services/registry";
import { useAnalysisStore } from "@/store/analysisStore";
import { useSettingsStore } from "@/store/settingsStore";
import type {
  OptimizationResult,
  RegistryDatabaseInfo,
  RegistrySearchResponse,
} from "@/types/api";
import { withheldSequence } from "@/utils/answerability";
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
  // Asked once on mount. Until it answers the card behaves normally; the point
  // is only to stop a user submitting into a server that has no database.
  const [dbStatus, setDbStatus] = useState<RegistryDatabaseInfo | null>(null);

  useEffect(() => {
    let cancelled = false;
    getRegistryStatus()
      .then((s) => {
        if (!cancelled) setDbStatus(s);
      })
      .catch(() => {
        // A failed status probe must not disable a working feature — leave the
        // card as it was and let the search itself report any problem.
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const unavailable = dbStatus?.status === "not_provisioned";
  const maxHits = useSettingsStore((s) => s.maxHits);
  const setRegistry = useAnalysisStore((s) => s.setRegistry);
  const setStatus = useAnalysisStore((s) => s.setStatus);
  const setError = useAnalysisStore((s) => s.setError);

  const { variant, activeSequence, setVariant } = useSequenceVariant(
    analysisId,
    sequence,
    optimization,
  );
  const { data: registry, resultVariant, staleReason } =
    useVariantResult<RegistrySearchResponse>(
      analysisId,
      variant,
      "registry",
      activeSequence,
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
      setRegistry(
        analysisId,
        job.result as RegistrySearchResponse,
        variant,
        activeSequence,
      );
      setStatus(analysisId, "draft");
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Registry search failed";
      setLocalError(message);
      setError(analysisId, message);
    }
  }

  const displayError = localError;
  const inputRefusal = withheldSequence(activeSequence);

  return (
    <Paper>
      <Stack gap="sm">
        <CardHeader
          eyebrow="Interoperability"
          title="Registry Interoperability"
          status={
            <Badge
              color={
                unavailable
                  ? "yellow"
                  : registry
                    ? "green"
                    : loading
                      ? "blue"
                      : "gray"
              }
            >
              {unavailable
                ? "unavailable"
                : loading
                  ? "running"
                  : registry
                    ? "complete"
                    : "idle"}
            </Badge>
          }
        />
        <Text size="sm" c="dimmed">
          Homologous BioBricks in a local snapshot of the iGEM Parts Registry
          (MMseqs2).
        </Text>
        {/* Said up front, not after the user submits and waits. The cause is
            outside this app: the iGEM Registry's bulk export sits behind a CDN
            that returns 403 to server networks, so the snapshot has to be
            installed by hand and often simply is not there. The user did nothing
            wrong and can do nothing about it, so the copy says exactly that and
            keeps the operator's shell command out of their way. */}
        {unavailable && (
          <Alert color="yellow" title="Not available on this server">
            <Stack gap={4}>
              <Text size="sm">{dbStatus?.detail}</Text>
              <Text size="xs" c="dimmed">
                The iGEM Parts Registry does not offer a public download that a
                server can fetch automatically, so this database has to be
                installed by an administrator. Ask whoever runs this deployment.
              </Text>
            </Stack>
          </Alert>
        )}
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
            {/* "No matches" is only interpretable next to what was searched. The
                previous card said "No matching parts found for this sequence"
                against a three-sequence placeholder database, which reads as a
                statement about the Registry rather than about this server. */}
            {registry.unavailable_reason ? (
              <Alert color="yellow" title="Registry search did not run">
                {registry.unavailable_detail ??
                  registry.database?.detail ??
                  "No parts database is available on this server."}
              </Alert>
            ) : registry.matches.length === 0 ? (
              <InfoNote
                summary={`No homologs in the ${
                  registry.database?.n_parts?.toLocaleString() ?? "?"
                }-part snapshot on this server.`}
              >
                The search ran and returned nothing. That is a statement about
                this snapshot, not about the iGEM Registry as a whole — a part
                absent from the snapshot cannot be found here.
              </InfoNote>
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
            {/* Provenance of the snapshot, on every result. */}
            {registry.database && (
              <Text size="xs" c="dimmed">
                Searched {registry.database.n_parts.toLocaleString()} parts
                {registry.database.built_at
                  ? ` · snapshot ${registry.database.built_at.slice(0, 10)}`
                  : ""}
                {registry.database.source ? ` · ${registry.database.source}` : ""}
              </Text>
            )}
          </Stack>
        )}
        <Group>
          <Button
            onClick={handleRun}
            loading={loading}
            disabled={
              !activeSequence || loading || inputRefusal != null || unavailable
            }
            color="brand"
          >
            Search Registry ({variant})
          </Button>
        </Group>
      </Stack>
    </Paper>
  );
}
