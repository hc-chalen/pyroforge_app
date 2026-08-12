import {
  Alert,
  Badge,
  Button,
  Group,
  Paper,
  SimpleGrid,
  Stack,
  Text,
  Title,
} from "@mantine/core";
import { GitCompare } from "lucide-react";
import { useState } from "react";
import { InfoNote } from "@/components/ui/InfoNote";
import { StatPaper } from "@/components/ui/StatPaper";
import { compareEmbeddings } from "@/services/embed";
import type { OptimizationResult, StructureComparison } from "@/types/api";
import {
  formatSimilarityPercent,
  getOriginalSequence,
  getOptimizedSequence,
  similarityColor,
} from "@/utils/sequenceVariant";

interface StructureCompareCardProps {
  sequence: string;
  optimization?: OptimizationResult;
  comparison?: StructureComparison;
  onComparison?: (comparison: StructureComparison) => void;
}

export function StructureCompareCard({
  sequence,
  optimization,
  comparison,
  onComparison,
}: StructureCompareCardProps) {
  const [loading, setLoading] = useState(false);
  const [localError, setLocalError] = useState<string | undefined>();

  const original = getOriginalSequence(sequence, optimization);
  const optimized = getOptimizedSequence(optimization);

  if (!optimized) return null;

  const data = comparison ?? optimization?.structure_comparison;

  async function handleCompare() {
    if (!optimized) return;
    setLoading(true);
    setLocalError(undefined);
    try {
      const result = await compareEmbeddings({
        sequence_original: original,
        sequence_optimized: optimized,
      });
      onComparison?.(result);
    } catch (err) {
      setLocalError(
        err instanceof Error ? err.message : "Structure comparison failed",
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <Paper withBorder p="md" bg="var(--pyro-card-nested-bg)">
      <Stack gap="sm">
        <Group justify="space-between">
          <Title order={5}>
            <GitCompare
              size={18}
              style={{
                display: "inline",
                verticalAlign: "middle",
                marginRight: 8,
              }}
            />
            Compare Structures
          </Title>
          {data && (
            <Badge color={similarityColor(data.similarity)} size="lg">
              {formatSimilarityPercent(data.similarity)} similar
            </Badge>
          )}
        </Group>
        <InfoNote summary="ESM-2 embedding distance between the original and optimized sequences.">
          High similarity suggests the variant preserves the protein's overall
          structural semantics; low similarity warrants manual review before
          wet-lab use. It is a similarity of representations, not of solved
          structures.
        </InfoNote>
        {localError && (
          <Alert color="red" title="Error">
            {localError}
          </Alert>
        )}
        {data && (
          <SimpleGrid cols={{ base: 1, sm: 2, md: 4 }}>
            <StatPaper>
              <Text size="xs" c="dimmed">
                Similarity
              </Text>
              <Text fw={700} size="lg">
                {formatSimilarityPercent(data.similarity)}
              </Text>
            </StatPaper>
            <StatPaper>
              <Text size="xs" c="dimmed">
                Cosine similarity
              </Text>
              <Text fw={700}>{data.cosine_similarity.toFixed(4)}</Text>
            </StatPaper>
            <StatPaper>
              <Text size="xs" c="dimmed">
                Mean |Δembedding|
              </Text>
              <Text fw={700}>{data.mean_absolute_difference.toFixed(4)}</Text>
            </StatPaper>
            <StatPaper>
              <Text size="xs" c="dimmed">
                Sequence identity
              </Text>
              <Text fw={700}>
                {data.sequence_identity != null
                  ? `${(data.sequence_identity * 100).toFixed(1)}%`
                  : "—"}
              </Text>
            </StatPaper>
            <StatPaper>
              <Text size="xs" c="dimmed">
                Mutations
              </Text>
              <Text fw={700}>{data.mutation_count ?? "—"}</Text>
            </StatPaper>
            <StatPaper>
              <Text size="xs" c="dimmed">
                Mean |emb| (original)
              </Text>
              <Text fw={700}>{data.embedding_mean_abs_original.toFixed(4)}</Text>
            </StatPaper>
            <StatPaper>
              <Text size="xs" c="dimmed">
                Mean |emb| (optimized)
              </Text>
              <Text fw={700}>
                {data.embedding_mean_abs_optimized.toFixed(4)}
              </Text>
            </StatPaper>
            <StatPaper>
              <Text size="xs" c="dimmed">
                L1 distance
              </Text>
              <Text fw={700}>{data.l1_distance.toFixed(2)}</Text>
            </StatPaper>
          </SimpleGrid>
        )}
        <Group>
          <Button
            size="compact-sm"
            variant="light"
            onClick={handleCompare}
            loading={loading}
          >
            {data ? "Re-run comparison" : "Compare structures"}
          </Button>
        </Group>
      </Stack>
    </Paper>
  );
}
