import { Group, Paper, Stack, Text, Textarea } from "@mantine/core";
import { CardHeader } from "@/components/ui/CardHeader";
import { WithheldNotice } from "@/components/ui/Withheld";
import { useAnalysisStore } from "@/store/analysisStore";
import { SEQ_MAX_LENGTH, withheldSequence } from "@/utils/answerability";

const PLACEHOLDER =
  "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK";

interface SequenceInputCardProps {
  analysisId: string;
  sequence: string;
}

export function SequenceInputCard({
  analysisId,
  sequence,
}: SequenceInputCardProps) {
  const updateSequence = useAnalysisStore((s) => s.updateSequence);
  const length = sequence.trim().length;
  // Refuse at the input rather than letting each downstream card fail its own
  // request with a backend error message.
  const refusal = withheldSequence(sequence);

  return (
    <Paper>
      <Stack gap="sm">
        <CardHeader eyebrow="Input" title="Sequence Input" />
        <Textarea
          aria-label="Protein sequence"
          placeholder={PLACEHOLDER}
          minRows={6}
          autosize
          error={refusal != null}
          value={sequence}
          onChange={(e) => updateSequence(analysisId, e.currentTarget.value)}
        />
        {refusal ? (
          <WithheldNotice withheld={refusal} />
        ) : (
          <Group justify="space-between" gap="xs">
            <Text size="xs" c="dimmed">
              Amino acids only, uppercased automatically.
            </Text>
            {length > 0 && (
              <Text size="xs" c="dimmed" ff="monospace">
                {length} / {SEQ_MAX_LENGTH} aa
              </Text>
            )}
          </Group>
        )}
      </Stack>
    </Paper>
  );
}
