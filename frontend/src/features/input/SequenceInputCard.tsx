import { Paper, Stack, Text, Textarea } from "@mantine/core";
import { CardHeader } from "@/components/ui/CardHeader";
import { useAnalysisStore } from "@/store/analysisStore";

interface SequenceInputCardProps {
  analysisId: string;
  sequence: string;
}

export function SequenceInputCard({
  analysisId,
  sequence,
}: SequenceInputCardProps) {
  const updateSequence = useAnalysisStore((s) => s.updateSequence);

  return (
    <Paper>
      <Stack gap="sm">
        <CardHeader eyebrow="Input" title="Sequence Input" />
        <Text size="sm" c="dimmed">
          Paste an amino acid sequence. Input is automatically uppercased.
        </Text>
        <Textarea
          aria-label="Protein sequence"
          placeholder="MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK"
          minRows={6}
          autosize
          value={sequence}
          onChange={(e) => updateSequence(analysisId, e.currentTarget.value)}
        />
      </Stack>
    </Paper>
  );
}
