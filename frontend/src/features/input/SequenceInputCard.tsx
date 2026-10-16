import {
  Code,
  Group,
  Paper,
  SegmentedControl,
  Stack,
  Text,
  Textarea,
} from "@mantine/core";
import { CardHeader } from "@/components/ui/CardHeader";
import { WithheldNotice } from "@/components/ui/Withheld";
import { useAnalysisStore } from "@/store/analysisStore";
import type { InputFormat } from "@/types/api";
import {
  DNA_MAX_LENGTH,
  SEQ_MAX_LENGTH,
  withheldSequence,
} from "@/utils/answerability";

const AA_PLACEHOLDER =
  "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK";

const DNA_PLACEHOLDER =
  "ATGTCTAAAGGTGAAGAATTATTCACGGGTGTTGTCCCAATTTTGGTTGAATTAGATGGTGATGTTAATGGGCACAAATTTTCTGTCAGTGGAGAGGGTGAAGGTGATGCTACATACGGAAAGCTTACCCTTAAATTTATTTGCACTACTGGAAAACTACCTGTTCCATGGCCAACACTTGTCACTACTTTCTCTTATGGTGTTCAATGCTTTTCCCGTTATCCGGATCACATGAAACGGCATGACTTTTTCAAGAGTGCCATGCCCGAAGGTTATGTACAGGAACGCACTATATCTTTCAAAGATGACGGGAACTACAAGACGCGTGCTGAAGTCAAGTTTGAAGGTGATACCCTTGTTAATCGTATCGAGTTAAAAGGTATTGATTTTAAAGAAGATGGAAACATTCTCGGACACAAACTCGAGTACAACTATAACTCACACAATGTATACATCACGGCAGACAAACAAAAGAATGGAATCAAAGCTAACTTCAAAATTCGCCACAACATTGAAGATGGATCCGTTCAACTAGCAGACCATTATCAACAAAATACTCCAATTGGCGATGGCCCTGTCCTTTTACCAGACAACCATTACCTGTCGACACAATCTGTCCTTTCGAAAGATCCCAACGAAAAGCGTGACCACATGGTCCTTCTTGAGTTTGTAACTGCTGCTGGGATTACACATGGCATGGATGAGCTCTACAAATAA";

interface SequenceInputCardProps {
  analysisId: string;
  sequence: string;
  inputFormat?: InputFormat;
  dnaSequence?: string;
}

export function SequenceInputCard({
  analysisId,
  sequence,
  inputFormat = "aa",
  dnaSequence = "",
}: SequenceInputCardProps) {
  const updateSequence = useAnalysisStore((s) => s.updateSequence);
  const updateDnaSequence = useAnalysisStore((s) => s.updateDnaSequence);
  const updateInputFormat = useAnalysisStore((s) => s.updateInputFormat);

  const raw = inputFormat === "dna" ? dnaSequence : sequence;
  const length = raw.trim().length;
  const proteinLength = sequence.trim().length;
  // Refuse at the input rather than letting each downstream card fail its own
  // request with a backend error message.
  const refusal = withheldSequence(raw, inputFormat);
  const maxLen = inputFormat === "dna" ? DNA_MAX_LENGTH : SEQ_MAX_LENGTH;

  return (
    <Paper>
      <Stack gap="sm">
        <Group justify="space-between" align="flex-start" gap="sm" wrap="wrap">
          <CardHeader eyebrow="Input" title="Sequence Input" />
          <SegmentedControl
            size="xs"
            value={inputFormat}
            onChange={(v) =>
              updateInputFormat(analysisId, v as InputFormat)
            }
            data={[
              { label: "Amino acid", value: "aa" },
              { label: "DNA", value: "dna" },
            ]}
            aria-label="Input format"
          />
        </Group>
        <Textarea
          aria-label={
            inputFormat === "dna" ? "DNA sequence" : "Protein sequence"
          }
          placeholder={
            inputFormat === "dna" ? DNA_PLACEHOLDER : AA_PLACEHOLDER
          }
          minRows={6}
          autosize
          error={refusal != null}
          value={raw}
          onChange={(e) => {
            const value = e.currentTarget.value;
            if (inputFormat === "dna") {
              updateDnaSequence(analysisId, value);
            } else {
              updateSequence(analysisId, value);
            }
          }}
        />
        {refusal && <WithheldNotice withheld={refusal} />}
        {inputFormat === "dna" && proteinLength > 0 && refusal == null && (
          <Text size="xs" c="dimmed">
            Translated CDS · {proteinLength} aa
            {proteinLength <= 80 ? (
              <>
                : <Code style={{ wordBreak: "break-all" }}>{sequence}</Code>
              </>
            ) : null}
          </Text>
        )}
        <Group justify="space-between" gap="xs">
          <Text size="xs" c="dimmed">
            {inputFormat === "dna"
              ? `DNA (A/C/G/T) · translated to protein · 10–${SEQ_MAX_LENGTH} aa after ORF`
              : `Amino acids only · 10–${SEQ_MAX_LENGTH} residues`}
          </Text>
          {length > 0 && (
            <Text
              size="xs"
              c={length > maxLen ? "red" : "dimmed"}
              ff="monospace"
            >
              {inputFormat === "dna"
                ? `${length} / ${maxLen} nt`
                : `${length} / ${maxLen} aa`}
            </Text>
          )}
        </Group>
      </Stack>
    </Paper>
  );
}
