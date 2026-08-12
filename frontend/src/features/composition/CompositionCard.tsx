import { Badge, Paper, SimpleGrid, Stack, Text } from "@mantine/core";
import { useMemo } from "react";
import { CardHeader } from "@/components/ui/CardHeader";
import { InfoNote } from "@/components/ui/InfoNote";
import { MetricLabel } from "@/components/ui/MetricLabel";
import { StatPaper } from "@/components/ui/StatPaper";
import { computeCompositionDescriptors } from "@/utils/composition";

const RISK_COLOR = { low: "gray", elevated: "yellow", high: "orange" } as const;
const RISK_LABEL = {
  low: "HCN risk: low",
  elevated: "HCN risk: elevated",
  high: "HCN risk: high",
} as const;

interface CompositionCardProps {
  sequence: string;
}

function Stat({
  label,
  tooltip,
  value,
}: {
  label: string;
  tooltip: string;
  value: string;
}) {
  return (
    <StatPaper>
      <MetricLabel label={label} method="derived" tooltip={tooltip} />
      <Text fw={600} ff="monospace" mt={4} style={{ fontSize: "1.05rem" }}>
        {value}
      </Text>
    </StatPaper>
  );
}

/**
 * Composition descriptors — atom counting on the sequence, nothing more.
 *
 * Sits directly under the input because it needs no backend call and no model:
 * it is a readout of what was typed, kept visually separate from the predicted
 * cards below so the two are never mistaken for each other.
 */
export function CompositionCard({ sequence }: CompositionCardProps) {
  const d = useMemo(() => computeCompositionDescriptors(sequence), [sequence]);
  if (!d) return null;

  const pct = (x: number) => `${(x * 100).toFixed(1)}%`;
  const r = d.ratios;

  return (
    <Paper>
      <Stack gap="sm">
        <CardHeader
          eyebrow="Descriptors"
          title="Elemental & Pyrolysis Composition"
          status={
            <Badge
              size="sm"
              variant="light"
              color={RISK_COLOR[d.hcnRisk]}
            >
              {RISK_LABEL[d.hcnRisk]}
            </Badge>
          }
        />

        <SimpleGrid cols={{ base: 2, sm: 4 }} spacing="xs">
          <Stat
            label="N/C"
            tooltip="Atomic nitrogen-to-carbon ratio of the peptide chain."
            value={r.nc.toFixed(3)}
          />
          <Stat
            label="O/C"
            tooltip="Atomic oxygen-to-carbon ratio, including the terminal water."
            value={r.oc.toFixed(3)}
          />
          <Stat
            label="H/C"
            tooltip="Atomic hydrogen-to-carbon ratio."
            value={r.hc.toFixed(3)}
          />
          <Stat
            label="P/C"
            tooltip="Canonical residues carry no phosphorus, so P/C is 0 unless the protein is phosphorylated — which the sequence alone cannot show."
            value={r.pc.toFixed(3)}
          />
        </SimpleGrid>

        <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="xs">
          <Stat
            label="N-rich residues"
            tooltip="Fraction of R, K, H, Q, N, W — the side chains that carry most of the nitrogen."
            value={pct(d.nRichFraction)}
          />
          <Stat
            label="Char precursors"
            tooltip="Fraction of aromatic F, Y, W plus His — ring systems associated with graphitization."
            value={pct(d.charPrecursorFraction)}
          />
        </SimpleGrid>

        {d.hcnRisk !== "low" && (
          <InfoNote
            tone="warning"
            summary={`Nitrogen content is high (N/C ${r.nc.toFixed(3)}, S/C ${r.sc.toFixed(3)}).`}
          >
            At high pyrolysis temperatures, nitrogen-rich feedstock tends to
            route through HCN and N₂ rather than staying in the char. This flag
            is a threshold on the counted N/C ratio, not a yield prediction.
          </InfoNote>
        )}

        <InfoNote summary="Descriptors, not predictions — computed by atom counting from the sequence.">
          These numbers come straight from residue formulas and letter counts.
          No model, no training data, and no validated pyrolysis dataset stands
          behind them. Read them as a description of the input composition, and
          do not treat the HCN flag or the fractions as predicted outcomes.
        </InfoNote>
      </Stack>
    </Paper>
  );
}
