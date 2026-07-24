import {
  Button,
  Divider,
  Group,
  Modal,
  NumberInput,
  Select,
  Stack,
  Text,
} from "@mantine/core";
import { ModelDesignPanel } from "@/features/prediction/ModelDesignPanel";
import { useSettingsStore } from "@/store/settingsStore";

interface SettingsModalProps {
  opened: boolean;
  onClose: () => void;
}

export function SettingsModal({ opened, onClose }: SettingsModalProps) {
  const maxMutations = useSettingsStore((s) => s.maxMutations);
  const perplexityThreshold = useSettingsStore((s) => s.perplexityThreshold);
  const organism = useSettingsStore((s) => s.organism);
  const maxHits = useSettingsStore((s) => s.maxHits);
  const setMaxMutations = useSettingsStore((s) => s.setMaxMutations);
  const setPerplexityThreshold = useSettingsStore(
    (s) => s.setPerplexityThreshold,
  );
  const setOrganism = useSettingsStore((s) => s.setOrganism);
  const setMaxHits = useSettingsStore((s) => s.setMaxHits);

  return (
    <Modal
      opened={opened}
      onClose={onClose}
      title="Pipeline Settings"
      centered
      size="lg"
    >
      <Stack gap="md">
        <NumberInput
          label="Max mutations"
          description="Maximum mutations for optimization (1–20)"
          value={maxMutations}
          onChange={(v) => setMaxMutations(Number(v) || 5)}
          min={1}
          max={20}
        />
        <NumberInput
          label="Perplexity threshold"
          description="ESM-2 perplexity guardrail (0–1)"
          value={perplexityThreshold}
          onChange={(v) => setPerplexityThreshold(Number(v) || 0.05)}
          min={0}
          max={1}
          step={0.01}
          decimalScale={2}
        />
        <Select
          label="Expression organism"
          description="Codon optimization target"
          value={organism}
          onChange={(v) => setOrganism(v ?? "ecoli")}
          data={[
            { value: "ecoli", label: "E. coli" },
            { value: "yeast", label: "S. cerevisiae" },
            { value: "human", label: "Human" },
          ]}
        />
        <NumberInput
          label="Registry max hits"
          description="Maximum iGEM parts matches (1–50)"
          value={maxHits}
          onChange={(v) => setMaxHits(Number(v) || 10)}
          min={1}
          max={50}
        />
        <Divider
          label="Model & algorithm details"
          labelPosition="left"
        />
        <Text size="xs" c="dimmed">
          How PyroForge predicts Tm, ΔG and ΔΔG, and the NAS-selected algorithm
          stack. Reference only — nothing here needs changing to run the
          pipeline.
        </Text>
        <ModelDesignPanel />
        <Group justify="flex-end">
          <Button onClick={onClose}>Done</Button>
        </Group>
      </Stack>
    </Modal>
  );
}
