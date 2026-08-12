import { Badge, Box, Group, Text, UnstyledButton } from "@mantine/core";
import { useDisclosure } from "@mantine/hooks";
import { Ban } from "lucide-react";
import type { Withheld } from "@/utils/answerability";

interface WithheldValueProps {
  withheld: Withheld;
  /** The number the model did produce, if any, pre-formatted. */
  value?: string | null;
}

/**
 * Stands in for a metric the models cannot honestly answer.
 *
 * Where an extrapolated value exists it is not deleted, just not shown by
 * default: the user has to click through a label that says what they are about
 * to see. That keeps "no answer" as the default reading of the card.
 */
export function WithheldValue({ withheld, value }: WithheldValueProps) {
  const [shown, { toggle }] = useDisclosure(false);
  const canReveal = withheld.extrapolated && value != null;

  return (
    <Box>
      <Group gap={6} wrap="nowrap" align="center">
        <Text fw={700} size="lg" ff="monospace" c="dimmed">
          {shown && canReveal ? value : "—"}
        </Text>
        <Badge size="xs" variant="light" color={shown ? "orange" : "gray"}>
          {shown ? "extrapolated" : "no answer"}
        </Badge>
      </Group>
      <Text size="xs" c="dimmed" mt={2}>
        {withheld.reason}
      </Text>
      {canReveal && (
        <UnstyledButton onClick={toggle} mt={4}>
          <Text span size="xs" c="dimmed" td="underline">
            {shown ? "Hide extrapolated value" : "Show extrapolated value anyway"}
          </Text>
        </UnstyledButton>
      )}
    </Box>
  );
}

/**
 * Card-level refusal: the step cannot run at all on this input, so say that
 * once instead of letting the user press a button that will fail.
 */
export function WithheldNotice({ withheld }: { withheld: Withheld }) {
  return (
    <Group
      gap={8}
      wrap="nowrap"
      align="flex-start"
      p="xs"
      style={{
        borderRadius: "var(--mantine-radius-sm)",
        border: "1px solid var(--mantine-color-default-border)",
        backgroundColor: "var(--pyro-card-nested-bg)",
      }}
    >
      <Ban size={14} style={{ opacity: 0.6, flexShrink: 0, marginTop: 1 }} />
      <Text size="xs" style={{ flex: 1, minWidth: 0 }}>
        <Text span size="xs" fw={600}>
          {withheld.headline}.
        </Text>{" "}
        <Text span size="xs" c="dimmed">
          {withheld.reason}
        </Text>
      </Text>
    </Group>
  );
}
