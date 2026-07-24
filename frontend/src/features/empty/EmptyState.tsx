import { Center, Paper, Stack, Text, Title } from "@mantine/core";

interface EmptyStateProps {
  minHeight?: string;
}

export function EmptyState({ minHeight }: EmptyStateProps) {
  return (
    <Center p="xl" style={{ minHeight }}>
      <Paper maw={420} w="100%" p="xl" ta="center">
        <Stack align="center" gap="sm">
          <Text
            fw={700}
            c="brand"
            tt="uppercase"
            style={{ fontSize: "0.66rem", letterSpacing: "0.08em" }}
          >
            Get started
          </Text>
          <Title order={3} ta="center">
            Start a New Analysis
          </Title>
          <Text size="sm" c="dimmed" ta="center">
            Use the <strong>+ New</strong> button in the sidebar to create your
            first analysis. Paste a protein sequence, then run prediction,
            optimization, visualization, registry search, and procurement
            steps from the cards below.
          </Text>
        </Stack>
      </Paper>
    </Center>
  );
}
