import { Box, Group, Text } from "@mantine/core";
import type { ReactNode } from "react";

interface CardHeaderProps {
  /** Small uppercase category label rendered above the title. */
  eyebrow: string;
  title: ReactNode;
  /** Optional element (usually a status Badge) pinned to the top-right. */
  status?: ReactNode;
}

export function CardHeader({ eyebrow, title, status }: CardHeaderProps) {
  return (
    <Group justify="space-between" align="flex-start" wrap="nowrap" gap="sm">
      <Box style={{ minWidth: 0 }}>
        <Text
          fw={700}
          c="brand"
          tt="uppercase"
          style={{ fontSize: "0.66rem", letterSpacing: "0.08em" }}
        >
          {eyebrow}
        </Text>
        <Text fw={600} mt={2} style={{ fontSize: "1rem", lineHeight: 1.25 }}>
          {title}
        </Text>
      </Box>
      {status && <Box style={{ flexShrink: 0 }}>{status}</Box>}
    </Group>
  );
}
