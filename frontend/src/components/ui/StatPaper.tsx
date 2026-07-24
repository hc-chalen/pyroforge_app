import { Paper } from "@mantine/core";
import type { ReactNode } from "react";

interface StatPaperProps {
  children: ReactNode;
}

export function StatPaper({ children }: StatPaperProps) {
  return (
    <Paper
      p="sm"
      radius="md"
      style={{
        backgroundColor: "var(--pyro-card-nested-bg)",
        border: "1px solid var(--mantine-color-default-border)",
      }}
    >
      {children}
    </Paper>
  );
}
