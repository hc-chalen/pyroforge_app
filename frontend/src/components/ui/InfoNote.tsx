import { Box, Collapse, Group, Text, UnstyledButton } from "@mantine/core";
import { useDisclosure } from "@mantine/hooks";
import { ChevronDown, Info, TriangleAlert } from "lucide-react";
import type { ReactNode } from "react";

type Tone = "muted" | "warning";

const TONE_COLOR: Record<Tone, string | undefined> = {
  muted: undefined,
  warning: "orange",
};

interface InfoNoteProps {
  /** One line, always visible. Keep it to a sentence. */
  summary: ReactNode;
  /** The long version, hidden until the user asks for it. */
  children?: ReactNode;
  tone?: Tone;
  /** Label on the disclosure toggle. */
  moreLabel?: string;
}

/**
 * A one-line note with the detail folded away behind "Why?".
 *
 * The cards used to carry their caveats as always-open Alerts. Every one of
 * them was true and worth saying, but six paragraphs of standing warning read
 * as decoration and get skipped — so the summary line stays, and the evidence
 * moves one click away where it is still there for anyone who looks.
 */
export function InfoNote({
  summary,
  children,
  tone = "muted",
  moreLabel = "Why?",
}: InfoNoteProps) {
  const [opened, { toggle }] = useDisclosure(false);
  const color = TONE_COLOR[tone];
  const Icon = tone === "warning" ? TriangleAlert : Info;

  return (
    <Box>
      <Group gap={6} wrap="nowrap" align="flex-start">
        <Icon
          size={13}
          color={color ? `var(--mantine-color-${color}-6)` : undefined}
          style={{ opacity: color ? 0.9 : 0.5, flexShrink: 0, marginTop: 2 }}
        />
        <Text size="xs" c={color ?? "dimmed"} style={{ flex: 1, minWidth: 0 }}>
          {summary}
          {children && (
            <>
              {" "}
              <UnstyledButton
                onClick={toggle}
                aria-expanded={opened}
                style={{ verticalAlign: "baseline" }}
              >
                <Text
                  span
                  size="xs"
                  c={color ?? "dimmed"}
                  td="underline"
                  style={{ whiteSpace: "nowrap" }}
                >
                  {opened ? "Hide" : moreLabel}
                  <ChevronDown
                    size={11}
                    style={{
                      display: "inline",
                      verticalAlign: "-1px",
                      marginLeft: 2,
                      transform: opened ? "rotate(180deg)" : undefined,
                    }}
                  />
                </Text>
              </UnstyledButton>
            </>
          )}
        </Text>
      </Group>
      {children && (
        <Collapse expanded={opened}>
          <Text size="xs" c="dimmed" pl={19} pt={6}>
            {children}
          </Text>
        </Collapse>
      )}
    </Box>
  );
}
