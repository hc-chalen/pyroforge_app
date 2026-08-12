import {
  ActionIcon,
  Badge,
  Box,
  CopyButton,
  Group,
  ScrollArea,
  Stack,
  Text,
  Tooltip,
} from "@mantine/core";
import { Check, ChevronDown, ChevronUp, Copy, Download } from "lucide-react";
import { useMemo, useState } from "react";

const RESIDUES_PER_BLOCK = 10;
const BLOCKS_PER_LINE = 6; // 60 residues per line, the FASTA/UniProt convention

interface SequenceViewerProps {
  sequence: string;
  /** Shown above the sequence, e.g. "Optimized sequence". */
  label?: string;
  /** 0-based positions to mark (mutated sites). */
  highlights?: number[];
  /** Header used for FASTA copy/download; defaults to "sequence". */
  fastaName?: string;
  /** Lines shown before the block collapses; the rest is behind "Show all". */
  collapsedLines?: number;
  /** Extra element rendered in the header row, right of the length badge. */
  meta?: React.ReactNode;
}

function toFasta(sequence: string, name: string): string {
  const lines: string[] = [];
  for (let i = 0; i < sequence.length; i += 60) {
    lines.push(sequence.slice(i, i + 60));
  }
  return `>${name}\n${lines.join("\n")}\n`;
}

function download(filename: string, text: string) {
  const url = URL.createObjectURL(new Blob([text], { type: "text/plain" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function CopyIconButton({
  value,
  tooltip,
}: {
  value: string;
  tooltip: string;
}) {
  return (
    <CopyButton value={value} timeout={1500}>
      {({ copied, copy }) => (
        <Tooltip label={copied ? "Copied" : tooltip} withArrow>
          <ActionIcon
            variant="subtle"
            color={copied ? "green" : "gray"}
            onClick={copy}
            aria-label={tooltip}
          >
            {copied ? <Check size={15} /> : <Copy size={15} />}
          </ActionIcon>
        </Tooltip>
      )}
    </CopyButton>
  );
}

/**
 * Full-sequence display with position ruler, mutation highlighting and copy.
 *
 * Sequences were previously rendered as a two-line clamped `<Text>`, which for
 * anything past ~150 residues showed a prefix and silently hid the rest — and
 * since the shortlisted mutations are usually a handful of residues in a few
 * hundred, the visible prefix of an optimized sequence is very often identical
 * to the wild type. That made a real result look like "the optimizer did
 * nothing". Here the whole sequence is reachable (scroll + expand), the changed
 * positions are marked, and the residue numbering is on screen so a mutation
 * listed at position 214 can actually be found.
 */
export function SequenceViewer({
  sequence,
  label,
  highlights,
  fastaName = "sequence",
  collapsedLines = 4,
  meta,
}: SequenceViewerProps) {
  const [expanded, setExpanded] = useState(false);
  const highlightSet = useMemo(
    () => new Set(highlights ?? []),
    [highlights],
  );

  const lines = useMemo(() => {
    const perLine = RESIDUES_PER_BLOCK * BLOCKS_PER_LINE;
    const out: { start: number; blocks: { start: number; text: string }[] }[] = [];
    for (let i = 0; i < sequence.length; i += perLine) {
      const blocks: { start: number; text: string }[] = [];
      for (let j = i; j < Math.min(i + perLine, sequence.length); j += RESIDUES_PER_BLOCK) {
        blocks.push({ start: j, text: sequence.slice(j, j + RESIDUES_PER_BLOCK) });
      }
      out.push({ start: i, blocks });
    }
    return out;
  }, [sequence]);

  if (!sequence) return null;

  const collapsible = lines.length > collapsedLines;
  const visible = collapsible && !expanded ? lines.slice(0, collapsedLines) : lines;
  const gutterWidth = `${String(sequence.length).length + 1}ch`;

  return (
    <Stack gap={4}>
      <Group justify="space-between" gap="xs" wrap="nowrap">
        <Group gap={6} wrap="nowrap">
          {label && (
            <Text size="xs" fw={600}>
              {label}
            </Text>
          )}
          <Badge size="xs" variant="light" color="gray">
            {sequence.length} aa
          </Badge>
          {highlightSet.size > 0 && (
            <Badge size="xs" variant="light" color="brand">
              {highlightSet.size} substitution
              {highlightSet.size === 1 ? "" : "s"}
            </Badge>
          )}
          {meta}
        </Group>
        <Group gap={2} wrap="nowrap">
          <CopyIconButton value={sequence} tooltip="Copy sequence" />
          <CopyButton value={toFasta(sequence, fastaName)} timeout={1500}>
            {({ copied, copy }) => (
              <Tooltip label={copied ? "Copied" : "Copy as FASTA"} withArrow>
                <ActionIcon
                  variant="subtle"
                  color={copied ? "green" : "gray"}
                  onClick={copy}
                  aria-label="Copy as FASTA"
                  size="sm"
                >
                  <Text size="9px" fw={700} ff="monospace">
                    {copied ? "✓" : ">_"}
                  </Text>
                </ActionIcon>
              </Tooltip>
            )}
          </CopyButton>
          <Tooltip label="Download FASTA" withArrow>
            <ActionIcon
              variant="subtle"
              color="gray"
              aria-label="Download FASTA"
              onClick={() =>
                download(`${fastaName}.fasta`, toFasta(sequence, fastaName))
              }
            >
              <Download size={15} />
            </ActionIcon>
          </Tooltip>
          {collapsible && (
            <Tooltip label={expanded ? "Collapse" : "Show all"} withArrow>
              <ActionIcon
                variant="subtle"
                color="gray"
                aria-label={expanded ? "Collapse sequence" : "Show full sequence"}
                onClick={() => setExpanded((v) => !v)}
              >
                {expanded ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
              </ActionIcon>
            </Tooltip>
          )}
        </Group>
      </Group>
      <Box
        style={{
          border: "1px solid var(--mantine-color-default-border)",
          borderRadius: "var(--mantine-radius-sm)",
          background: "var(--pyro-input-bg)",
        }}
      >
        <ScrollArea.Autosize mah={expanded ? 320 : undefined} type="auto">
          <Box p="xs" style={{ minWidth: "fit-content" }}>
            {visible.map((line) => (
              <Box
                key={line.start}
                style={{
                  display: "flex",
                  gap: "0.75ch",
                  fontFamily: "'IBM Plex Mono', monospace",
                  fontSize: "0.75rem",
                  lineHeight: 1.6,
                  whiteSpace: "pre",
                }}
              >
                <Text
                  component="span"
                  c="dimmed"
                  ff="monospace"
                  fz="0.75rem"
                  style={{ width: gutterWidth, textAlign: "right", userSelect: "none" }}
                >
                  {line.start + 1}
                </Text>
                {line.blocks.map((block) => {
                  const marked = [...block.text].some((_, i) =>
                    highlightSet.has(block.start + i),
                  );
                  if (!marked) {
                    return <span key={block.start}>{block.text}</span>;
                  }
                  return (
                    <span key={block.start}>
                      {[...block.text].map((aa, i) => {
                        const pos = block.start + i;
                        return highlightSet.has(pos) ? (
                          <span
                            key={pos}
                            title={`Position ${pos + 1}: ${aa}`}
                            style={{
                              background: "var(--mantine-color-brand-light)",
                              color: "var(--mantine-color-brand-light-color)",
                              fontWeight: 700,
                              borderRadius: 2,
                            }}
                          >
                            {aa}
                          </span>
                        ) : (
                          <span key={pos}>{aa}</span>
                        );
                      })}
                    </span>
                  );
                })}
              </Box>
            ))}
            {collapsible && !expanded && (
              <Text
                size="xs"
                c="dimmed"
                mt={4}
                style={{ cursor: "pointer" }}
                onClick={() => setExpanded(true)}
              >
                … {lines.length - collapsedLines} more line
                {lines.length - collapsedLines === 1 ? "" : "s"} — show all
              </Text>
            )}
          </Box>
        </ScrollArea.Autosize>
      </Box>
    </Stack>
  );
}
