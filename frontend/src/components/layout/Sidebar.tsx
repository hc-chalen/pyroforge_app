import { useMemo, useState } from "react";
import {
  Box,
  Button,
  Checkbox,
  Group,
  ScrollArea,
  Stack,
  Text,
} from "@mantine/core";
import { Pin, Plus, Trash2 } from "lucide-react";
import { AnalysisListItem } from "./AnalysisListItem";
import { useAnalysisStore } from "@/store/analysisStore";

function sortAnalyses(analyses: ReturnType<typeof useAnalysisStore.getState>["analyses"]) {
  return [...analyses].sort((a, b) => {
    const aPinned = a.pinned ?? false;
    const bPinned = b.pinned ?? false;
    if (aPinned === bPinned) return 0;
    return aPinned ? -1 : 1;
  });
}

interface SidebarProps {
  onSelectAnalysis?: (id: string) => void;
}

export function Sidebar({ onSelectAnalysis }: SidebarProps) {
  const [selectionMode, setSelectionMode] = useState(false);
  const analyses = useAnalysisStore((s) => s.analyses);
  const selectedId = useAnalysisStore((s) => s.selectedId);
  const checkedIds = useAnalysisStore((s) => s.checkedIds);
  const addAnalysis = useAnalysisStore((s) => s.addAnalysis);
  const selectAnalysis = useAnalysisStore((s) => s.selectAnalysis);
  const selectAllChecked = useAnalysisStore((s) => s.selectAllChecked);
  const clearChecked = useAnalysisStore((s) => s.clearChecked);
  const removeChecked = useAnalysisStore((s) => s.removeChecked);
  const togglePinChecked = useAnalysisStore((s) => s.togglePinChecked);

  const sortedAnalyses = useMemo(() => sortAnalyses(analyses), [analyses]);
  const allChecked =
    sortedAnalyses.length > 0 &&
    sortedAnalyses.every((analysis) => checkedIds.includes(analysis.id));
  const someChecked = checkedIds.length > 0;
  const indeterminate = someChecked && !allChecked;

  const exitSelectionMode = () => {
    clearChecked();
    setSelectionMode(false);
  };

  const handleRemoveChecked = () => {
    removeChecked();
    setSelectionMode(false);
  };

  return (
    <Stack gap="sm" h="100%" style={{ minWidth: 0 }}>
      <Group justify="space-between" wrap="nowrap" align="flex-start">
        <Box>
          <Text
            fw={600}
            c="brand"
            tt="uppercase"
            style={{ fontSize: "0.6875rem", letterSpacing: "0.08em" }}
          >
            Workspace
          </Text>
          <Text fw={600} style={{ fontSize: "1rem", lineHeight: 1.25 }}>
            Analyses
          </Text>
        </Box>
        <Group gap="xs" wrap="nowrap">
          {sortedAnalyses.length > 0 && (
            <Button
              size="compact-sm"
              variant={selectionMode ? "filled" : "subtle"}
              color="brand"
              onClick={() =>
                selectionMode ? exitSelectionMode() : setSelectionMode(true)
              }
            >
              {selectionMode ? "Done" : "Select"}
            </Button>
          )}
          <Button
            size="compact-sm"
            variant="filled"
            color="brand"
            leftSection={<Plus size={14} />}
            aria-label="Add new analysis"
            onClick={addAnalysis}
          >
            New
          </Button>
        </Group>
      </Group>

      {selectionMode && sortedAnalyses.length > 0 && (
        <>
          <Checkbox
            checked={allChecked}
            indeterminate={indeterminate}
            onChange={() => (allChecked ? clearChecked() : selectAllChecked())}
            label={
              <Text size="xs" c="dimmed">
                {someChecked ? `${checkedIds.length} selected` : "Select all"}
              </Text>
            }
            styles={{ label: { paddingLeft: 6 } }}
          />
          {someChecked && (
            <Group gap="xs" wrap="nowrap">
              <Button
                size="compact-xs"
                variant="light"
                color="brand"
                leftSection={<Pin size={14} />}
                onClick={togglePinChecked}
                style={{ flex: 1 }}
              >
                Pin
              </Button>
              <Button
                size="compact-xs"
                variant="light"
                color="red"
                leftSection={<Trash2 size={14} />}
                onClick={handleRemoveChecked}
                style={{ flex: 1 }}
              >
                Delete
              </Button>
            </Group>
          )}
        </>
      )}

      <ScrollArea flex={1} type="auto" offsetScrollbars>
        <Stack gap="sm">
          {sortedAnalyses.length === 0 ? (
            <Text size="sm" c="dimmed">
              No analyses yet. Click + New to start.
            </Text>
          ) : (
            sortedAnalyses.map((analysis) => (
              <AnalysisListItem
                key={analysis.id}
                analysis={analysis}
                selected={analysis.id === selectedId}
                checked={checkedIds.includes(analysis.id)}
                selectionMode={selectionMode}
                onSelect={() =>
                  onSelectAnalysis
                    ? onSelectAnalysis(analysis.id)
                    : selectAnalysis(analysis.id)
                }
              />
            ))
          )}
        </Stack>
      </ScrollArea>
    </Stack>
  );
}
