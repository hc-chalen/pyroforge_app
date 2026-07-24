import { useCallback, useEffect, useRef, useState } from "react";
import {
  ActionIcon,
  Badge,
  Checkbox,
  Group,
  Text,
  TextInput,
  UnstyledButton,
} from "@mantine/core";
import { Pin, Trash2 } from "lucide-react";
import type { Analysis, AnalysisStatus } from "@/store/analysisStore";
import { useAnalysisStore } from "@/store/analysisStore";

const STATUS_COLORS: Record<AnalysisStatus, string> = {
  draft: "gray",
  running: "blue",
  complete: "green",
  error: "red",
};

const ACTION_WIDTH = 96;
const SWIPE_THRESHOLD = 48;

interface AnalysisListItemProps {
  analysis: Analysis;
  selected: boolean;
  checked: boolean;
  selectionMode: boolean;
  onSelect: () => void;
}

export function AnalysisListItem({
  analysis,
  selected,
  checked,
  selectionMode,
  onSelect,
}: AnalysisListItemProps) {
  const removeAnalysis = useAnalysisStore((s) => s.removeAnalysis);
  const renameAnalysis = useAnalysisStore((s) => s.renameAnalysis);
  const togglePin = useAnalysisStore((s) => s.togglePin);
  const toggleChecked = useAnalysisStore((s) => s.toggleChecked);

  const [offset, setOffset] = useState(0);
  const [dragging, setDragging] = useState(false);
  const [editing, setEditing] = useState(false);
  const [editName, setEditName] = useState(analysis.name);

  const draggingRef = useRef(false);

  const startX = useRef(0);
  const startOffset = useRef(0);
  const moved = useRef(false);
  const rowRef = useRef<HTMLDivElement>(null);

  const preview =
    analysis.sequence.length > 0
      ? analysis.sequence.slice(0, 24) +
        (analysis.sequence.length > 24 ? "…" : "")
      : "No sequence";

  const snapOpen = offset > SWIPE_THRESHOLD;
  const displayOffset = snapOpen ? ACTION_WIDTH : offset;

  const closeSwipe = useCallback(() => setOffset(0), []);

  useEffect(() => {
    setEditName(analysis.name);
  }, [analysis.name]);

  useEffect(() => {
    if (!selected) closeSwipe();
  }, [selected, closeSwipe]);

  useEffect(() => {
    const handleOutside = (e: MouseEvent | TouchEvent) => {
      if (rowRef.current && !rowRef.current.contains(e.target as Node)) {
        closeSwipe();
      }
    };
    document.addEventListener("mousedown", handleOutside);
    document.addEventListener("touchstart", handleOutside);
    return () => {
      document.removeEventListener("mousedown", handleOutside);
      document.removeEventListener("touchstart", handleOutside);
    };
  }, [closeSwipe]);

  const onDragStart = (clientX: number, target: EventTarget | null) => {
    if (editing || selectionMode) return;
    if (
      target instanceof HTMLElement &&
      target.closest("[data-no-swipe], input, label")
    ) {
      return;
    }
    draggingRef.current = true;
    setDragging(true);
    moved.current = false;
    startX.current = clientX;
    startOffset.current = snapOpen ? ACTION_WIDTH : offset;
  };

  const onDragMove = (clientX: number) => {
    if (!draggingRef.current) return;
    const delta = startX.current - clientX;
    if (Math.abs(delta) > 4) moved.current = true;
    const next = Math.max(0, Math.min(ACTION_WIDTH, startOffset.current + delta));
    setOffset(next);
  };

  const onDragEnd = () => {
    if (!draggingRef.current) return;
    draggingRef.current = false;
    setDragging(false);
    setOffset((current) => (current > SWIPE_THRESHOLD ? ACTION_WIDTH : 0));
  };

  const commitRename = () => {
    renameAnalysis(analysis.id, editName);
    setEditing(false);
  };

  const handleDelete = () => {
    closeSwipe();
    removeAnalysis(analysis.id);
  };

  const handlePin = () => {
    closeSwipe();
    togglePin(analysis.id);
  };

  return (
    <div
      ref={rowRef}
      style={{
        position: "relative",
        overflow: "hidden",
        borderRadius: "var(--mantine-radius-md)",
        border: selected
          ? "1px solid var(--mantine-color-brand-filled)"
          : "var(--pyro-item-border)",
        backgroundColor: "var(--pyro-item-bg)",
      }}
    >
      <Group
        gap={0}
        wrap="nowrap"
        style={{
          position: "absolute",
          top: 0,
          right: 0,
          bottom: 0,
          width: ACTION_WIDTH,
        }}
      >
        <ActionIcon
          variant="filled"
          color={analysis.pinned ? "brand" : "gray"}
          size="lg"
          radius={0}
          h="100%"
          w={ACTION_WIDTH / 2}
          aria-label={analysis.pinned ? "Unpin analysis" : "Pin analysis"}
          onClick={handlePin}
        >
          <Pin size={16} fill={analysis.pinned ? "currentColor" : "none"} />
        </ActionIcon>
        <ActionIcon
          variant="filled"
          color="red"
          size="lg"
          radius={0}
          h="100%"
          w={ACTION_WIDTH / 2}
          aria-label="Delete analysis"
          onClick={handleDelete}
        >
          <Trash2 size={16} />
        </ActionIcon>
      </Group>

      <div
        style={{
          transform: `translateX(-${displayOffset}px)`,
          transition: dragging ? "none" : "transform 0.2s ease",
          touchAction: "pan-y",
          backgroundColor: selected
            ? "var(--pyro-item-selected-bg)"
            : "var(--pyro-item-bg)",
          boxShadow: selected ? "var(--pyro-shadow-inset)" : undefined,
        }}
        onTouchStart={(e) => onDragStart(e.touches[0].clientX, e.target)}
        onTouchMove={(e) => onDragMove(e.touches[0].clientX)}
        onTouchEnd={onDragEnd}
        onMouseDown={(e) => {
          if (e.button !== 0) return;
          onDragStart(e.clientX, e.target);

          const onMouseMove = (ev: MouseEvent) => onDragMove(ev.clientX);
          const onMouseUp = () => {
            onDragEnd();
            window.removeEventListener("mousemove", onMouseMove);
            window.removeEventListener("mouseup", onMouseUp);
          };
          window.addEventListener("mousemove", onMouseMove);
          window.addEventListener("mouseup", onMouseUp);
        }}
      >
        <Group
          gap="xs"
          wrap="nowrap"
          align="flex-start"
          style={{ width: "100%", padding: "0.625rem 0.75rem" }}
        >
          {selectionMode && (
            <Checkbox
              checked={checked}
              onChange={() => toggleChecked(analysis.id)}
              onClick={(e) => e.stopPropagation()}
              onMouseDown={(e) => e.stopPropagation()}
              aria-label={`Select ${analysis.name} for bulk actions`}
              data-no-swipe
              style={{ flexShrink: 0, marginTop: 2 }}
            />
          )}
          <UnstyledButton
            onClick={() => {
              if (moved.current) {
                moved.current = false;
                return;
              }
              if (snapOpen) {
                closeSwipe();
                return;
              }
              if (selectionMode) {
                toggleChecked(analysis.id);
                return;
              }
              onSelect();
            }}
            onMouseDown={(e) => e.stopPropagation()}
            onDoubleClick={(e) => {
              e.stopPropagation();
              setEditing(true);
              setEditName(analysis.name);
            }}
            aria-label={`Open ${analysis.name}`}
            aria-current={selected ? "true" : undefined}
            style={{
              display: "block",
              flex: 1,
              minWidth: 0,
            }}
          >
          <Group justify="space-between" mb={4} wrap="nowrap" gap="xs">
            {editing ? (
              <TextInput
                size="xs"
                value={editName}
                autoFocus
                onClick={(e) => e.stopPropagation()}
                onChange={(e) => setEditName(e.currentTarget.value)}
                onBlur={commitRename}
                onKeyDown={(e) => {
                  if (e.key === "Enter") commitRename();
                  if (e.key === "Escape") {
                    setEditName(analysis.name);
                    setEditing(false);
                  }
                }}
                styles={{ input: { fontWeight: 600 } }}
              />
            ) : (
              <Group gap={4} wrap="nowrap" style={{ flex: 1, minWidth: 0 }}>
                {analysis.pinned && (
                  <Pin
                    size={12}
                    style={{ flexShrink: 0, color: "var(--mantine-color-brand-6)" }}
                    fill="currentColor"
                  />
                )}
                <Text size="sm" fw={600} truncate>
                  {analysis.name}
                </Text>
              </Group>
            )}
            <Badge
              size="xs"
              color={STATUS_COLORS[analysis.status]}
              aria-label={`Status: ${analysis.status}`}
              style={{ flexShrink: 0 }}
            >
              {analysis.status}
            </Badge>
          </Group>
          <Text size="xs" c="dimmed" ff="monospace" lineClamp={1}>
            {preview}
          </Text>
        </UnstyledButton>
        </Group>
      </div>
    </div>
  );
}
