import { Badge, Group, Text, Tooltip } from "@mantine/core";
import { Info } from "lucide-react";

interface MetricLabelProps {
  label: string;
  method?: "absolute" | "derived" | "direct";
  tooltip?: string;
}

const METHOD_COLOR: Record<NonNullable<MetricLabelProps["method"]>, string> = {
  absolute: "gray",
  derived: "teal",
  direct: "violet",
};

const METHOD_LABEL: Record<NonNullable<MetricLabelProps["method"]>, string> = {
  absolute: "absolute",
  derived: "derived",
  direct: "direct",
};

export function MetricLabel({ label, method, tooltip }: MetricLabelProps) {
  return (
    <Group gap={4} wrap="nowrap">
      <Text size="xs" c="dimmed">
        {label}
      </Text>
      {method && (
        <Badge size="xs" variant="light" color={METHOD_COLOR[method]}>
          {METHOD_LABEL[method]}
        </Badge>
      )}
      {tooltip && (
        <Tooltip label={tooltip} multiline maw={280} withArrow>
          <Info size={12} style={{ opacity: 0.55, flexShrink: 0 }} />
        </Tooltip>
      )}
    </Group>
  );
}
