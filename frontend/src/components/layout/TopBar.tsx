import { Box, Burger, Button, Group, Image, Text } from "@mantine/core";
import { Moon, Settings, Sun } from "lucide-react";
import { useAnalysisStore } from "@/store/analysisStore";
import { useSettingsStore } from "@/store/settingsStore";

interface TopBarProps {
  onOpenSettings: () => void;
  mobileNavOpened: boolean;
  onToggleMobileNav: () => void;
}

export function TopBar({
  onOpenSettings,
  mobileNavOpened,
  onToggleMobileNav,
}: TopBarProps) {
  const selected = useAnalysisStore((s) =>
    s.analyses.find((a) => a.id === s.selectedId),
  );
  const colorScheme = useSettingsStore((s) => s.colorScheme);
  const toggleColorScheme = useSettingsStore((s) => s.toggleColorScheme);

  return (
    <Group justify="space-between" h="100%" px="lg" wrap="nowrap" gap="sm">
      <Group gap="sm" wrap="nowrap" style={{ flex: 1, minWidth: 0 }}>
        <Burger
          opened={mobileNavOpened}
          onClick={onToggleMobileNav}
          hiddenFrom="sm"
          size="sm"
          aria-label="Toggle analyses list"
        />
        <Image
          src="/PyroForge_icon.png"
          alt="PyroForge logo"
          w={32}
          h={32}
          radius="sm"
          fit="contain"
          visibleFrom="xs"
        />
        <Box style={{ minWidth: 0 }} visibleFrom="sm">
          <Text fw={700} style={{ fontSize: "1.0625rem", lineHeight: 1.1 }}>
            PyroForge
          </Text>
          <Text
            c="dimmed"
            tt="uppercase"
            style={{ fontSize: "0.65rem", letterSpacing: "0.04em" }}
          >
            Protein Thermostability Platform
          </Text>
        </Box>
        {selected && (
          <Text
            size="sm"
            c="dimmed"
            fw={500}
            truncate
            style={{ flex: 1, minWidth: 0 }}
            hiddenFrom="sm"
          >
            {selected.name}
          </Text>
        )}
      </Group>

      {selected && (
        <Text
          size="sm"
          c="dimmed"
          fw={500}
          truncate
          style={{ flex: 1, textAlign: "center", minWidth: 0 }}
          visibleFrom="sm"
        >
          {selected.name}
        </Text>
      )}

      <Group gap="xs" wrap="nowrap" style={{ flexShrink: 0 }}>
        <Button
          variant="default"
          size="xs"
          leftSection={
            colorScheme === "dark" ? <Sun size={16} /> : <Moon size={16} />
          }
          onClick={toggleColorScheme}
          aria-label={
            colorScheme === "dark" ? "Switch to light mode" : "Switch to dark mode"
          }
        >
          <Box component="span" visibleFrom="xs">
            {colorScheme === "dark" ? "Light" : "Dark"}
          </Box>
        </Button>
        <Button
          variant="default"
          size="xs"
          leftSection={<Settings size={16} />}
          onClick={onOpenSettings}
          aria-label="Open settings"
        >
          <Box component="span" visibleFrom="xs">
            Settings
          </Box>
        </Button>
      </Group>
    </Group>
  );
}
