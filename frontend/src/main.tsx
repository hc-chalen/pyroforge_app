import { StrictMode, useEffect } from "react";
import { createRoot } from "react-dom/client";
import { MantineProvider } from "@mantine/core";
import { Notifications } from "@mantine/notifications";
import "@mantine/core/styles.css";
import "@mantine/notifications/styles.css";
import "@fontsource/ibm-plex-sans/400.css";
import "@fontsource/ibm-plex-sans/500.css";
import "@fontsource/ibm-plex-sans/600.css";
import "@fontsource/ibm-plex-sans/700.css";
import "@fontsource/ibm-plex-mono/400.css";
import "@fontsource/ibm-plex-mono/500.css";
import "@fontsource/ibm-plex-mono/600.css";
import { App } from "@/components/App";
import { cssVariablesResolver, theme } from "@/theme";
import { useSettingsStore } from "@/store/settingsStore";

function Root() {
  const colorScheme = useSettingsStore((s) => s.colorScheme);

  useEffect(() => {
    document.documentElement.setAttribute("data-mantine-color-scheme", colorScheme);
  }, [colorScheme]);

  return (
    <MantineProvider
      theme={theme}
      defaultColorScheme={colorScheme}
      cssVariablesResolver={cssVariablesResolver}
      forceColorScheme={colorScheme}
    >
      <Notifications position="top-right" />
      <App />
    </MantineProvider>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <Root />
  </StrictMode>,
);
