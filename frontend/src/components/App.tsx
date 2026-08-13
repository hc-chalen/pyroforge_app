import { Alert, AppShell, Box, Stack } from "@mantine/core";
import { useDisclosure, useMediaQuery } from "@mantine/hooks";
import { useCallback, useEffect } from "react";
import { Sidebar } from "@/components/layout/Sidebar";
import { TopBar } from "@/components/layout/TopBar";
import { NavbarResizeHandle } from "@/components/layout/NavbarResizeHandle";
import { SettingsModal } from "@/components/layout/SettingsModal";
import { EmptyState } from "@/features/empty/EmptyState";
import { SequenceInputCard } from "@/features/input/SequenceInputCard";
import { CompositionCard } from "@/features/composition/CompositionCard";
import { PredictionCard } from "@/features/prediction/PredictionCard";
import { OptimizationCard } from "@/features/optimization/OptimizationCard";
import { ThermalStructureCard } from "@/features/visualization/ThermalStructureCard";
import { ActiveVariantBar } from "@/components/ui/ActiveVariantBar";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { RegistryCard } from "@/features/registry/RegistryCard";
import { ProcurementCard } from "@/features/procurement/ProcurementCard";
import { useHealth } from "@/hooks/useHealth";
import { useAnalysisStore } from "@/store/analysisStore";
import { useSettingsStore, NAVBAR_DEFAULT_WIDTH } from "@/store/settingsStore";

const HEADER_HEIGHT = 64;
const PANEL_PADDING = "var(--mantine-spacing-xl)";

export function App() {
  const [settingsOpened, { open: openSettings, close: closeSettings }] =
    useDisclosure(false);
  const analyses = useAnalysisStore((s) => s.analyses);
  const ensureActiveSelection = useAnalysisStore((s) => s.ensureActiveSelection);
  const selectAnalysis = useAnalysisStore((s) => s.selectAnalysis);
  const selected = useAnalysisStore((s) =>
    s.analyses.find((a) => a.id === s.selectedId),
  );
  const navbarWidth = useSettingsStore((s) => s.navbarWidth) ?? NAVBAR_DEFAULT_WIDTH;
  const setNavbarWidth = useSettingsStore((s) => s.setNavbarWidth);
  const { error: healthError, failureKind } = useHealth();
  const isDesktop = useMediaQuery("(min-width: 48em)");
  const [mobileNavOpened, { toggle: toggleMobileNav, close: closeMobileNav, open: openMobileNav }] =
    useDisclosure(false);

  const handleSelectAnalysis = useCallback(
    (id: string) => {
      selectAnalysis(id);
      if (!isDesktop) closeMobileNav();
    },
    [closeMobileNav, isDesktop, selectAnalysis],
  );

  useEffect(() => {
    if (isDesktop) return;
    if (selected) closeMobileNav();
    else openMobileNav();
  }, [closeMobileNav, isDesktop, openMobileNav, selected?.id]);

  useEffect(() => {
    const unsub = useAnalysisStore.persist.onFinishHydration(() => {
      useAnalysisStore.getState().ensureActiveSelection();
    });
    useAnalysisStore.getState().ensureActiveSelection();
    return unsub;
  }, [ensureActiveSelection, analyses.length]);

  return (
    <>
      <AppShell
        navbar={{
          width: navbarWidth,
          breakpoint: "sm",
          collapsed: { mobile: !mobileNavOpened },
        }}
        header={{ height: HEADER_HEIGHT }}
        padding={0}
      >
        <AppShell.Navbar>
          <Box
            h="100%"
            style={{ position: "relative", overflow: "hidden" }}
          >
            <Box p={PANEL_PADDING} h="100%">
              <Sidebar onSelectAnalysis={handleSelectAnalysis} />
            </Box>
            {isDesktop && (
              <NavbarResizeHandle
                width={navbarWidth}
                onWidthChange={setNavbarWidth}
              />
            )}
          </Box>
        </AppShell.Navbar>

        <AppShell.Header>
          <TopBar
            onOpenSettings={openSettings}
            mobileNavOpened={mobileNavOpened}
            onToggleMobileNav={toggleMobileNav}
          />
        </AppShell.Header>

        <AppShell.Main>
          <Box p={PANEL_PADDING} pb="xl" style={{ minWidth: 0, overflowX: "hidden" }}>
            {/* A 429 is the server throttling this connection — it arrives *from*
                the stack, so the stack is up. Reporting it as "Backend
                unreachable" and telling the user to start the API sent them
                chasing a problem that does not exist. The two cases now read
                differently and only the genuine one is an error. */}
            {healthError && failureKind === "throttled" && (
              <Alert color="yellow" mb="md" title="Server is busy">
                {healthError} Your work is not affected — requests will go through
                again shortly. If it persists, fewer browser tabs open on this
                network will help.
              </Alert>
            )}
            {healthError && failureKind !== "throttled" && (
              <Alert color="red" mb="md" title="Backend unreachable">
                {healthError}. Expected the API at{" "}
                {import.meta.env.VITE_API_URL || window.location.origin}.
              </Alert>
            )}

            {!selected ? (
              <EmptyState minHeight="calc(100dvh - 64px - 2 * var(--mantine-spacing-xl))" />
            ) : (
              <Stack gap="md" maw={880} w="100%" mx="auto">
                <SequenceInputCard
                  analysisId={selected.id}
                  sequence={selected.sequence}
                />
                <CompositionCard sequence={selected.sequence} />
                <PredictionCard
                  analysisId={selected.id}
                  sequence={selected.sequence}
                  prediction={selected.prediction}
                  optimization={selected.optimization}
                />
                <OptimizationCard
                  analysisId={selected.id}
                  sequence={selected.sequence}
                  optimization={selected.optimization}
                />
                <ActiveVariantBar
                  analysisId={selected.id}
                  sequence={selected.sequence}
                  optimization={selected.optimization}
                />
                <ErrorBoundary title="3D visualization error">
                  <ThermalStructureCard
                    analysisId={selected.id}
                    sequence={selected.sequence}
                    prediction={selected.prediction}
                    optimization={selected.optimization}
                  />
                </ErrorBoundary>
                <RegistryCard
                  analysisId={selected.id}
                  sequence={selected.sequence}
                  optimization={selected.optimization}
                />
                <ProcurementCard
                  analysisId={selected.id}
                  sequence={selected.sequence}
                  optimization={selected.optimization}
                />
              </Stack>
            )}
          </Box>
        </AppShell.Main>
      </AppShell>

      <SettingsModal opened={settingsOpened} onClose={closeSettings} />
    </>
  );
}
