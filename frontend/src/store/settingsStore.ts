import { create } from "zustand";
import { persist } from "zustand/middleware";

export type ColorScheme = "light" | "dark";

/** Organisms the backend has a codon-adaptiveness table for (app.services.codon). */
export const ORGANISMS = [
  { value: "ecoli", label: "E. coli" },
  { value: "yeast", label: "S. cerevisiae" },
] as const;

export const DEFAULT_ORGANISM = "ecoli";

/** Keep the stored organism to one the backend supports — it rejects the rest. */
function coerceOrganism(value: unknown): string {
  return ORGANISMS.some((o) => o.value === value)
    ? (value as string)
    : DEFAULT_ORGANISM;
}

export const NAVBAR_MIN_WIDTH = 200;
export const NAVBAR_MAX_WIDTH = 480;
export const NAVBAR_DEFAULT_WIDTH = 280;

function clampNavbarWidth(width: number) {
  return Math.round(
    Math.min(NAVBAR_MAX_WIDTH, Math.max(NAVBAR_MIN_WIDTH, width)),
  );
}

interface SettingsState {
  maxMutations: number;
  perplexityThreshold: number;
  organism: string;
  maxHits: number;
  colorScheme: ColorScheme;
  navbarWidth: number;
  setMaxMutations: (value: number) => void;
  setPerplexityThreshold: (value: number) => void;
  setOrganism: (value: string) => void;
  setMaxHits: (value: number) => void;
  setColorScheme: (scheme: ColorScheme) => void;
  toggleColorScheme: () => void;
  setNavbarWidth: (width: number) => void;
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set, get) => ({
      maxMutations: 5,
      perplexityThreshold: 0.05,
      organism: DEFAULT_ORGANISM,
      maxHits: 10,
      colorScheme: "light",
      navbarWidth: NAVBAR_DEFAULT_WIDTH,
      setMaxMutations: (maxMutations) => set({ maxMutations }),
      setPerplexityThreshold: (perplexityThreshold) =>
        set({ perplexityThreshold }),
      setOrganism: (organism) => set({ organism: coerceOrganism(organism) }),
      setMaxHits: (maxHits) => set({ maxHits }),
      setColorScheme: (colorScheme) => set({ colorScheme }),
      toggleColorScheme: () =>
        set({
          colorScheme: get().colorScheme === "light" ? "dark" : "light",
        }),
      setNavbarWidth: (width) => set({ navbarWidth: clampNavbarWidth(width) }),
    }),
    {
      name: "pyroforge-settings",
      // A previously persisted organism may no longer be supported (e.g. "human",
      // which silently produced E. coli DNA); drop it back to the default on load.
      merge: (persisted, current) => {
        const merged = { ...current, ...(persisted as Partial<SettingsState>) };
        return { ...merged, organism: coerceOrganism(merged.organism) };
      },
    },
  ),
);
