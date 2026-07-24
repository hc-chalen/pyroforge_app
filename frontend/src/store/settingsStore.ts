import { create } from "zustand";
import { persist } from "zustand/middleware";

export type ColorScheme = "light" | "dark";

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
      organism: "ecoli",
      maxHits: 10,
      colorScheme: "light",
      navbarWidth: NAVBAR_DEFAULT_WIDTH,
      setMaxMutations: (maxMutations) => set({ maxMutations }),
      setPerplexityThreshold: (perplexityThreshold) =>
        set({ perplexityThreshold }),
      setOrganism: (organism) => set({ organism }),
      setMaxHits: (maxHits) => set({ maxHits }),
      setColorScheme: (colorScheme) => set({ colorScheme }),
      toggleColorScheme: () =>
        set({
          colorScheme: get().colorScheme === "light" ? "dark" : "light",
        }),
      setNavbarWidth: (width) => set({ navbarWidth: clampNavbarWidth(width) }),
    }),
    { name: "pyroforge-settings" },
  ),
);
