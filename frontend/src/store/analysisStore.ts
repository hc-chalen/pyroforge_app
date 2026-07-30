import { create } from "zustand";
import { persist } from "zustand/middleware";
import { clearFoldPdb, setFoldPdb } from "@/store/foldPdbCache";
import type {
  FoldResult,
  HeatmapResponse,
  OptimizationResult,
  PredictResponse,
  ProcureResponse,
  RegistrySearchResponse,
  SequenceVariant,
  StructureComparison,
  VariantResults,
} from "@/types/api";

// crypto.randomUUID() only exists in a secure context (HTTPS or localhost).
// When the app is served over plain http:// (e.g. a LAN/Tailscale IP), fall
// back to a manual UUIDv4 so creating an analysis still works.
function genId(): string {
  const c = globalThis.crypto;
  if (c && typeof c.randomUUID === "function") return c.randomUUID();
  if (c && typeof c.getRandomValues === "function") {
    const b = c.getRandomValues(new Uint8Array(16));
    b[6] = (b[6] & 0x0f) | 0x40;
    b[8] = (b[8] & 0x3f) | 0x80;
    const h = [...b].map((x) => x.toString(16).padStart(2, "0"));
    return `${h[0]}${h[1]}${h[2]}${h[3]}-${h[4]}${h[5]}-${h[6]}${h[7]}-${h[8]}${h[9]}-${h[10]}${h[11]}${h[12]}${h[13]}${h[14]}${h[15]}`;
  }
  return `id-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export type AnalysisStatus = "draft" | "running" | "complete" | "error";

export interface Analysis {
  id: string;
  name: string;
  sequence: string;
  status: AnalysisStatus;
  pinned?: boolean;
  sequenceVariant?: SequenceVariant;
  prediction?: PredictResponse;
  optimization?: OptimizationResult;
  structureComparison?: StructureComparison;
  /** @deprecated use variantResults */
  fold?: FoldResult;
  heatmap?: HeatmapResponse;
  /** @deprecated use variantResults */
  registry?: RegistrySearchResponse;
  /** @deprecated use variantResults */
  procurement?: ProcureResponse;
  variantResults?: Partial<Record<SequenceVariant, VariantResults>>;
  error?: string;
}

interface AnalysisState {
  analyses: Analysis[];
  selectedId: string | null;
  checkedIds: string[];
  addAnalysis: () => void;
  selectAnalysis: (id: string) => void;
  toggleChecked: (id: string) => void;
  selectAllChecked: () => void;
  clearChecked: () => void;
  removeChecked: () => void;
  togglePinChecked: () => void;
  ensureActiveSelection: () => void;
  removeAnalysis: (id: string) => void;
  renameAnalysis: (id: string, name: string) => void;
  togglePin: (id: string) => void;
  updateSequence: (id: string, sequence: string) => void;
  setStatus: (id: string, status: AnalysisStatus) => void;
  setError: (id: string, error: string | undefined) => void;
  // Each setter records the sequence the result was computed on, so a later edit to
  // the input can be detected instead of silently invalidating the displayed numbers.
  setPrediction: (
    id: string,
    prediction: PredictResponse,
    variant: SequenceVariant,
    sequence: string,
  ) => void;
  setOptimization: (id: string, optimization: OptimizationResult) => void;
  setSequenceVariant: (id: string, variant: SequenceVariant) => void;
  setStructureComparison: (id: string, comparison: StructureComparison) => void;
  setFold: (
    id: string,
    fold: FoldResult,
    variant: SequenceVariant,
    sequence: string,
  ) => void;
  setHeatmap: (id: string, heatmap: HeatmapResponse) => void;
  setRegistry: (
    id: string,
    registry: RegistrySearchResponse,
    variant: SequenceVariant,
    sequence: string,
  ) => void;
  setProcurement: (
    id: string,
    procurement: ProcureResponse,
    variant: SequenceVariant,
    sequence: string,
  ) => void;
  getSelected: () => Analysis | undefined;
}

function nextAnalysisName(analyses: Analysis[]): string {
  const used = new Set(
    analyses.map((a) => {
      const match = /^Analysis (\d+)$/.exec(a.name);
      return match ? Number(match[1]) : 0;
    }),
  );
  let n = 1;
  while (used.has(n)) n += 1;
  return `Analysis ${n}`;
}

export const useAnalysisStore = create<AnalysisState>()(
  persist(
    (set, get) => ({
      analyses: [],
      selectedId: null,
      checkedIds: [],

      addAnalysis: () => {
        const id = genId();
        const name = nextAnalysisName(get().analyses);
        const analysis: Analysis = {
          id,
          name,
          sequence: "",
          status: "draft",
        };
        set((state) => ({
          analyses: [...state.analyses, analysis],
          selectedId: id,
        }));
      },

      selectAnalysis: (id) => set({ selectedId: id }),

      toggleChecked: (id) =>
        set((state) => ({
          checkedIds: state.checkedIds.includes(id)
            ? state.checkedIds.filter((checkedId) => checkedId !== id)
            : [...state.checkedIds, id],
        })),

      selectAllChecked: () =>
        set((state) => ({
          checkedIds: state.analyses.map((a) => a.id),
        })),

      clearChecked: () => set({ checkedIds: [] }),

      removeChecked: () => {
        const { checkedIds } = get();
        if (checkedIds.length === 0) return;
        for (const id of checkedIds) {
          clearFoldPdb(id);
        }
        set((state) => {
          const toRemove = new Set(checkedIds);
          const analyses = state.analyses.filter((a) => !toRemove.has(a.id));
          const selectedId =
            state.selectedId && toRemove.has(state.selectedId)
              ? (analyses[0]?.id ?? null)
              : state.selectedId;
          return { analyses, selectedId, checkedIds: [] };
        });
      },

      togglePinChecked: () => {
        const { checkedIds } = get();
        if (checkedIds.length === 0) return;
        const checked = new Set(checkedIds);
        set((state) => ({
          analyses: state.analyses.map((a) =>
            checked.has(a.id) ? { ...a, pinned: !a.pinned } : a,
          ),
        }));
      },

      ensureActiveSelection: () => {
        const { analyses, selectedId, checkedIds } = get();
        if (analyses.length === 0) {
          if (selectedId !== null || checkedIds.length > 0) {
            set({ selectedId: null, checkedIds: [] });
          }
          return;
        }
        const validSelected = analyses.some((a) => a.id === selectedId);
        const validChecked = checkedIds.filter((id) =>
          analyses.some((a) => a.id === id),
        );
        const nextSelected = validSelected ? selectedId : analyses[0].id;
        if (
          nextSelected !== selectedId ||
          validChecked.length !== checkedIds.length
        ) {
          set({ selectedId: nextSelected, checkedIds: validChecked });
        }
      },

      removeAnalysis: (id) => {
        clearFoldPdb(id);
        set((state) => {
          const analyses = state.analyses.filter((a) => a.id !== id);
          const selectedId =
            state.selectedId === id
              ? (analyses[0]?.id ?? null)
              : state.selectedId;
          const checkedIds = state.checkedIds.filter(
            (checkedId) => checkedId !== id,
          );
          return { analyses, selectedId, checkedIds };
        });
      },

      renameAnalysis: (id, name) => {
        const trimmed = name.trim();
        if (!trimmed) return;
        set((state) => ({
          analyses: state.analyses.map((a) =>
            a.id === id ? { ...a, name: trimmed } : a,
          ),
        }));
      },

      togglePin: (id) =>
        set((state) => ({
          analyses: state.analyses.map((a) =>
            a.id === id ? { ...a, pinned: !a.pinned } : a,
          ),
        })),

      updateSequence: (id, sequence) =>
        set((state) => ({
          analyses: state.analyses.map((a) =>
            a.id === id ? { ...a, sequence: sequence.toUpperCase() } : a,
          ),
        })),

      setStatus: (id, status) =>
        set((state) => ({
          analyses: state.analyses.map((a) =>
            a.id === id ? { ...a, status } : a,
          ),
        })),

      setError: (id, error) =>
        set((state) => ({
          analyses: state.analyses.map((a) =>
            a.id === id ? { ...a, error, status: error ? "error" : a.status } : a,
          ),
        })),

      setPrediction: (id, prediction, variant, sequence) =>
        set((state) => ({
          analyses: state.analyses.map((a) => {
            if (a.id !== id) return a;
            const vr = { ...(a.variantResults ?? {}) };
            vr[variant] = {
              ...(vr[variant] ?? {}),
              prediction,
              predictionVariant: variant,
              predictionSequence: sequence,
            };
            return { ...a, prediction, variantResults: vr, error: undefined };
          }),
        })),

      setOptimization: (id, optimization) =>
        set((state) => ({
          analyses: state.analyses.map((a) =>
            a.id === id
              ? {
                  ...a,
                  optimization,
                  structureComparison:
                    optimization.structure_comparison ?? a.structureComparison,
                  sequenceVariant: a.sequenceVariant ?? "optimized",
                  error: undefined,
                }
              : a,
          ),
        })),

      setSequenceVariant: (id, variant) =>
        set((state) => ({
          analyses: state.analyses.map((a) =>
            a.id === id ? { ...a, sequenceVariant: variant } : a,
          ),
        })),

      setStructureComparison: (id, comparison) =>
        set((state) => ({
          analyses: state.analyses.map((a) =>
            a.id === id ? { ...a, structureComparison: comparison } : a,
          ),
        })),

      setFold: (id, fold, variant, sequence) => {
        if (fold.pdb) {
          setFoldPdb(id, fold.pdb, variant);
        }
        set((state) => ({
          analyses: state.analyses.map((a) => {
            if (a.id !== id) return a;
            const vr = { ...(a.variantResults ?? {}) };
            vr[variant] = {
              ...(vr[variant] ?? {}),
              fold,
              foldVariant: variant,
              foldSequence: sequence,
            };
            return { ...a, fold, variantResults: vr, error: undefined };
          }),
        }));
      },

      setHeatmap: (id, heatmap) =>
        set((state) => ({
          analyses: state.analyses.map((a) =>
            a.id === id ? { ...a, heatmap, error: undefined } : a,
          ),
        })),

      setRegistry: (id, registry, variant, sequence) =>
        set((state) => ({
          analyses: state.analyses.map((a) => {
            if (a.id !== id) return a;
            const vr = { ...(a.variantResults ?? {}) };
            vr[variant] = {
              ...(vr[variant] ?? {}),
              registry,
              registryVariant: variant,
              registrySequence: sequence,
            };
            return { ...a, registry, variantResults: vr, error: undefined };
          }),
        })),

      setProcurement: (id, procurement, variant, sequence) =>
        set((state) => ({
          analyses: state.analyses.map((a) => {
            if (a.id !== id) return a;
            const vr = { ...(a.variantResults ?? {}) };
            vr[variant] = {
              ...(vr[variant] ?? {}),
              procurement,
              procurementVariant: variant,
              procurementSequence: sequence,
            };
            return {
              ...a,
              procurement,
              variantResults: vr,
              error: undefined,
              status: "complete",
            };
          }),
        })),

      getSelected: () => {
        const { analyses, selectedId } = get();
        return analyses.find((a) => a.id === selectedId);
      },
    }),
    {
      name: "pyroforge-analyses",
      partialize: (state) => ({
        analyses: state.analyses.map((a) => ({
          ...a,
          fold: a.fold
            ? {
                pdb_path: a.fold.pdb_path,
                sequence_length: a.fold.sequence_length,
                rmsf: a.fold.rmsf ?? [],
                pdb: "",
              }
            : undefined,
        })),
        selectedId: state.selectedId,
        checkedIds: state.checkedIds,
      }),
    },
  ),
);
