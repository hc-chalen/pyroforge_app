import type { RmsfEntry } from "@/types/api";

/** Correct legacy RMSF rows where rigidity was over-scaled (pLDDT 0–1 ÷ 100). */
export function normalizeRmsfEntries(rmsf: RmsfEntry[]): RmsfEntry[] {
  if (rmsf.length === 0) return rmsf;

  const maxRigidity = Math.max(...rmsf.map((entry) => entry.rigidity));
  if (maxRigidity > 0.05) return rmsf;

  return rmsf.map((entry) => ({
    ...entry,
    rigidity: Math.min(1, entry.rigidity * 100),
    rmsf_proxy: entry.rmsf_proxy * 100,
  }));
}
