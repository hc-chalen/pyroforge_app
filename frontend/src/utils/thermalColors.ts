import type { RmsfEntry } from "@/types/api";

export interface ResidueThermalState {
  chain: string;
  position: number;
  aa: string;
  color: string;
  meltOnset: number;
  labileFraction: number;
  vulnerability: number;
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function rgbToHex(r: number, g: number, b: number): string {
  return `#${[r, g, b]
    .map((x) => Math.round(x).toString(16).padStart(2, "0"))
    .join("")}`;
}

/**
 * Map per-residue flexibility (from pLDDT-derived rigidity) to 0–1 vulnerability.
 * Low pLDDT / rigidity → high vulnerability (melts earlier).
 */
export function vulnerabilityByPosition(
  rmsf: RmsfEntry[],
  sequenceLength: number,
): Map<number, number> {
  const rigidityByPosition = new Map<number, number>();
  for (const entry of rmsf) {
    rigidityByPosition.set(entry.position, entry.rigidity);
  }

  const rigidities: number[] = [];
  for (let position = 1; position <= sequenceLength; position += 1) {
    rigidities.push(rigidityByPosition.get(position) ?? 0.5);
  }

  const min = Math.min(...rigidities);
  const max = Math.max(...rigidities);
  const span = max - min;

  const result = new Map<number, number>();
  for (let position = 1; position <= sequenceLength; position += 1) {
    const rigidity = rigidityByPosition.get(position) ?? 0.5;
    let vulnerability: number;
    if (span < 0.05) {
      // Flat pLDDT — use absolute flexibility (low confidence = labile).
      vulnerability = clamp(1 - rigidity, 0, 1);
    } else {
      const normalized = (rigidity - min) / span;
      vulnerability = 1 - normalized;
    }
    result.set(position, vulnerability);
  }
  return result;
}

/**
 * Heuristic per-residue melt onset (°C) for the colour ramp only.
 *
 * **This is not a prediction.** No model in PyroForge predicts a per-residue melting
 * temperature. It is an illustrative interpolation — a labile site is shown as melting
 * up to 45% below the whole-protein Tm — driven by pLDDT-derived vulnerability, which
 * is a structure-confidence score rather than a thermodynamic quantity. Present it as
 * a relative ordering of fragile regions, never as a measured or predicted °C.
 */
export function localMeltTemperature(
  baselineTm: number,
  vulnerability: number,
): number {
  return baselineTm - vulnerability * baselineTm * 0.45;
}

/** Map vulnerability + system temperature to blue (stable) → red (labile). */
export function thermalColorAtTemp(
  vulnerability: number,
  baselineTm: number,
  temperature: number,
): { color: string; labileFraction: number; meltOnset: number } {
  const meltOnset = localMeltTemperature(baselineTm, vulnerability);
  const transitionWidth = 8;
  const labileFraction = clamp(
    (temperature - (meltOnset - transitionWidth)) / transitionWidth,
    0,
    1,
  );
  const r = lerp(40, 255, labileFraction);
  const g = lerp(130, 45, labileFraction);
  const b = lerp(220, 35, labileFraction);
  return {
    color: rgbToHex(r, g, b),
    labileFraction,
    meltOnset,
  };
}

export function computeResidueThermalStates(
  sequence: string,
  rmsf: RmsfEntry[] = [],
  baselineTm: number,
  temperature: number,
): ResidueThermalState[] {
  const vulnerabilities = vulnerabilityByPosition(rmsf, sequence.length);
  const chainByPosition = new Map<number, string>();
  for (const entry of rmsf) {
    chainByPosition.set(entry.position, entry.chain);
  }

  return sequence.split("").map((aa, index) => {
    const position = index + 1;
    const chain = chainByPosition.get(position) ?? "A";
    const vulnerability = vulnerabilities.get(position) ?? 0.5;
    const { color, labileFraction, meltOnset } = thermalColorAtTemp(
      vulnerability,
      baselineTm,
      temperature,
    );
    return {
      chain,
      position,
      aa,
      color,
      meltOnset,
      labileFraction,
      vulnerability,
    };
  });
}

/**
 * Reference anchor for the colour ramp when no Tm has been predicted. The thermal
 * view needs *some* anchor to render, but a value derived from this is NOT a
 * prediction — callers must check {@link hasPredictedBaselineTm} before labelling it
 * as one.
 */
export const FALLBACK_BASELINE_TM = 65;

export function baselineMeltingTemperature(
  predictionTm?: number,
  optimizationBaselineTm?: number,
): number {
  return optimizationBaselineTm ?? predictionTm ?? FALLBACK_BASELINE_TM;
}

/** True when the anchor came from a model, not from {@link FALLBACK_BASELINE_TM}. */
export function hasPredictedBaselineTm(
  predictionTm?: number | null,
  optimizationBaselineTm?: number | null,
): boolean {
  return optimizationBaselineTm != null || predictionTm != null;
}

export function temperatureRange(baselineTm: number): {
  min: number;
  max: number;
  default: number;
} {
  return {
    min: 20,
    max: Math.round(Math.max(baselineTm + 35, 95)),
    default: 25,
  };
}
