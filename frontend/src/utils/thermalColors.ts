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

/** Per-residue melt onset (°C). Labile sites melt up to ~45% below global Tm. */
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

export function baselineMeltingTemperature(
  predictionTm?: number,
  optimizationBaselineTm?: number,
): number {
  return optimizationBaselineTm ?? predictionTm ?? 65;
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
