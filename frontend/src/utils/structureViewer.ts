import type { ResidueThermalState } from "@/utils/thermalColors";

interface ViewerAtom {
  resi: number;
}

interface StructureViewer {
  setStyle: (
    sel: Record<string, unknown>,
    style: Record<string, unknown>,
  ) => void;
  render: (callback?: () => void) => void;
}

/**
 * Color cartoon by residue index using 3Dmol's colorfunc (required for
 * per-residue cartoon colors; global setStyle color applies uniformly).
 */
export function applyThermalStylesToViewer(
  viewer: StructureViewer,
  residues: ResidueThermalState[],
): void {
  const colorByResi = new Map<number, string>();
  for (const residue of residues) {
    colorByResi.set(residue.position, residue.color);
  }

  viewer.setStyle(
    {},
    {
      cartoon: {
        colorfunc: (atom: ViewerAtom) =>
          colorByResi.get(atom.resi) ?? "#b0b8c4",
      },
    },
  );

  viewer.render();
}

export function downloadPdb(pdb: string, filename = "structure.pdb"): void {
  const blob = new Blob([pdb], { type: "chemical/x-pdb" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}
