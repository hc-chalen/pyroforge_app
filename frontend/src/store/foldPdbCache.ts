/** Session-scoped PDB text — too large for localStorage. Keyed by analysisId:variant */
const cache = new Map<string, string>();

function cacheKey(analysisId: string, variant = "default"): string {
  return variant === "default" ? analysisId : `${analysisId}:${variant}`;
}

export function setFoldPdb(
  analysisId: string,
  pdb: string,
  variant = "default",
): void {
  const key = cacheKey(analysisId, variant);
  cache.set(key, pdb);
  try {
    sessionStorage.setItem(
      `pyroforge-pdb:${key}`,
      pdb.slice(0, 500_000),
    );
  } catch {
    // sessionStorage quota — in-memory cache still works this session
  }
}

export function getFoldPdb(
  analysisId: string,
  variant = "default",
): string | undefined {
  const key = cacheKey(analysisId, variant);
  if (cache.has(key)) {
    return cache.get(key);
  }
  try {
    const stored = sessionStorage.getItem(`pyroforge-pdb:${key}`);
    if (stored) {
      cache.set(key, stored);
      return stored;
    }
  } catch {
    // ignore
  }
  return undefined;
}

export function clearFoldPdb(analysisId: string): void {
  for (const key of [...cache.keys()]) {
    if (key === analysisId || key.startsWith(`${analysisId}:`)) {
      cache.delete(key);
    }
  }
  try {
    for (let i = sessionStorage.length - 1; i >= 0; i -= 1) {
      const k = sessionStorage.key(i);
      if (k?.startsWith(`pyroforge-pdb:${analysisId}`)) {
        sessionStorage.removeItem(k);
      }
    }
  } catch {
    // ignore
  }
}
