import { useEffect, useState } from "react";
import { fetchHealth } from "@/services/health";

export function useHealth(pollMs = 30000) {
  const [modelsReady, setModelsReady] = useState(false);
  const [esmReady, setEsmReady] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | undefined>();

  useEffect(() => {
    let cancelled = false;
    let timeoutId: ReturnType<typeof setTimeout> | undefined;

    async function check() {
      try {
        const health = await fetchHealth();
        if (!cancelled) {
          const ready = health.esm_ready !== "false";
          setModelsReady(health.models_ready === "true");
          setEsmReady(ready);
          setError(undefined);
          timeoutId = setTimeout(check, ready ? pollMs : 5000);
        }
      } catch (err) {
        if (!cancelled) {
          setModelsReady(false);
          setEsmReady(false);
          setError(
            err instanceof Error ? err.message : "Failed to reach backend",
          );
          timeoutId = setTimeout(check, 5000);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    check();
    return () => {
      cancelled = true;
      if (timeoutId) clearTimeout(timeoutId);
    };
  }, [pollMs]);

  return { modelsReady, esmReady, loading, error };
}
