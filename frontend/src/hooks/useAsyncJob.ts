import { useCallback, useEffect, useRef, useState } from "react";
import { notifications } from "@mantine/notifications";
import { pollJob } from "@/services/jobs";
import type { JobStatus } from "@/types/api";
import { formatJobError } from "@/utils/formatJobError";

export function useAsyncJob() {
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  // Stop polling when the component goes away, instead of looping in the background
  // for the rest of the session.
  const abortRef = useRef<AbortController | null>(null);

  useEffect(
    () => () => {
      abortRef.current?.abort();
    },
    [],
  );

  const runJob = useCallback(
    async (jobId: string, label: string): Promise<JobStatus> => {
      setLoading(true);
      setProgress(0);
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;
      try {
        const result = await pollJob(jobId, {
          onProgress: setProgress,
          signal: controller.signal,
        });
        if (result.status === "failed") {
          throw new Error(
            formatJobError(result.error, `${label} failed`),
          );
        }
        notifications.show({
          title: `${label} complete`,
          message: "Results are ready.",
          color: "green",
        });
        return result;
      } catch (err) {
        // An abort is a deliberate teardown, not a job failure — don't alarm the user.
        if (err instanceof DOMException && err.name === "AbortError") {
          throw err;
        }
        const message =
          err instanceof Error ? err.message : `${label} failed`;
        notifications.show({
          title: `${label} failed`,
          message,
          color: "red",
        });
        throw err;
      } finally {
        setLoading(false);
        setProgress(0);
      }
    },
    [],
  );

  return { loading, progress, runJob };
}
