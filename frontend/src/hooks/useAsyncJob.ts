import { useCallback, useState } from "react";
import { notifications } from "@mantine/notifications";
import { pollJob } from "@/services/jobs";
import type { JobStatus } from "@/types/api";
import { formatJobError } from "@/utils/formatJobError";

export function useAsyncJob() {
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);

  const runJob = useCallback(
    async (jobId: string, label: string): Promise<JobStatus> => {
      setLoading(true);
      setProgress(0);
      try {
        const result = await pollJob(jobId, setProgress);
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
