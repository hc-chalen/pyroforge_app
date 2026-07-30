import type { JobStatus } from "@/types/api";
import api from "./api";

const POLL_INTERVAL_MS = 2000;
/** Must exceed the backend's task_soft_time_limit (3600 s) so a slow-but-alive job
 *  is not abandoned by the client before the worker itself gives up. */
const POLL_TIMEOUT_MS = 3_900_000;
/** Consecutive network/5xx failures tolerated before giving up. A transient blip
 *  must not be reported as a job failure while the worker is still running. */
const MAX_CONSECUTIVE_ERRORS = 5;

export async function fetchJob(jobId: string): Promise<JobStatus> {
  const { data } = await api.get<JobStatus>(`/api/v1/jobs/${jobId}`);
  return data;
}

export interface PollOptions {
  onProgress?: (progress: number) => void;
  /** Abort polling (e.g. the component unmounted or the user navigated away). */
  signal?: AbortSignal;
  timeoutMs?: number;
}

export async function pollJob(
  jobId: string,
  onProgressOrOptions?: ((progress: number) => void) | PollOptions,
): Promise<JobStatus> {
  const opts: PollOptions =
    typeof onProgressOrOptions === "function"
      ? { onProgress: onProgressOrOptions }
      : (onProgressOrOptions ?? {});
  const timeoutMs = opts.timeoutMs ?? POLL_TIMEOUT_MS;
  const startedAt = Date.now();
  let consecutiveErrors = 0;

  for (;;) {
    if (opts.signal?.aborted) {
      throw new DOMException("Polling aborted", "AbortError");
    }
    try {
      const job = await fetchJob(jobId);
      consecutiveErrors = 0;
      if (job.status === "running" && job.progress != null) {
        opts.onProgress?.(job.progress);
      }
      if (job.status === "completed" || job.status === "failed") {
        return job;
      }
    } catch (err) {
      // Celery reports an unknown or expired job id as "pending", so a dead worker
      // is indistinguishable from a slow one — the timeout below is the only exit.
      consecutiveErrors += 1;
      if (consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
        throw err;
      }
    }

    if (Date.now() - startedAt > timeoutMs) {
      throw new Error(
        `Job ${jobId} did not finish within ${Math.round(timeoutMs / 60000)} minutes. ` +
          `It may still be running on the worker, or the job may have been lost ` +
          `(a restarted worker or an expired result reports as "pending" forever).`,
      );
    }
    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
  }
}
