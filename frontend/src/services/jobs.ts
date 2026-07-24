import type { JobStatus } from "@/types/api";
import api from "./api";

const POLL_INTERVAL_MS = 2000;

export async function fetchJob(jobId: string): Promise<JobStatus> {
  const { data } = await api.get<JobStatus>(`/api/v1/jobs/${jobId}`);
  return data;
}

export async function pollJob(
  jobId: string,
  onProgress?: (progress: number) => void,
): Promise<JobStatus> {
  for (;;) {
    const job = await fetchJob(jobId);
    if (job.status === "running" && job.progress != null) {
      onProgress?.(job.progress);
    }
    if (job.status === "completed" || job.status === "failed") {
      return job;
    }
    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
  }
}
