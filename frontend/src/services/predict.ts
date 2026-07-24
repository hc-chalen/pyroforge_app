import type { PredictRequest, PredictResponse } from "@/types/api";
import api from "./api";
import { pollJob } from "./jobs";

export async function predict(
  request: PredictRequest,
): Promise<PredictResponse> {
  const { data } = await api.post<{ job_id: string; status: string }>(
    "/api/v1/predict/job",
    request,
  );
  const job = await pollJob(data.job_id);
  if (job.status === "failed") {
    throw new Error(job.error ?? "Prediction failed");
  }
  return job.result as PredictResponse;
}
