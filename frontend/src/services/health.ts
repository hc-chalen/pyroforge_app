import type { HealthResponse } from "@/types/api";
import api from "./api";

export async function fetchHealth(): Promise<HealthResponse> {
  const { data } = await api.get<HealthResponse>("/health");
  return data;
}
