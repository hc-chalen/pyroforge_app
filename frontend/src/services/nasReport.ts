import type { NasReportResponse } from "@/types/api";
import api from "./api";

export async function fetchNasReport(): Promise<NasReportResponse> {
  const { data } = await api.get<NasReportResponse>("/api/v1/models/nas-report");
  return data;
}
