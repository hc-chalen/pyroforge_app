import type {
  FoldRequest,
  FoldResponse,
  HeatmapRequest,
  HeatmapResponse,
} from "@/types/api";
import api from "./api";

function extractApiError(error: unknown, fallback: string): string {
  if (
    typeof error === "object" &&
    error !== null &&
    "response" in error &&
    typeof (error as { response?: { data?: { detail?: unknown } } }).response
      ?.data?.detail === "string"
  ) {
    return (error as { response: { data: { detail: string } } }).response.data
      .detail;
  }
  if (error instanceof Error) return error.message;
  return fallback;
}

export async function foldStructure(
  request: FoldRequest,
): Promise<FoldResponse> {
  try {
    const { data } = await api.post<FoldResponse>(
      "/api/v1/visualize/fold",
      request,
    );
    return data;
  } catch (error) {
    throw new Error(extractApiError(error, "Failed to start structure folding"));
  }
}

export async function heatmap(
  request: HeatmapRequest,
): Promise<HeatmapResponse> {
  const { data } = await api.post<HeatmapResponse>(
    "/api/v1/visualize/heatmap",
    request,
  );
  return data;
}
