import type { OptimizeRequest, OptimizeResponse } from "@/types/api";
import api from "./api";

export async function optimize(
  request: OptimizeRequest,
): Promise<OptimizeResponse> {
  const { data } = await api.post<OptimizeResponse>(
    "/api/v1/optimize",
    request,
  );
  return data;
}
