import type { RegistrySearchRequest, OptimizeResponse } from "@/types/api";
import api from "./api";

export async function searchRegistry(
  request: RegistrySearchRequest,
): Promise<OptimizeResponse> {
  const { data } = await api.post<OptimizeResponse>(
    "/api/v1/registry/search",
    request,
  );
  return data;
}
