import type {
  OptimizeResponse,
  RegistryDatabaseInfo,
  RegistrySearchRequest,
} from "@/types/api";
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

/**
 * Whether this server has a parts snapshot at all — cheap and Celery-free.
 *
 * Asked on mount so the card can say "unavailable on this server" up front,
 * rather than letting the user submit a sequence and wait on a job that was
 * never going to search anything.
 */
export async function getRegistryStatus(): Promise<RegistryDatabaseInfo> {
  const { data } = await api.get<RegistryDatabaseInfo>(
    "/api/v1/registry/status",
  );
  return data;
}
