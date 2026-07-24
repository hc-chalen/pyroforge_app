import type { ProcureRequest, ProcureResponse } from "@/types/api";
import api from "./api";

export async function procureQuote(
  request: ProcureRequest,
): Promise<ProcureResponse> {
  const { data } = await api.post<ProcureResponse>(
    "/api/v1/procure/quote",
    request,
  );
  return data;
}
