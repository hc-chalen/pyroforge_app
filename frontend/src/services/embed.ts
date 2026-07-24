import api from "./api";
import type { EmbedCompareRequest, EmbedCompareResponse } from "@/types/api";

export async function compareEmbeddings(
  body: EmbedCompareRequest,
): Promise<EmbedCompareResponse> {
  const { data } = await api.post<EmbedCompareResponse>(
    "/api/v1/embed/compare",
    body,
  );
  return data;
}
