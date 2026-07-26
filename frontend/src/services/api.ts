import axios from "axios";

const api = axios.create({
  // Empty baseURL → same-origin relative requests (/api/v1/…, /health), which
  // the frontend nginx reverse-proxies to the backend. Set VITE_API_URL only for
  // cross-origin dev (e.g. Vite dev server on :5173 calling the API on :8000).
  baseURL: import.meta.env.VITE_API_URL || "",
});

export default api;
