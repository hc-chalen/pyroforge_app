import axios from "axios";

const api = axios.create({
  // Empty baseURL → same-origin relative requests (/api/v1/…, /health), which
  // the frontend nginx reverse-proxies to the backend. Set VITE_API_URL only for
  // cross-origin dev (e.g. Vite dev server on :5173 calling the API on :8000).
  baseURL: import.meta.env.VITE_API_URL || "",
});

/**
 * Give transport-level failures a message a user can act on.
 *
 * Cards report failures with `err.message`, and axios writes that itself:
 * a throttled request arrived as "Request failed with status code 429", which
 * reads as a broken app rather than as "wait a moment". nginx returns its own
 * HTML error page for 429, so there is no `detail` field to fall back on and
 * formatJobError never sees it either — that one formats *job* errors, which
 * come back inside a 200 response.
 *
 * The rejected value stays the original axios error, with only `message`
 * rewritten: useHealth classifies throttling from `err.response.status`, and
 * replacing the error with a plain Error would lose that and turn every 429 into
 * a red "Backend unreachable" banner.
 */
api.interceptors.response.use(undefined, (error) => {
  const status: number | undefined = error?.response?.status;
  const detail: unknown = error?.response?.data?.detail;

  if (status === 429) {
    error.message =
      "Too many requests right now — this server is shared with everyone " +
      "else using it. Wait about 10 seconds and try again.";
  } else if (typeof detail === "string" && detail.trim()) {
    // FastAPI's own validation and refusal messages are already user-facing.
    error.message = detail;
  } else if (status === 503) {
    error.message = "The server is busy. Retry in a moment.";
  } else if (status != null && status >= 500) {
    error.message = "The server hit an error. Retry in a moment.";
  } else if (!error?.response) {
    error.message = "Cannot reach the server. Check your connection and retry.";
  }

  return Promise.reject(error);
});

export default api;
