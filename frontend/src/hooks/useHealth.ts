import { useEffect, useState } from "react";
import { fetchHealth } from "@/services/health";

/**
 * Backend health, polled **once per tab** and shared by every consumer.
 *
 * The previous version created an independent poller per `useHealth()` call —
 * App, PredictionCard and OptimizationCard, so three per tab — and dropped each
 * one to a 5-second interval whenever `/health` reported `esm_ready: "false"`.
 * That state is not transient: the API process only loads ESM-2 for the two
 * synchronous routes, while everything the UI actually uses runs in the Celery
 * workers, so with `PRELOAD_ESM2_ON_STARTUP=false` (the compose default) it is
 * false permanently. The result was ~36 requests/minute per tab, forever, for a
 * flag no component reads.
 *
 * `/health` is rate-limited per IP, and a lab full of students shares one campus
 * NAT address, so that budget ran out and nginx started returning 429 — which the
 * old code reported as "Backend unreachable" and answered by continuing to poll
 * at 5 s, keeping itself throttled. Hence the banner that blinked in and out.
 *
 * So: one poller, a steady interval, exponential backoff on failure, and 429
 * treated as what it is — the server telling us to slow down, not to panic.
 */

export type HealthFailure = "throttled" | "unreachable";

export interface HealthState {
  modelsReady: boolean;
  esmReady: boolean;
  loading: boolean;
  /** Set only once a failure has persisted; see FAILURES_BEFORE_REPORTING. */
  error?: string;
  failureKind?: HealthFailure;
}

const POLL_OK_MS = 30_000;
const BACKOFF_BASE_MS = 5_000;
const BACKOFF_MAX_MS = 120_000;
/** Being throttled means back off hard, not retry sooner. */
const THROTTLED_MIN_MS = 60_000;
/**
 * A single failed poll is a blip — a dropped packet, a worker restart, one
 * request over the burst. Showing a red banner for it and clearing it 30 s later
 * is the flicker itself, so a failure has to persist before it is surfaced.
 */
const FAILURES_BEFORE_REPORTING = 3;

let state: HealthState = { modelsReady: false, esmReady: false, loading: true };
const listeners = new Set<(s: HealthState) => void>();
let timer: ReturnType<typeof setTimeout> | undefined;
let consecutiveFailures = 0;
let started = false;

function publish(next: Partial<HealthState>): void {
  state = { ...state, ...next };
  listeners.forEach((l) => l(state));
}

function classify(err: unknown): HealthFailure {
  const status = (err as { response?: { status?: number } })?.response?.status;
  // A 429 came *from* nginx, so the stack is up and reachable — it is a quota
  // problem, and telling the user to start the API would be wrong.
  return status === 429 ? "throttled" : "unreachable";
}

function backoffMs(kind: HealthFailure): number {
  const base =
    kind === "throttled"
      ? THROTTLED_MIN_MS
      : Math.min(BACKOFF_BASE_MS * 2 ** (consecutiveFailures - 1), BACKOFF_MAX_MS);
  // Jitter so that tabs behind one NAT address do not re-synchronise into a
  // thundering herd against the same per-IP bucket.
  return base * (0.8 + Math.random() * 0.4);
}

async function check(): Promise<void> {
  try {
    const health = await fetchHealth();
    consecutiveFailures = 0;
    publish({
      modelsReady: health.models_ready === "true",
      esmReady: health.esm_ready !== "false",
      loading: false,
      error: undefined,
      failureKind: undefined,
    });
    schedule(POLL_OK_MS);
  } catch (err) {
    consecutiveFailures += 1;
    const kind = classify(err);
    const reportable = consecutiveFailures >= FAILURES_BEFORE_REPORTING;
    publish({
      loading: false,
      // modelsReady is deliberately left as it was: one failed health poll is
      // not evidence that the checkpoints unloaded, and flipping it made the
      // cards flash "Model checkpoints are not loaded yet" as well.
      ...(reportable
        ? {
            error:
              kind === "throttled"
                ? "Too many requests — the server is rate-limiting this connection."
                : err instanceof Error
                  ? err.message
                  : "Failed to reach backend",
            failureKind: kind,
          }
        : {}),
    });
    schedule(backoffMs(kind));
  }
}

function schedule(ms: number): void {
  if (timer) clearTimeout(timer);
  timer = setTimeout(check, ms);
}

export function useHealth(): HealthState {
  const [snapshot, setSnapshot] = useState<HealthState>(state);

  useEffect(() => {
    listeners.add(setSnapshot);
    if (!started) {
      started = true;
      void check();
    }
    return () => {
      listeners.delete(setSnapshot);
      // The poller is intentionally left running when the last consumer
      // unmounts: this is a tab-lifetime singleton, and tearing it down on every
      // card remount is what made the request rate depend on how the UI
      // re-rendered.
    };
  }, []);

  return snapshot;
}
