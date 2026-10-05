#!/usr/bin/env python3
"""Simulate N people using PyroForge at the same time.

Written for the class UAT — roughly 30 students, one protein between them, every
card tried — and reusable for any "will it hold up" question afterwards. It walks
the same endpoints the UI does, in the same order, polling jobs on the same 2 s
interval, and reports what each step cost and how often it was refused.

Stdlib only, so it runs on the GPU server without installing anything.

    # Through nginx, the way a real user arrives (rate limits apply)
    python3 scripts/uat_loadtest.py --users 30 --base-url http://localhost:3002

    # Straight at the backend: no nginx, no rate limits. This is what to use when
    # the question is "how much work can the workers actually get through",
    # which is the class-day case — everyone on their own mobile data has their
    # own IP, so the per-IP limits never bind.
    python3 scripts/uat_loadtest.py --users 30 --base-url http://127.0.0.1:8000

Flows:
    full     every card, optimize included. What "everyone tries everything" costs.
    light    everything except optimize (and embed, which needs its output).
    predict  predict only — the cheapest path, for isolating queue behaviour.

Optimize is a site-saturation scan: one Celery slot, no result cache, so N users
on the *same* protein is N identical scans, not one. Expect `full` with 30 users
to take a long time, and read that as the finding rather than as the tool hanging.
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# The 84-residue protein whose folding started all of this. Short enough that an
# optimize scan finishes in minutes rather than an hour.
DEFAULT_SEQUENCE = (
    "MSGPNGDLGMPVEAGAEGEEDGFGEAEYAAINSMLDQINSCLDHLEEKND"
    "HLHARLQELLESNRQTRLEFQQQLGEAPSDASP"
)

# Same interval the UI polls on (frontend/src/services/jobs.ts).
POLL_INTERVAL_S = 2.0

STEP_ORDER = [
    "predict",
    "registry/search",
    "fold",
    "heatmap",
    "procure/quote",
    "optimize",
    "embed/compare",
]


class Recorder:
    """Per-step outcomes, aggregated across threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.seconds: dict[str, list[float]] = defaultdict(list)
        self.outcomes: dict[str, Counter] = defaultdict(Counter)
        self.notes: dict[str, Counter] = defaultdict(Counter)
        self.poll_429 = 0

    def record(
        self,
        step: str,
        outcome: object,
        seconds: float,
        note: str | None = None,
    ) -> None:
        with self._lock:
            self.outcomes[step][str(outcome)] += 1
            if outcome == 200:
                self.seconds[step].append(seconds)
            if note:
                self.notes[step][note] += 1

    def count_poll_429(self) -> None:
        with self._lock:
            self.poll_429 += 1


def _parse(body: str) -> dict:
    try:
        parsed = json.loads(body)
    except ValueError:
        return {"_text": body.strip()[:200]}
    return parsed if isinstance(parsed, dict) else {"_json": parsed}


def http(
    base_url: str,
    path: str,
    payload: dict | None = None,
    timeout: float = 120.0,
) -> tuple[object, dict, float]:
    """POST when given a payload, GET otherwise. Never raises on an HTTP error.

    Returns (status, body, seconds); status is 0 when the request never got an
    HTTP response at all (connection refused, DNS, read timeout).
    """
    url = base_url.rstrip("/") + path
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        method="POST" if data is not None else "GET",
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, _parse(response.read().decode()), time.monotonic() - started
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        return exc.code, _parse(body), time.monotonic() - started
    except Exception as exc:  # connection refused, timeout, DNS
        return 0, {"_error": type(exc).__name__}, time.monotonic() - started


def run_job(
    rec: Recorder,
    base_url: str,
    step: str,
    path: str,
    payload: dict,
    job_timeout_s: float,
) -> dict | None:
    """Submit an async job and poll it to completion, as the UI does."""
    status, body, seconds = http(base_url, path, payload)
    if status != 200:
        rec.record(step, status, seconds, body.get("_error") or body.get("detail"))
        return None

    job_id = body.get("job_id")
    if not job_id:
        rec.record(step, "no-job-id", seconds)
        return None

    started = time.monotonic()
    while time.monotonic() - started < job_timeout_s:
        time.sleep(POLL_INTERVAL_S)
        poll_status, poll_body, _ = http(base_url, f"/api/v1/jobs/{job_id}", timeout=30)
        if poll_status == 429:
            # Throttled polling is not a failed job: the UI keeps trying too.
            rec.count_poll_429()
            continue
        if poll_status != 200:
            rec.record(step, f"poll-{poll_status}", time.monotonic() - started)
            return None
        state = poll_body.get("status")
        if state == "completed":
            rec.record(step, 200, time.monotonic() - started)
            return poll_body.get("result") or {}
        if state == "failed":
            error = (poll_body.get("error") or "")[:120]
            rec.record(step, "job-failed", time.monotonic() - started, error)
            return None

    rec.record(step, "timeout", time.monotonic() - started)
    return None


def run_sync(
    rec: Recorder,
    base_url: str,
    step: str,
    path: str,
    payload: dict,
) -> dict | None:
    status, body, seconds = http(base_url, path, payload)
    rec.record(step, status, seconds, body.get("_error") or body.get("detail"))
    return body if status == 200 else None


def user_flow(
    rec: Recorder,
    base_url: str,
    sequence: str,
    flow: str,
    job_timeout_s: float,
) -> None:
    run_job(rec, base_url, "predict", "/api/v1/predict/job", {"sequence": sequence}, job_timeout_s)

    if flow == "predict":
        return

    run_job(
        rec,
        base_url,
        "registry/search",
        "/api/v1/registry/search",
        {"sequence": sequence, "max_hits": 10},
        job_timeout_s,
    )

    fold = run_job(
        rec,
        base_url,
        "fold",
        "/api/v1/visualize/fold",
        {"sequence": sequence},
        job_timeout_s,
    )
    if fold and fold.get("rmsf"):
        scores = {
            str(entry["position"]): float(entry["rigidity"])
            for entry in fold["rmsf"]
        }
        run_sync(
            rec,
            base_url,
            "heatmap",
            "/api/v1/visualize/heatmap",
            {"sequence": sequence, "residue_scores": scores},
        )

    run_sync(
        rec,
        base_url,
        "procure/quote",
        "/api/v1/procure/quote",
        {"protein_sequence": sequence, "organism": "ecoli"},
    )

    if flow != "full":
        return

    optimized = run_job(
        rec,
        base_url,
        "optimize",
        "/api/v1/optimize",
        {"sequence": sequence, "max_mutations": 5},
        job_timeout_s,
    )
    if optimized and optimized.get("optimized_sequence"):
        run_sync(
            rec,
            base_url,
            "embed/compare",
            "/api/v1/embed/compare",
            {
                "sequence_original": sequence,
                "sequence_optimized": optimized["optimized_sequence"],
            },
        )


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


def report(rec: Recorder, users: int, flow: str, wall_s: float) -> None:
    print()
    print(f"{users} users · flow={flow} · wall clock {wall_s:.1f}s")
    print()
    header = f"{'step':<16}{'ok':>5}{'other':>7}{'p50':>9}{'p90':>9}{'max':>9}"
    print(header)
    print("-" * len(header))

    for step in STEP_ORDER:
        outcomes = rec.outcomes.get(step)
        if not outcomes:
            continue
        ok = outcomes.get("200", 0)
        other = sum(count for key, count in outcomes.items() if key != "200")
        times = rec.seconds.get(step, [])
        if times:
            cells = (
                f"{statistics.median(times):>8.1f}s"
                f"{percentile(times, 0.9):>8.1f}s"
                f"{max(times):>8.1f}s"
            )
        else:
            cells = f"{'-':>9}{'-':>9}{'-':>9}"
        print(f"{step:<16}{ok:>5}{other:>7}{cells}")

    print()
    problems = False
    for step in STEP_ORDER:
        for outcome, count in sorted(rec.outcomes.get(step, {}).items()):
            if outcome == "200":
                continue
            problems = True
            label = {
                "429": "429 throttled by nginx",
                "0": "no HTTP response (refused/timed out)",
                "timeout": "job never finished in time",
                "job-failed": "job returned an error",
            }.get(outcome, outcome)
            notes = rec.notes.get(step)
            detail = f"  ({notes.most_common(1)[0][0]})" if notes else ""
            print(f"  {step}: {count} × {label}{detail}")
    if rec.poll_429:
        problems = True
        print(f"  job polling: {rec.poll_429} × 429 (throttled, retried)")
    if not problems:
        print("  no failures, no throttling")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", type=int, default=30)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--sequence", default=DEFAULT_SEQUENCE)
    parser.add_argument("--flow", choices=["full", "light", "predict"], default="light")
    parser.add_argument(
        "--ramp",
        type=float,
        default=0.0,
        help="Seconds to spread the arrivals over. 0 = everyone at once, which is "
        "the worst case and what happens when a room is told to open the link.",
    )
    parser.add_argument("--job-timeout", type=float, default=900.0)
    args = parser.parse_args()

    rec = Recorder()
    started = time.monotonic()

    def one_user(index: int) -> None:
        if args.ramp:
            time.sleep(args.ramp * index / max(1, args.users))
        user_flow(rec, args.base_url, args.sequence.strip().upper(), args.flow, args.job_timeout)

    print(f"starting {args.users} users against {args.base_url} ...", flush=True)
    with ThreadPoolExecutor(max_workers=args.users) as pool:
        futures = [pool.submit(one_user, i) for i in range(args.users)]
        for future in as_completed(futures):
            future.result()

    report(rec, args.users, args.flow, time.monotonic() - started)


if __name__ == "__main__":
    main()
