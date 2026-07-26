#!/usr/bin/env python3
"""Per-account usage report for the PyroForge trial deployment.

Reads the JSON access log written by the auth gateway (auth/gateway.conf) and
summarises who used the app, how much, and how hard. Standard library only —
run it straight on the server with no install and no container:

    python3 monitoring/usage_report.py
    python3 monitoring/usage_report.py --since 7
    python3 monitoring/usage_report.py --format csv > usage.csv

GoAccess (docker-compose.monitor.yml) covers overall traffic shape; this script
is the per-account view, which GoAccess has no panel for.

Temporary, like the rest of the access gate — delete monitoring/ with auth/.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics
import sys
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG = REPO_ROOT / "logs" / "usage.log"

# The endpoints that actually cost GPU time, worth counting separately from
# page loads and status polls.
HEAVY = {
    "/api/v1/optimize": "optimize",
    "/api/v1/predict": "predict",
    "/api/v1/predict/job": "predict",
    "/api/v1/visualize/fold": "fold",
    "/api/v1/visualize/heatmap": "heatmap",
    "/api/v1/embed/compare": "embed",
    "/api/v1/registry/search": "registry",
}
# Status polling is high-volume and low-signal; counted, but never "activity".
NOISE_PREFIXES = ("/api/v1/jobs/", "/assets/", "/_auth/verify")


class Row:
    __slots__ = ("ts", "user", "ip", "method", "path", "status", "req_time")

    def __init__(self, d: dict):
        self.ts = d.get("ts", "")
        self.user = d.get("user") or ""
        self.ip = d.get("ip", "")
        self.method = d.get("method", "")
        self.path = d.get("path", "")
        self.status = int(d.get("status", 0) or 0)
        try:
            self.req_time = float(d.get("req_time", 0) or 0)
        except (TypeError, ValueError):
            self.req_time = 0.0

    @property
    def when(self) -> datetime | None:
        try:
            return datetime.fromisoformat(self.ts)
        except ValueError:
            return None


def load(log_path: Path, since_days: int | None) -> list[Row]:
    if not log_path.is_file():
        sys.exit(
            f"No usage log at {log_path}.\n"
            "It is created once the auth gateway serves its first request "
            "(docker compose ... -f docker-compose.auth.yml up -d)."
        )

    cutoff = None
    if since_days is not None:
        # Compared against tz-aware timestamps from nginx's $time_iso8601.
        cutoff = datetime.now().astimezone() - timedelta(days=since_days)

    rows: list[Row] = []
    malformed = 0
    with log_path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = Row(json.loads(line))
            except (json.JSONDecodeError, ValueError):
                malformed += 1
                continue
            if cutoff is not None:
                when = row.when
                if when is None or when < cutoff:
                    continue
            rows.append(row)

    if malformed:
        print(f"note: skipped {malformed} unparseable log line(s)\n", file=sys.stderr)
    return rows


def summarise(rows: list[Row]) -> dict[str, dict]:
    per_user: dict[str, dict] = collections.defaultdict(
        lambda: {
            "requests": 0,
            "heavy": collections.Counter(),
            "days": set(),
            "ips": set(),
            "times": [],
            "errors": 0,
            "first": None,
            "last": None,
        }
    )

    for row in rows:
        # Unauthenticated traffic is mostly the login page itself; it has no
        # account to attribute to and would drown out the real numbers.
        if not row.user:
            continue
        u = per_user[row.user]
        u["requests"] += 1
        u["ips"].add(row.ip)
        if row.status >= 500:
            u["errors"] += 1

        when = row.when
        if when is not None:
            u["days"].add(when.date())
            if u["first"] is None or when < u["first"]:
                u["first"] = when
            if u["last"] is None or when > u["last"]:
                u["last"] = when

        label = HEAVY.get(row.path)
        if label:
            u["heavy"][label] += 1
            u["times"].append(row.req_time)

    return per_user


def fmt_dt(value: datetime | None) -> str:
    return value.strftime("%Y-%m-%d %H:%M") if value else "-"


def print_text(rows: list[Row], per_user: dict[str, dict]) -> None:
    total_users = len(per_user)
    authed = sum(u["requests"] for u in per_user.values())

    print("=" * 78)
    print("PyroForge — usage by account")
    print("=" * 78)
    if not rows:
        print("\nNo requests in the selected window.\n")
        return

    span = [r.when for r in rows if r.when]
    if span:
        print(f"Window     : {fmt_dt(min(span))}  →  {fmt_dt(max(span))}")
    print(f"Accounts   : {total_users} active")
    print(f"Requests   : {authed} authenticated, {len(rows) - authed} unauthenticated")

    print("\n" + "-" * 78)
    print(f"{'account':<14}{'reqs':>7}{'days':>6}{'jobs':>7}{'p50 s':>8}{'p95 s':>8}  {'last seen':<17}")
    print("-" * 78)

    ranked = sorted(
        per_user.items(), key=lambda kv: sum(kv[1]["heavy"].values()), reverse=True
    )
    for name, u in ranked:
        times = sorted(u["times"])
        p50 = statistics.median(times) if times else 0.0
        p95 = times[min(int(len(times) * 0.95), len(times) - 1)] if times else 0.0
        print(
            f"{name[:13]:<14}{u['requests']:>7}{len(u['days']):>6}"
            f"{sum(u['heavy'].values()):>7}{p50:>8.2f}{p95:>8.2f}  {fmt_dt(u['last']):<17}"
        )

    print("\n" + "-" * 78)
    print("Compute jobs by account and kind")
    print("-" * 78)
    for name, u in ranked:
        if not u["heavy"]:
            continue
        breakdown = "  ".join(f"{k}={v}" for k, v in u["heavy"].most_common())
        print(f"{name[:13]:<14}{breakdown}")

    multi_ip = {n: u for n, u in per_user.items() if len(u["ips"]) > 3}
    if multi_ip:
        print("\n" + "-" * 78)
        print("Accounts seen from >3 addresses (possible credential sharing)")
        print("-" * 78)
        for name, u in multi_ip.items():
            print(f"{name[:13]:<14}{len(u['ips'])} addresses")

    print("\n" + "-" * 78)
    print("Daily active accounts")
    print("-" * 78)
    per_day: dict = collections.defaultdict(set)
    for row in rows:
        when = row.when
        if row.user and when:
            per_day[when.date()].add(row.user)
    for day in sorted(per_day):
        names = sorted(per_day[day])
        bar = "█" * len(names)
        print(f"{day}  {bar:<12} {len(names):>2}  {', '.join(names)}")
    print()


def print_csv(per_user: dict[str, dict]) -> None:
    writer = csv.writer(sys.stdout)
    writer.writerow(
        ["account", "requests", "active_days", "compute_jobs", "distinct_ips",
         "server_errors", "first_seen", "last_seen"]
    )
    for name, u in sorted(per_user.items()):
        writer.writerow([
            name, u["requests"], len(u["days"]), sum(u["heavy"].values()),
            len(u["ips"]), u["errors"], fmt_dt(u["first"]), fmt_dt(u["last"]),
        ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG,
                        help=f"path to usage.log (default: {DEFAULT_LOG})")
    parser.add_argument("--since", type=int, metavar="DAYS",
                        help="only count the last N days")
    parser.add_argument("--format", choices=("text", "csv"), default="text")
    args = parser.parse_args()

    rows = load(args.log, args.since)
    per_user = summarise(rows)

    if args.format == "csv":
        print_csv(per_user)
    else:
        print_text(rows, per_user)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
