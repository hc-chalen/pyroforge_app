"""Systems benchmark — latency, throughput and scaling of the served path.

The model benchmarks say whether the predictions are any good. This one says
whether the software is usable: what a user waits for, where the time goes, and
at what sequence length the interactive path stops being interactive.

Method follows MLPerf Inference (Reddi et al., ISCA 2020) in the parts that
matter at this scale: warm-up runs are discarded, latency is reported as
percentiles rather than means (the mean hides the tail users actually complain
about), and the single-stream (one user waiting) and offline (batch throughput)
scenarios are measured separately.

The engines are driven directly rather than over HTTP. `PredictionEngine` and
`OptimizationEngine` *are* the served path — the API layer above them is a thin
FastAPI wrapper and Celery only moves the same call to another process — so this
avoids requiring Redis and a running stack while measuring the work that
actually dominates.

Cost model for the scan: a site-saturation scan of an L-residue protein needs
L masked forward passes (one per position, each yielding all 20 residue
probabilities) plus up to 19L mutant embeddings — so ~20L forward passes of
ESM-2 650M. That is expected to be linear in L with a large constant, and the
point of measuring is to find where it crosses "too slow to wait for".

Run::

    uv run python -m scripts.bench.systems               # default sweep
    uv run python -m scripts.bench.systems --max-scan-length 80
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger("bench.systems")

_BLUE, _ORANGE, _GREEN, _RED, _GREY = "#2f6f9f", "#c26a12", "#1d874b", "#c0392f", "#8fa0ad"

# A real-ish sequence to slice to length: the repeated motif keeps composition
# stable across lengths so timing differences reflect length, not content.
_MOTIF = "MKVLATGSAEYQRLIDNPFHGWTCAVMKDLSPQERNAYTIGHFVWLKDESRAMPQNCTYIVGHLFDKSAERPQMWNTYIC"


def _sequence(length: int) -> str:
    return (_MOTIF * (length // len(_MOTIF) + 1))[:length]


def _percentiles(samples: list[float]) -> dict:
    arr = np.array(samples, dtype=float)
    return {
        "n": int(len(arr)),
        "mean_s": float(arr.mean()),
        "p50_s": float(np.percentile(arr, 50)),
        "p90_s": float(np.percentile(arr, 90)),
        "p99_s": float(np.percentile(arr, 99)) if len(arr) >= 5 else float(arr.max()),
        "min_s": float(arr.min()),
        "max_s": float(arr.max()),
    }


def _peak_rss_mb() -> float:
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kB, macOS reports bytes.
    return peak / (1024 * 1024) if platform.system() == "Darwin" else peak / 1024


def run(out_dir: Path, plots_dir: Path, lengths, scan_lengths, reps: int) -> dict:
    from app.core.config import settings
    from app.engines.optimization import OptimizationEngine
    from app.engines.prediction import PredictionEngine
    from app.services.esm2 import get_esm2_service
    from app.services.model_store import get_model_store

    report = {
        "benchmark": "systems",
        "platform": {
            "system": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "device": str(settings.resolved_device),
        },
    }

    # --- Cold start --------------------------------------------------------- #
    # First-request cost after a deploy: loading ~2.5 GB of ESM-2 weights plus the
    # small heads. Measured before anything else touches the singletons.
    t0 = time.perf_counter()
    store = get_model_store()
    store.load()
    store_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    esm2 = get_esm2_service()
    esm_seconds = time.perf_counter() - t0
    report["cold_start"] = {
        "model_store_load_s": round(store_seconds, 3),
        "esm2_load_s": round(esm_seconds, 3),
        "total_s": round(store_seconds + esm_seconds, 3),
        "note": "ESM-2 weights already in the local torch hub cache; a first-ever "
                "run additionally downloads ~2.5 GB.",
    }

    prediction = PredictionEngine(esm2=esm2)
    optimization = OptimizationEngine(prediction=prediction, esm2=esm2)

    # --- Single-stream /predict latency vs length --------------------------- #
    predict_latency = {}
    for length in lengths:
        seq = _sequence(length)
        prediction.predict_absolute(seq)  # warm-up, discarded
        samples = []
        for _ in range(reps):
            t0 = time.perf_counter()
            prediction.predict_absolute(seq)
            samples.append(time.perf_counter() - t0)
        predict_latency[str(length)] = _percentiles(samples)
        logger.info("predict L=%d p50=%.3fs p90=%.3fs", length,
                    predict_latency[str(length)]["p50_s"],
                    predict_latency[str(length)]["p90_s"])
    report["predict_latency_by_length"] = predict_latency

    # --- Offline embedding throughput vs batch size ------------------------- #
    # The cheapest available throughput lever, and currently unmeasured: the
    # service defaults to batch_size=8 with no evidence behind that number.
    batch_throughput = {}
    for batch in (1, 2, 4, 8, 16):
        seqs = [_sequence(200)] * batch
        esm2.batch_embed(seqs, batch_size=batch)  # warm-up
        t0 = time.perf_counter()
        esm2.batch_embed(seqs, batch_size=batch)
        elapsed = time.perf_counter() - t0
        batch_throughput[str(batch)] = {
            "seconds": round(elapsed, 3),
            "seq_per_s": round(batch / elapsed, 3),
            "s_per_seq": round(elapsed / batch, 4),
        }
        logger.info("batch=%d %.2f seq/s", batch, batch / elapsed)
    report["embedding_throughput_by_batch"] = batch_throughput

    # --- Optimization scan wall-clock vs length ----------------------------- #
    # The expensive interactive path. Capped well below the API's 400-residue
    # limit because cost is ~20L forward passes; the fit is used to extrapolate.
    scan = {}
    for length in scan_lengths:
        seq = _sequence(length)
        t0 = time.perf_counter()
        candidates = optimization.saturation_scan(seq)
        elapsed = time.perf_counter() - t0
        scan[str(length)] = {
            "seconds": round(elapsed, 2),
            "candidates_scored": len(candidates),
            "s_per_residue": round(elapsed / length, 3),
        }
        logger.info("scan L=%d %.1fs (%d candidates)", length, elapsed, len(candidates))
    report["saturation_scan_by_length"] = scan

    # Linear fit through the scan timings, used only to extrapolate to the API cap.
    if len(scan) >= 2:
        xs = np.array([int(k) for k in scan], dtype=float)
        ys = np.array([v["seconds"] for v in scan.values()], dtype=float)
        slope, intercept = np.polyfit(xs, ys, 1)
        report["scan_scaling"] = {
            "seconds_per_residue": float(slope),
            "intercept_s": float(intercept),
            "r2": float(np.corrcoef(xs, ys)[0, 1] ** 2),
            "extrapolated_s_at_L400": float(slope * 400 + intercept),
            "extrapolated_min_at_L400": float((slope * 400 + intercept) / 60),
            "caveat": "linear extrapolation from short sequences; attention is "
                      "O(L^2) per pass, so long-sequence cost is a lower bound.",
        }

    report["peak_rss_mb"] = round(_peak_rss_mb(), 1)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "systems_benchmark.json").write_text(json.dumps(report, indent=1))
    _plot(report, plots_dir)
    return report


def _plot(report, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    lat = report["predict_latency_by_length"]
    xs = [int(k) for k in lat]
    ax.plot(xs, [lat[k]["p50_s"] for k in lat], "o-", color=_BLUE, label="p50")
    ax.plot(xs, [lat[k]["p90_s"] for k in lat], "s--", color=_ORANGE, label="p90")
    ax.plot(xs, [lat[k]["p99_s"] for k in lat], "^:", color=_RED, label="p99")
    ax.set_xlabel("sequence length (residues)"); ax.set_ylabel("seconds")
    ax.set_title("/predict single-stream latency")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    bt = report["embedding_throughput_by_batch"]
    xs = [int(k) for k in bt]
    ax.plot(xs, [bt[k]["seq_per_s"] for k in bt], "o-", color=_GREEN)
    ax.axvline(8, color=_GREY, ls="--", lw=1, label="current default (8)")
    ax.set_xlabel("batch size"); ax.set_ylabel("sequences / s")
    ax.set_title("Embedding throughput (L=200)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    scan = report["saturation_scan_by_length"]
    xs = np.array([int(k) for k in scan], dtype=float)
    ys = np.array([scan[k]["seconds"] for k in scan], dtype=float)
    ax.plot(xs, ys, "o-", color=_BLUE, label="measured")
    if "scan_scaling" in report:
        fit = report["scan_scaling"]
        grid = np.linspace(0, 400, 50)
        ax.plot(grid, fit["seconds_per_residue"] * grid + fit["intercept_s"], "--",
                color=_RED, label=f"linear fit → {fit['extrapolated_min_at_L400']:.1f} min at L=400")
    ax.set_xlabel("sequence length (residues)"); ax.set_ylabel("seconds")
    ax.set_title("Saturation scan wall-clock (the /optimize path)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.axis("off")
    cold = report["cold_start"]
    lines = [
        f"device: {report['platform']['device']}",
        f"cpu count: {report['platform']['cpu_count']}",
        "",
        f"cold start (weights cached): {cold['total_s']:.1f} s",
        f"   model heads: {cold['model_store_load_s']:.2f} s",
        f"   ESM-2 650M:  {cold['esm2_load_s']:.2f} s",
        "",
        f"peak RSS: {report['peak_rss_mb']:.0f} MB",
    ]
    if "scan_scaling" in report:
        fit = report["scan_scaling"]
        lines += [
            "",
            f"scan cost: {fit['seconds_per_residue']:.2f} s/residue (R²={fit['r2']:.3f})",
            f"extrapolated at API cap L=400: {fit['extrapolated_min_at_L400']:.1f} min",
        ]
    ax.text(0.02, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=10)
    ax.set_title("Summary", loc="left")

    fig.suptitle("PyroForge — systems benchmark (served path)", fontsize=13)
    fig.tight_layout()
    fig.savefig(plots_dir / "systems_benchmark.png", dpi=150)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--plots-dir", type=Path, default=Path("checkpoints/plots"))
    parser.add_argument("--reps", type=int, default=7)
    parser.add_argument("--lengths", type=int, nargs="+", default=[50, 100, 200, 300, 400])
    parser.add_argument("--scan-lengths", type=int, nargs="+", default=[20, 40, 60, 90])
    args = parser.parse_args()
    report = run(args.out_dir, args.plots_dir, args.lengths, args.scan_lengths, args.reps)
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("predict_latency_by_length",)}, indent=1))


if __name__ == "__main__":
    main()
