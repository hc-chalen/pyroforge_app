#!/usr/bin/env python3
"""Overview of PyTorch ``.pt`` files used by PyroForge.

Covers:
  * Embedding datasets (``esm2_*.pt``) — keys, shapes, label coverage
  * Model checkpoints (``model_*.pt``) — config, split, weight inventory

Examples::

  uv run python scripts/inspect_pt.py
  uv run python scripts/inspect_pt.py ../esm2_50k.pt checkpoints/model_ddg.pt
  uv run python scripts/inspect_pt.py --json ../esm2_test.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATHS = [
    REPO_ROOT / "esm2_50k.pt",
    REPO_ROOT / "esm2_train.pt",
    REPO_ROOT / "esm2_test.pt",
    REPO_ROOT / "backend" / "checkpoints" / "model_thermo.pt",
    REPO_ROOT / "backend" / "checkpoints" / "model_ddg.pt",
]

LABEL_KEYS = (
    "ddg",
    "dg",
    "tm",
    "dtm",
    "domainome_ddg",
    "domainome_fitness",
    "chymotrypsin_ml",
    "trypsin_ml",
    "cm",
    "dcp",
    "m",
    "dh",
    "dhvh",
)


def _fmt_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def _tensor_summary(t: torch.Tensor) -> dict[str, Any]:
    out: dict[str, Any] = {
        "type": "Tensor",
        "shape": list(t.shape),
        "dtype": str(t.dtype).replace("torch.", ""),
        "device": str(t.device),
        "numel": int(t.numel()),
        "nbytes": int(t.numel() * t.element_size()),
    }
    if t.numel() == 0:
        return out
    if t.is_floating_point():
        flat = t.detach().float().reshape(-1)
        finite = torch.isfinite(flat)
        n_finite = int(finite.sum())
        out["n_nan"] = int(torch.isnan(flat).sum())
        out["n_inf"] = int(torch.isinf(flat).sum())
        out["finite_frac"] = float(n_finite / flat.numel())
        if n_finite:
            vals = flat[finite]
            out["min"] = float(vals.min())
            out["max"] = float(vals.max())
            out["mean"] = float(vals.mean())
            out["std"] = float(vals.std(unbiased=False))
    elif t.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        flat = t.detach().reshape(-1)
        # Keep stats cheap on huge integer id tensors
        if flat.numel() <= 2_000_000:
            out["min"] = int(flat.min())
            out["max"] = int(flat.max())
            out["n_unique"] = int(torch.unique(flat).numel())
        else:
            sample = flat[:: max(1, flat.numel() // 200_000)][:200_000]
            out["min_sampled"] = int(sample.min())
            out["max_sampled"] = int(sample.max())
            out["n_unique_sampled"] = int(torch.unique(sample).numel())
    return out


def _scalar_summary(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        if len(v) <= 12 and all(isinstance(x, (int, float, str, bool, type(None))) for x in v):
            return list(v)
        return {"type": type(v).__name__, "len": len(v), "preview": list(v[:5])}
    if isinstance(v, dict):
        return {k: _scalar_summary(x) for k, x in v.items()}
    if torch.is_tensor(v):
        return _tensor_summary(v)
    if isinstance(v, np.ndarray):
        return {
            "type": "ndarray",
            "shape": list(v.shape),
            "dtype": str(v.dtype),
            "nbytes": int(v.nbytes),
        }
    return {"type": type(v).__name__, "repr": repr(v)[:160]}


def _state_dict_summary(sd: dict[str, Any]) -> dict[str, Any]:
    entries = []
    total_params = 0
    total_bytes = 0
    for name, tensor in sd.items():
        if not torch.is_tensor(tensor):
            entries.append({"name": name, "type": type(tensor).__name__})
            continue
        n = int(tensor.numel())
        b = int(tensor.numel() * tensor.element_size())
        total_params += n
        total_bytes += b
        entries.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).replace("torch.", ""),
                "numel": n,
            }
        )
    return {
        "n_tensors": len(entries),
        "total_params": total_params,
        "total_bytes": total_bytes,
        "total_bytes_human": _fmt_bytes(total_bytes),
        "tensors": entries,
    }


def _detect_kind(obj: Any) -> str:
    if not isinstance(obj, dict):
        return "other"
    if "embeddings" in obj:
        return "embedding_dataset"
    if "state_dict" in obj and ("model_name" in obj or "config" in obj):
        return "model_checkpoint"
    if "state_dict" in obj:
        return "state_dict_checkpoint"
    return "dict"


def inspect_object(obj: Any) -> dict[str, Any]:
    kind = _detect_kind(obj)
    report: dict[str, Any] = {"kind": kind, "top_type": type(obj).__name__}

    if not isinstance(obj, dict):
        report["value"] = _scalar_summary(obj)
        return report

    report["keys"] = list(obj.keys())

    if kind == "embedding_dataset":
        meta_keys = [
            "model_name",
            "pooling",
            "hidden_size",
            "row_count",
            "scope",
            "export_limit",
            "export_offset",
            "available_in_scope",
            "aa_vocab",
        ]
        report["metadata"] = {k: obj.get(k) for k in meta_keys if k in obj}

        tensors: dict[str, Any] = {}
        for k, v in obj.items():
            if torch.is_tensor(v):
                tensors[k] = _tensor_summary(v)
        report["tensors"] = tensors

        labels: dict[str, Any] = {}
        for k in LABEL_KEYS:
            if k not in obj or not torch.is_tensor(obj[k]):
                continue
            t = obj[k]
            finite = torch.isfinite(t.float())
            n = int(t.numel())
            n_ok = int(finite.sum())
            labels[k] = {
                "n": n,
                "n_finite": n_ok,
                "finite_frac": float(n_ok / n) if n else 0.0,
            }
            if n_ok:
                vals = t.float()[finite]
                labels[k].update(
                    {
                        "min": float(vals.min()),
                        "max": float(vals.max()),
                        "mean": float(vals.mean()),
                        "std": float(vals.std(unbiased=False)),
                    }
                )
        report["labels"] = labels

        id_keys = [
            "parent_protein_ids",
            "source_sequence_ids",
            "target_sequence_ids",
            "mutant_ids",
        ]
        ids: dict[str, Any] = {}
        for k in id_keys:
            if k in obj and torch.is_tensor(obj[k]):
                ids[k] = _tensor_summary(obj[k])
        if ids:
            report["ids"] = ids

    elif kind in ("model_checkpoint", "state_dict_checkpoint"):
        for k in (
            "model_name",
            "feature_spec",
            "config",
            "calibration",
            "dtm_calibration",
            "residual_std",
            "split",
        ):
            if k in obj:
                report[k] = _scalar_summary(obj[k])

        if "norm_stats" in obj and isinstance(obj["norm_stats"], dict):
            ns = obj["norm_stats"]
            report["norm_stats"] = {
                k: (
                    {
                        "type": "list",
                        "len": len(v),
                        "preview_head": v[:3],
                        "preview_tail": v[-3:],
                    }
                    if isinstance(v, list)
                    else _scalar_summary(v)
                )
                for k, v in ns.items()
            }

        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            report["state_dict"] = _state_dict_summary(obj["state_dict"])

        if "nas" in obj and isinstance(obj["nas"], dict):
            nas = obj["nas"]
            report["nas"] = {
                "budget_secs": nas.get("budget_secs"),
                "best_config": _scalar_summary(nas.get("best_config")),
                "search_history_len": len(nas["search_history"])
                if isinstance(nas.get("search_history"), list)
                else None,
            }

        # Any leftover non-heavy keys
        known = {
            "model_name",
            "feature_spec",
            "config",
            "calibration",
            "dtm_calibration",
            "residual_std",
            "split",
            "norm_stats",
            "state_dict",
            "nas",
        }
        extras = {k: _scalar_summary(v) for k, v in obj.items() if k not in known}
        if extras:
            report["other"] = extras
    else:
        report["contents"] = {k: _scalar_summary(v) for k, v in obj.items()}

    return report


def inspect_file(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "path": str(path.resolve()),
        "exists": path.exists(),
    }
    if not path.exists():
        info["error"] = "file not found"
        return info

    info["size_bytes"] = path.stat().st_size
    info["size_human"] = _fmt_bytes(path.stat().st_size)

    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 — surface load failures cleanly
        info["error"] = f"{type(exc).__name__}: {exc}"
        return info

    info.update(inspect_object(obj))
    return info


def _print_tensor_line(name: str, s: dict[str, Any], indent: str = "    ") -> None:
    shape = "x".join(str(d) for d in s["shape"])
    line = f"{indent}{name:28s}  [{shape}]  {s['dtype']}"
    if "finite_frac" in s:
        line += f"  finite={s['finite_frac']:.1%}"
        if "mean" in s:
            line += f"  mean={s['mean']:.4g}  [{s['min']:.4g}, {s['max']:.4g}]"
    elif "n_unique" in s:
        line += f"  range=[{s['min']}, {s['max']}]  unique={s['n_unique']}"
    elif "n_unique_sampled" in s:
        line += (
            f"  range≈[{s['min_sampled']}, {s['max_sampled']}]"
            f"  unique≈{s['n_unique_sampled']} (sampled)"
        )
    print(line)


def print_report(report: dict[str, Any]) -> None:
    print("=" * 72)
    print(report["path"])
    if not report.get("exists"):
        print("  MISSING")
        return
    print(f"  size: {report['size_human']} ({report['size_bytes']:,} bytes)")
    if "error" in report:
        print(f"  ERROR: {report['error']}")
        return

    print(f"  kind: {report['kind']}")

    if report["kind"] == "embedding_dataset":
        meta = report.get("metadata") or {}
        if meta:
            print("  metadata:")
            for k, v in meta.items():
                print(f"    {k}: {v}")

        print("  tensors:")
        for name, s in (report.get("tensors") or {}).items():
            _print_tensor_line(name, s)

        labels = report.get("labels") or {}
        if labels:
            print("  labels (finite coverage):")
            for name, s in labels.items():
                line = (
                    f"    {name:28s}  {s['n_finite']}/{s['n']}"
                    f" ({s['finite_frac']:.1%})"
                )
                if "mean" in s:
                    line += f"  mean={s['mean']:.4g}  [{s['min']:.4g}, {s['max']:.4g}]"
                print(line)

        ids = report.get("ids") or {}
        if ids:
            print("  id columns:")
            for name, s in ids.items():
                _print_tensor_line(name, s)

    elif report["kind"] in ("model_checkpoint", "state_dict_checkpoint"):
        if "model_name" in report:
            print(f"  model_name: {report['model_name']}")
        if "feature_spec" in report:
            print(f"  feature_spec: {report['feature_spec']}")
        if "config" in report:
            print(f"  config: {json.dumps(report['config'], default=str)}")
        if "split" in report:
            print(f"  split: {json.dumps(report['split'], default=str)}")
        if "residual_std" in report:
            print(f"  residual_std: {json.dumps(report['residual_std'], default=str)}")
        if "calibration" in report:
            print(f"  calibration: {json.dumps(report['calibration'], default=str)}")
        if "dtm_calibration" in report:
            print(f"  dtm_calibration: {json.dumps(report['dtm_calibration'], default=str)}")
        if "norm_stats" in report:
            print("  norm_stats:")
            for k, v in report["norm_stats"].items():
                if isinstance(v, dict) and v.get("type") == "list":
                    print(f"    {k}: list[{v['len']}] head={v['preview_head']} tail={v['preview_tail']}")
                else:
                    print(f"    {k}: {v}")
        if "nas" in report:
            print(f"  nas: budget={report['nas'].get('budget_secs')}s"
                  f"  trials={report['nas'].get('search_history_len')}"
                  f"  best={json.dumps(report['nas'].get('best_config'), default=str)}")
        sd = report.get("state_dict")
        if sd:
            print(
                f"  state_dict: {sd['n_tensors']} tensors,"
                f" {sd['total_params']:,} params,"
                f" {sd['total_bytes_human']}"
            )
            for e in sd["tensors"]:
                shape = "x".join(str(d) for d in e["shape"])
                print(f"    {e['name']:40s}  [{shape}]  {e['dtype']}")
        if report.get("other"):
            print(f"  other: {json.dumps(report['other'], default=str)}")
    else:
        print(f"  keys: {report.get('keys')}")
        if "contents" in report:
            print(f"  contents: {json.dumps(report['contents'], default=str)[:2000]}")


def resolve_paths(args: argparse.Namespace) -> list[Path]:
    if args.paths:
        return [Path(p) for p in args.paths]
    found = [p for p in DEFAULT_PATHS if p.exists()]
    if args.all:
        # Also pick up any other .pt under repo root / checkpoints (non-recursive deep)
        extras = list(REPO_ROOT.glob("*.pt"))
        extras += list((REPO_ROOT / "backend" / "checkpoints").glob("*.pt"))
        extras += list((REPO_ROOT / "backend" / "checkpoints").glob("**/*.pt"))
        for p in extras:
            if p not in found:
                found.append(p)
    return found


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize contents of PyroForge .pt files")
    p.add_argument(
        "paths",
        nargs="*",
        help="Specific .pt files (default: known esm2_*.pt + model_*.pt if present)",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Also include any other .pt under the repo / checkpoints",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of text",
    )
    args = p.parse_args()

    paths = resolve_paths(args)
    if not paths:
        print("No .pt files found. Pass paths explicitly.", file=sys.stderr)
        sys.exit(1)

    reports = [inspect_file(path) for path in paths]

    if args.json:
        print(json.dumps(reports, indent=2, default=str))
    else:
        for r in reports:
            print_report(r)
            print()


if __name__ == "__main__":
    main()
