"""Train the canonical absolute-Tm model on the ingested Meltome Atlas.

Produces ``checkpoints/improved/model_tm.joblib`` — a single-database
:class:`~app.models.tm_regressor.TmRegressor` (PCA-100 + Ridge+GBM) trained on
Meltome proteins only. A learning curve showed absolute Tm was starved of
proteins, not of model capacity; with ~4k Meltome proteins this reaches
r≈0.81 / MAE≈5.9 °C on a held-out split — the scientifically reliable range.

Why one database, not two: the original tiny-domain Tm labels are a *different
assay* (folding Tm on designed domains) and do not transfer to the Meltome
whole-protein regime (adding them left Meltome-test r unchanged-to-slightly-worse
and did not help the original test). This trainer therefore reports the
cross-assay transfer number for the record but trains the served model on Meltome
alone. A mutant's absolute Tm is obtained by running this model on the mutant
sequence embedding; the mutation's stability effect is reported as ΔΔG.

Run:  uv run python -m app.training.train_tm_meltome
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from app.core.config import settings
from app.models.tm_regressor import TmRegressor

logger = logging.getLogger(__name__)


def _extract_tm(path: Path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    tm = d["tm"].numpy().astype(np.float64)
    m = ~np.isnan(tm)
    emb = d["embeddings"].numpy()[m].astype(np.float32)
    pid = d["parent_protein_ids"].numpy()[m]
    del d
    return emb, tm[m].astype(np.float32), pid


def _metrics(pred, true, pid=None):
    out = dict(n=int(len(pred)),
               pearson=float(pearsonr(pred, true)[0]),
               spearman=float(spearmanr(pred, true).correlation),
               mae=float(np.mean(np.abs(pred - true))),
               rmse=float(np.sqrt(np.mean((pred - true) ** 2))))
    if pid is not None:
        ps = np.unique(pid)
        out["baseline_r"] = float(pearsonr([pred[pid == q].mean() for q in ps],
                                           [true[pid == q].mean() for q in ps])[0])
    return out


def run(meltome_npz: Path | None = None, out: Path | None = None) -> dict:
    ck = settings.checkpoint_dir / "improved"
    meltome_npz = meltome_npz or ck / "meltome_tm.npz"
    out = out or ck / "model_tm.joblib"

    M = np.load(meltome_npz, allow_pickle=True)
    emb, tm, split = M["emb"], M["tm"].astype(np.float32), M["split"].astype(str)
    tr, te = split != "test", split == "test"

    model = TmRegressor().fit(emb[tr], tm[tr])
    test_metrics = _metrics(model.predict(emb[te]), tm[te])
    # Held-out residual, not train — this becomes the served ±95% interval on Tm.
    resid = float(np.std(model.predict(emb[te]) - tm[te]))

    # transfer diagnostic (record only): does the Meltome model read across to the
    # original tiny-domain assay? (Expected: weak — different measurement.)
    transfer = None
    try:
        oe, oy, opid = _extract_tm(settings.test_data_path)
        transfer = _metrics(model.predict(oe), oy, opid)
    except Exception as e:  # noqa: BLE001
        logger.warning("transfer diagnostic skipped: %s", e)

    joblib.dump({"model": model, "recipe": "PCA100 + Ridge+GBM on Meltome (single database)",
                 "residual_std": {"tm": resid}, "n_train": int(tr.sum()),
                 "test_metrics": test_metrics, "transfer_original_assay": transfer}, out)
    report = {"model_tm": {"meltome_test": test_metrics, "residual_std_tm": resid,
                           "n_train": int(tr.sum())},
              "transfer_to_original_assay": transfer}
    with open(ck / "tm_report.json", "w") as f:
        json.dump(report, f, indent=2)
    logger.info("model_tm (Meltome) — test r=%.3f MAE=%.2f°C (n=%d) | saved %s",
                test_metrics["pearson"], test_metrics["mae"], test_metrics["n"], out)
    return report


def main():
    ap = argparse.ArgumentParser(description="Train canonical absolute-Tm model on Meltome")
    ap.add_argument("--meltome", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    print(json.dumps(run(args.meltome, args.out), indent=2))


if __name__ == "__main__":
    main()
