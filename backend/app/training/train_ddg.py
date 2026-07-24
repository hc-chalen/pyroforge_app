"""Train the direct ΔΔG model on difference embeddings, tuned by autoPyTorch NAS.

ΔΔG is the primary signal the optimizer ranks on. Predicting it from the
*difference* embedding Δemb = emb(mut) − emb(wt) (rather than deriving it from
two absolute dG predictions, or regressing it from the mutant embedding alone)
lifts held-out correlation from ~0.5 to ~0.8 on unseen proteins.

Pipeline:
  1. Build [mutant ⊕ Δemb] features on the parent split (train file only).
  2. Run Hyperband NAS (autoPyTorch BOHB methodology) on a subsample to find the
     best MLP architecture.
  3. Refit that architecture on all anchored ΔΔG training rows — each augmented
     with its exact reverse mutation (see ``antisymmetric_augment``) — with early
     stopping on the parent-held-out validation set.
  4. Save ``model_ddg.pt`` (config + weights + norm stats + split + NAS report).

Antisymmetry: ΔΔG(A→B) must equal −ΔΔG(B→A). Training on forward mutations only
left the model badly antisymmetric (bias −1.29 kcal/mol, corr(fwd,−rev)=0.27).
Adding every mutation's reverse to the training set fixes this (bias ≈ 0, corr
≈ 0.95) at no measurable accuracy cost on the 266-protein grouped CV. This is the
model served as ``model_ddg.pt``; keep the augmentation so retraining reproduces
it rather than regressing to the forward-only variant.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, TensorDataset

from app.core.config import settings
from app.data.dataset import build_parent_split, load_raw_data
from app.models.regression import DeltaMLP
from app.training.features import build_ddg_delta_arrays, build_ddg_delta_eval_arrays
from app.training.hyperband_nas import HyperbandRegressor, NASConfig

logger = logging.getLogger(__name__)

FEATURE_SPEC = "concat_mut_delta"  # [mutant_norm ⊕ Δemb_norm]


def _pick_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _pooled(pred, true) -> dict:
    return {
        "n": int(len(pred)),
        "mae": float(np.mean(np.abs(pred - true))),
        "rmse": float(np.sqrt(np.mean((pred - true) ** 2))),
        "pearson_r": float(pearsonr(pred, true)[0]) if len(pred) > 1 else float("nan"),
    }


def within_protein(pred, true, pid, min_n: int = 8) -> dict:
    """Median within-protein Spearman — the metric that matches how the
    optimizer uses the model (ranking mutations inside one protein)."""
    rs = []
    for p in np.unique(pid):
        m = pid == p
        if m.sum() >= min_n and np.std(pred[m]) > 0 and np.std(true[m]) > 0:
            r = spearmanr(pred[m], true[m]).correlation
            if not np.isnan(r):
                rs.append(r)
    rs = np.array(rs)
    return {"median_spearman": float(np.median(rs)) if len(rs) else float("nan"),
            "n_proteins": int(len(rs))}


def antisymmetric_augment(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Append each mutation's exact reverse to enforce ΔΔG(A→B) = −ΔΔG(B→A).

    Features are ``[mutant_norm ⊕ Δemb_norm]`` where (see
    ``features._delta_features``) ``mutant_norm = (emb_mut − μ)/σ`` and
    ``Δemb_norm = (emb_mut − emb_wt)/σ``. The reverse mutation swaps mutant and
    wild-type, so its features are constructed exactly from the forward row:

        reverse mutant_norm = (emb_wt − μ)/σ = mutant_norm − Δemb_norm
        reverse Δemb_norm    = (emb_wt − emb_mut)/σ = −Δemb_norm
        reverse label        = −y

    No embeddings are re-read; the reverse is algebraically exact.
    """
    d = x.shape[1] // 2
    mut, delta = x[:, :d], x[:, d:]
    x_rev = np.concatenate([mut - delta, -delta], axis=1).astype(np.float32)
    return np.concatenate([x, x_rev], axis=0), np.concatenate([y, -y]).astype(np.float32)


def _train_full(config, x_tr, y_tr, x_va, y_va, device, epochs, patience) -> DeltaMLP:
    model = DeltaMLP(config["input_dim"], config["widths"], config["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    mse = torch.nn.MSELoss()
    ds = TensorDataset(torch.tensor(x_tr), torch.tensor(y_tr))
    loader = DataLoader(ds, batch_size=config["batch_size"], shuffle=True)
    x_va_t = torch.tensor(x_va, device=device)

    best_r, best_state, best_ep = -9.0, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            loss = mse(model(xb.to(device)), yb.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(x_va_t).cpu().numpy()
        r = float(pearsonr(pv, y_va)[0])
        logger.info("model_ddg refit epoch %2d/%d | val ΔΔG r=%.3f", ep, epochs, r)
        if r > best_r:
            best_r, best_ep = r, ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif ep - best_ep >= patience:
            logger.info("model_ddg early stop at epoch %d (best %d, r=%.3f)", ep, best_ep, best_r)
            break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def _predict(model, x, device, bs=16384) -> np.ndarray:
    out = np.empty(len(x), dtype=np.float32)
    for s in range(0, len(x), bs):
        e = min(s + bs, len(x))
        out[s:e] = model(torch.tensor(x[s:e], device=device)).cpu().numpy()
    return out


def run_training(
    data_path: Path | None = None,
    test_path: Path | None = None,
    checkpoint_dir: Path | None = None,
    device: str | None = None,
    nas_time: int = 420,
    nas_subsample: int = 100_000,
    refit_epochs: int = 40,
    patience: int = 6,
    seed: int = 42,
) -> dict:
    data_path = data_path or settings.data_path
    test_path = test_path or settings.test_data_path
    checkpoint_dir = checkpoint_dir or settings.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    device = _pick_device(device)

    data = load_raw_data(data_path)
    split = build_parent_split(data, seed=seed)
    mean, std = split.embedding_mean, split.embedding_std

    x_tr, y_tr, x_va, y_va, pid_va = build_ddg_delta_arrays(data, split, concat_mut=True)
    # Enforce ΔΔG antisymmetry by training on each mutation and its exact reverse.
    x_tr, y_tr = antisymmetric_augment(x_tr, y_tr)
    logger.info("ΔΔG anchored rows — train %d (incl. reverse aug), val %d, features %d-d",
                len(y_tr), len(y_va), x_tr.shape[1])

    # 1) autoPyTorch NAS on a subsample to find the architecture
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(x_tr), min(nas_subsample, len(x_tr)), replace=False)
    logger.info("running autoPyTorch NAS on %d-row subsample (%ds budget) ...", len(sub), nas_time)
    nas = HyperbandRegressor(NASConfig(max_runtime=nas_time, seed=seed), device=device)
    nas.fit(x_tr[sub], y_tr[sub], x_va, y_va)
    best = {k: v for k, v in nas.best_trial.config.items() if not k.startswith("_")}
    logger.info("NAS best config: widths=%s dropout=%.3f lr=%.2e wd=%.2e bs=%d",
                best["widths"], best["dropout"], best["lr"], best["weight_decay"], best["batch_size"])

    # 2) refit best architecture on ALL anchored training rows
    model = _train_full(best, x_tr, y_tr, x_va, y_va, device, refit_epochs, patience)

    # 3) evaluate
    pv = _predict(model, x_va, device)
    val_metrics = {**_pooled(pv, y_va), "within_protein": within_protein(pv, y_va, pid_va)}

    test_data = load_raw_data(test_path)
    x_te, y_te, pid_te = build_ddg_delta_eval_arrays(
        test_data, np.arange(test_data["embeddings"].shape[0]), mean, std, concat_mut=True
    )
    pt = _predict(model, x_te, device)
    test_metrics = {**_pooled(pt, y_te), "within_protein": within_protein(pt, y_te, pid_te)}

    ddg_res = float(np.std(pv - y_va))
    residual_std = {"ddg": ddg_res}

    ckpt = {
        "model_name": "model_ddg",
        "feature_spec": FEATURE_SPEC,
        "config": {"input_dim": int(x_tr.shape[1]), "widths": best["widths"], "dropout": best["dropout"]},
        "state_dict": model.state_dict(),
        "norm_stats": {"embedding_mean": mean.tolist(), "embedding_std": std.tolist()},
        "residual_std": residual_std,
        "split": split.manifest(),
        "nas": {"best_config": best, "search_history": nas.search_history, "budget_secs": nas_time},
    }
    torch.save(ckpt, checkpoint_dir / "model_ddg.pt")

    report = {
        "model_ddg": {"val_metrics": val_metrics, "test_metrics": test_metrics, "nas_best_config": best},
        "device": device,
        "feature_spec": FEATURE_SPEC,
        "split": split.manifest(),
    }
    with open(checkpoint_dir / "ddg_training_report.json", "w") as f:
        json.dump(report, f, indent=2)
    logger.info("model_ddg — val r=%.3f, test r=%.3f (within-protein test median ρ=%.3f over %d proteins)",
                val_metrics["pearson_r"], test_metrics["pearson_r"],
                test_metrics["within_protein"]["median_spearman"], test_metrics["within_protein"]["n_proteins"])
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Train ΔΔG model on difference embeddings via autoPyTorch NAS")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--nas-time", type=int, default=420, help="NAS wall-clock budget (s)")
    p.add_argument("--nas-subsample", type=int, default=100_000)
    p.add_argument("--refit-epochs", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    report = run_training(
        device=args.device, nas_time=args.nas_time, nas_subsample=args.nas_subsample,
        refit_epochs=args.refit_epochs, seed=args.seed,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
