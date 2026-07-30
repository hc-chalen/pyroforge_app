"""Training utilities for the dG-ensemble thermo model (model_thermo)."""

from __future__ import annotations

import logging

import numpy as np
import torch
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, TensorDataset

from app.models.regression import ModelThermoDg

logger = logging.getLogger(__name__)


def metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    if len(pred) == 0:
        return {"n": 0, "mae": float("nan"), "rmse": float("nan"), "pearson_r": float("nan")}
    mae = float(np.mean(np.abs(pred - target)))
    rmse = float(np.sqrt(np.mean((pred - target) ** 2)))
    # pearsonr is undefined when either side is constant (e.g. tiny val slices).
    if len(pred) > 1 and np.std(pred) > 0 and np.std(target) > 0:
        r = float(pearsonr(pred, target)[0])
    else:
        r = float("nan")
    return {"n": int(len(pred)), "mae": mae, "rmse": rmse, "pearson_r": r}


def _single_task_loader(data, indices, label, mean, std, bs, shuffle):
    y = data[label][indices].float().numpy()
    keep = ~np.isnan(y)
    idx = indices[keep]
    emb = ((data["embeddings"][idx].numpy() - mean) / std).astype(np.float32)
    ds = TensorDataset(torch.tensor(emb), torch.tensor(y[keep], dtype=torch.float32))
    return DataLoader(ds, batch_size=bs, shuffle=shuffle)


def _train_tower(label, widths, data, split, device, epochs, bs, lr, wd, dropout, patience, seed=0):
    """Single-task :class:`_Tower` for one absolute target; best epoch by val r."""
    from app.models.regression import _Tower

    torch.manual_seed(seed)
    mean, std = split.embedding_mean, split.embedding_std
    tr = _single_task_loader(data, split.train_idx, label, mean, std, bs, True)
    va = _single_task_loader(data, split.val_idx, label, mean, std, bs, False)

    tower = _Tower(1280, widths, dropout).to(device)
    opt = torch.optim.AdamW(tower.parameters(), lr=lr, weight_decay=wd)
    mse = torch.nn.MSELoss()

    @torch.no_grad()
    def val_metrics():
        tower.eval(); P, T = [], []
        for xb, yb in va:
            P.extend(tower(xb.to(device)).cpu().numpy()); T.extend(yb.numpy())
        return metrics(np.array(P), np.array(T))

    best_r, best_ep, best_state, history = -9.0, 0, None, []
    for ep in range(1, epochs + 1):
        tower.train()
        for xb, yb in tr:
            loss = mse(tower(xb.to(device)), yb.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
        m = val_metrics()
        history.append({"epoch": ep, label: m})
        logger.info("model_thermo[%s#%d] epoch %2d/%d | val r=%.3f (n=%d)", label, seed, ep, epochs, m["pearson_r"], m["n"])
        if m["pearson_r"] > best_r:
            best_r, best_ep = m["pearson_r"], ep
            best_state = {k: v.cpu().clone() for k, v in tower.state_dict().items()}
        elif ep - best_ep >= patience:
            logger.info("model_thermo[%s#%d] early stop at %d (best %d, r=%.3f)", label, seed, ep, best_ep, best_r)
            break
    return best_state, history, best_r


@torch.no_grad()
def _ensemble_val_metrics(model, data, split, device, bs) -> dict[str, float]:
    """Validation metrics of the averaged ensemble (not of its members)."""
    model.eval()
    va = _single_task_loader(data, split.val_idx, "dg", split.embedding_mean,
                             split.embedding_std, bs, False)
    P, T = [], []
    for xb, yb in va:
        P.extend(model(xb.to(device)).cpu().numpy()); T.extend(yb.numpy())
    return metrics(np.array(P), np.array(T))


def train_model_thermo(
    data: dict,
    split,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float = 1e-4,
    dg_widths: list[int] | None = None,
    dropout: float = 0.2,
    n_dg: int = 3,
) -> tuple[ModelThermoDg, dict]:
    """Train model_thermo — an averaged ensemble of dG towers.

    Absolute Tm is no longer trained here; it is served by the Meltome-trained
    ``model_tm``. Each tower is trained single-task on the dG labels with a
    different seed; the checkpoint averages them at inference.
    """
    from app.models.regression import DEFAULT_DG_WIDTHS

    dg_widths = dg_widths or DEFAULT_DG_WIDTHS

    # dG ensemble: n_dg large towers with different seeds, averaged at inference.
    # ``patience`` must be < epochs or the early-stop branch is unreachable
    # (``ep - best_ep >= epochs`` cannot hold within ``epochs`` iterations).
    patience = max(3, epochs // 5)
    dg_states, dg_rs = [], []
    for s in range(n_dg):
        sd, _, r = _train_tower("dg", dg_widths, data, split, device, epochs, batch_size,
                                lr, weight_decay, dropout, patience=patience, seed=s)
        dg_states.append(sd); dg_rs.append(r)

    model = ModelThermoDg(dg_widths=dg_widths, dropout=dropout, n_dg=n_dg).to(device)
    for tower, sd in zip(model.dg_towers, dg_states):
        tower.load_state_dict(sd)
    model.eval()

    # Score the *ensemble* — the thing actually served. The mean of the individual
    # towers' val r is a different (and lower) quantity, and reporting it as the
    # model's metric means nothing in the checkpoint ever measured what ships.
    ens = _ensemble_val_metrics(model, data, split, device, batch_size)

    return model, {
        "val_metrics": {"dg": ens},
        "history": {"dg_tower_val_r": dg_rs, "dg_tower_mean_val_r": float(np.mean(dg_rs))},
        "config": {"dg_widths": dg_widths, "dropout": dropout, "n_dg": n_dg,
                   "early_stop_patience": patience},
    }
