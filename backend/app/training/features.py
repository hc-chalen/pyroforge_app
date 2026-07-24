"""Feature matrices for autoPyTorch tabular regression."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.data.dataset import DataSplit, load_raw_data, wt_reference_index

EMBEDDING_DIM = 1280


# --------------------------------------------------------------------------- #
# Difference-embedding features for direct ΔΔG regression
# --------------------------------------------------------------------------- #
def _delta_features(
    data: dict,
    rows: np.ndarray,
    wt_ref: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    concat_mut: bool,
) -> np.ndarray:
    """[mutant_norm ⊕ Δemb_norm] for the given rows.

    Δemb_norm = (emb(mut) − emb(wt)) / std, so the per-protein offset cancels.
    """
    emb = data["embeddings"].numpy()
    mut_norm = (emb[rows] - mean) / std
    delta_norm = (emb[rows] - emb[wt_ref[rows]]) / std
    if concat_mut:
        return np.concatenate([mut_norm, delta_norm], axis=1).astype(np.float32)
    return delta_norm.astype(np.float32)


def anchored_ddg_rows(data: dict, indices: np.ndarray, wt_ref: np.ndarray) -> np.ndarray:
    """Mutant rows within ``indices`` that have a ddG label and a WT anchor."""
    ddg = data["ddg"].numpy()
    nsub = data["num_substitutions"].numpy()
    mask = (nsub[indices] > 0) & ~np.isnan(ddg[indices]) & (wt_ref[indices] >= 0)
    return indices[mask]


def build_ddg_delta_arrays(
    data: dict, split: DataSplit, concat_mut: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train/val difference-embedding arrays for ΔΔG (parent split)."""
    wt_ref = wt_reference_index(data)
    tr = anchored_ddg_rows(data, split.train_idx, wt_ref)
    va = anchored_ddg_rows(data, split.val_idx, wt_ref)
    x_tr = _delta_features(data, tr, wt_ref, split.embedding_mean, split.embedding_std, concat_mut)
    x_va = _delta_features(data, va, wt_ref, split.embedding_mean, split.embedding_std, concat_mut)
    y_tr = data["ddg"].numpy()[tr].astype(np.float32)
    y_va = data["ddg"].numpy()[va].astype(np.float32)
    pid_va = data["parent_protein_ids"].numpy()[va]
    return x_tr, y_tr, x_va, y_va, pid_va


def build_ddg_delta_eval_arrays(
    data: dict, indices: np.ndarray, mean: np.ndarray, std: np.ndarray, concat_mut: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Difference-embedding arrays + parent ids for ΔΔG evaluation on a file."""
    wt_ref = wt_reference_index(data)
    rows = anchored_ddg_rows(data, indices, wt_ref)
    x = _delta_features(data, rows, wt_ref, mean, std, concat_mut)
    y = data["ddg"].numpy()[rows].astype(np.float32)
    pid = data["parent_protein_ids"].numpy()[rows]
    return x, y, pid


def _normalize_embeddings(embeddings: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (embeddings - mean) / std


def _embedding_column_names() -> list[str]:
    return [f"emb_{i}" for i in range(EMBEDDING_DIM)]


def build_model_dg_frames(
    data: dict,
    split: DataSplit,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Build train/val feature frames for absolute dG (thermo model)."""
    train_idx = split.train_idx
    val_idx = split.val_idx

    train_mask = ~np.isnan(data["dg"][train_idx].numpy())
    val_mask = ~np.isnan(data["dg"][val_idx].numpy())
    train_idx = train_idx[train_mask]
    val_idx = val_idx[val_mask]

    train_emb = _normalize_embeddings(
        data["embeddings"][train_idx].numpy(),
        split.embedding_mean,
        split.embedding_std,
    )
    val_emb = _normalize_embeddings(
        data["embeddings"][val_idx].numpy(),
        split.embedding_mean,
        split.embedding_std,
    )

    col_names = _embedding_column_names()
    x_train = pd.DataFrame(train_emb, columns=col_names)
    x_val = pd.DataFrame(val_emb, columns=col_names)
    y_train = pd.Series(data["dg"][train_idx].numpy(), name="dg")
    y_val = pd.Series(data["dg"][val_idx].numpy(), name="dg")
    return x_train, y_train, x_val, y_val


def build_dg_eval_frame(
    data: dict,
    split: DataSplit,
    indices: np.ndarray,
) -> tuple[pd.DataFrame, pd.Series]:
    dg_mask = ~np.isnan(data["dg"][indices].numpy())
    eval_idx = indices[dg_mask]
    emb = _normalize_embeddings(
        data["embeddings"][eval_idx].numpy(),
        split.embedding_mean,
        split.embedding_std,
    )
    col_names = _embedding_column_names()
    x = pd.DataFrame(emb, columns=col_names)
    y = pd.Series(data["dg"][eval_idx].numpy(), name="dg")
    return x, y


def load_data_and_split(data_path):
    from pathlib import Path

    data = load_raw_data(Path(data_path))
    from app.data.dataset import build_split

    split = build_split(data)
    return data, split
