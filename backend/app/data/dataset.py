from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit

# Column carrying the wild-type parent protein identity. Splitting on this key
# (rather than a random row split) guarantees every mutant of a given parent
# lands entirely in train *or* val — never both — so validation measures
# generalization to unseen proteins instead of memorized parent embeddings.
PARENT_KEY = "parent_protein_ids"


@dataclass
class DataSplit:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    embedding_mean: np.ndarray
    embedding_std: np.ndarray
    seed: int = 42
    split_method: str = "random"
    train_parents: np.ndarray | None = None
    val_parents: np.ndarray | None = None

    def manifest(self) -> dict:
        """Serializable split description stored alongside checkpoints."""
        tp = 0 if self.train_parents is None else int(len(self.train_parents))
        vp = 0 if self.val_parents is None else int(len(self.val_parents))
        return {
            "split_method": self.split_method,
            "seed": int(self.seed),
            "train_rows": int(len(self.train_idx)),
            "val_rows": int(len(self.val_idx)),
            "test_rows": int(len(self.test_idx)),
            "train_parent_count": tp,
            "val_parent_count": vp,
            "total_parent_count": tp + vp,
        }


def load_raw_data(data_path: Path) -> dict:
    return torch.load(data_path, map_location="cpu", weights_only=False)


def wt_reference_index(data: dict) -> np.ndarray:
    """Row index of each row's wild-type anchor (the 0-substitution row that
    shares its ``source_sequence_ids``), or -1 when none exists.

    Used to build difference embeddings Δemb = emb(mutant) − emb(wild-type),
    which cancel the per-protein offset that dominates a mean-pooled embedding
    and washes out a single-residue change.
    """
    sids = data["source_sequence_ids"].numpy()
    nsub = data["num_substitutions"].numpy()
    wt: dict[int, int] = {}
    for i in range(len(sids)):
        if nsub[i] == 0 and int(sids[i]) not in wt:
            wt[int(sids[i])] = i
    ref = np.full(len(sids), -1, dtype=np.int64)
    for i in range(len(sids)):
        r = wt.get(int(sids[i]))
        if r is not None:
            ref[i] = r
    return ref


def _embedding_stats(data: dict, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standardization stats computed on the *training rows only* (no leakage)."""
    train_embeddings = data["embeddings"][train_idx].numpy()
    mean = train_embeddings.mean(axis=0)
    std = train_embeddings.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean, std


def build_parent_split(data: dict, seed: int = 42, val_fraction: float = 0.1) -> DataSplit:
    """Group the rows by parent protein and hold out whole parents for validation.

    This is the leakage-free replacement for the old random split: because a
    single parent contributes many near-identical mutant embeddings, a random
    row split lets the model see a parent in training and be graded on the same
    parent in validation, badly inflating the score.
    """
    if PARENT_KEY not in data:
        # esm2_50k.pt ships without this column, so it cannot be used for training at
        # all — a random row split would leak parents across train/val and inflate
        # every metric. Fail with an actionable message rather than a bare KeyError:
        # this surfaces through /admin/train, whose Docker default points here.
        raise ValueError(
            f"Training dataset is missing the {PARENT_KEY!r} column, so a "
            f"leakage-free parent-group split cannot be built. Available columns: "
            f"{sorted(k for k, v in data.items() if hasattr(v, 'shape'))}. Point "
            f"DATA_PATH at a dataset that carries {PARENT_KEY!r} (esm2_train.pt); "
            f"the esm2_50k.pt subset does not."
        )
    n = data["embeddings"].shape[0]
    indices = np.arange(n)
    groups = data[PARENT_KEY].numpy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(splitter.split(indices, groups=groups))
    train_idx = np.sort(train_idx)
    val_idx = np.sort(val_idx)

    mean, std = _embedding_stats(data, train_idx)
    return DataSplit(
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=np.array([], dtype=np.int64),
        embedding_mean=mean,
        embedding_std=std,
        seed=seed,
        split_method=f"parent_group_validation:{PARENT_KEY}",
        train_parents=np.unique(groups[train_idx]),
        val_parents=np.unique(groups[val_idx]),
    )


def build_split(data: dict, seed: int = 42) -> DataSplit:
    """Default split used across training/eval — now parent-grouped.

    Kept as the historical entry point, but delegates to
    :func:`build_parent_split` so every caller gets the leakage-free split.
    """
    return build_parent_split(data, seed=seed)


