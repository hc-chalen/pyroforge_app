"""Fast, data-free tests for the leakage-free parent split and WT anchoring.

These guard PyroForge's central correctness property: because one parent protein
contributes many near-identical mutant embeddings, the train/val split must hold
out *whole parents* — never let a parent appear on both sides — otherwise the
held-out metrics are inflated by memorized parent identity. No ``esm2_*.pt`` file
is needed; a tiny synthetic dataset exercises the same code paths.

Run directly (``python tests/test_dataset_split.py``) or with ``pytest``.
"""
from __future__ import annotations

import numpy as np
import torch

from app.data.dataset import (
    build_parent_split,
    build_split,
    wt_reference_index,
)


def _synthetic_data(n_parents: int = 12, muts_per_parent: int = 20, dim: int = 16) -> dict:
    """One wild-type row + several mutant rows per parent, distinct embeddings."""
    rng = np.random.default_rng(0)
    emb, parent, source, nsub = [], [], [], []
    for p in range(n_parents):
        base = rng.normal(size=dim)
        # wild-type anchor (num_substitutions == 0) then mutants
        emb.append(base); parent.append(p); source.append(p); nsub.append(0)
        for _ in range(muts_per_parent):
            emb.append(base + rng.normal(scale=0.01, size=dim))
            parent.append(p); source.append(p); nsub.append(1)
    return {
        "embeddings": torch.tensor(np.array(emb), dtype=torch.float32),
        "parent_protein_ids": torch.tensor(parent, dtype=torch.int64),
        "source_sequence_ids": torch.tensor(source, dtype=torch.int64),
        "num_substitutions": torch.tensor(nsub, dtype=torch.int64),
    }


def test_parents_never_leak_across_split() -> None:
    data = _synthetic_data()
    split = build_parent_split(data, seed=42, val_fraction=0.25)
    groups = data["parent_protein_ids"].numpy()
    train_parents = set(groups[split.train_idx].tolist())
    val_parents = set(groups[split.val_idx].tolist())
    assert train_parents.isdisjoint(val_parents), "a parent appears in both train and val"
    assert val_parents, "validation split is empty"


def test_split_partitions_all_rows() -> None:
    data = _synthetic_data()
    split = build_parent_split(data, seed=42, val_fraction=0.25)
    n = data["embeddings"].shape[0]
    both = np.concatenate([split.train_idx, split.val_idx])
    assert len(both) == n, "train+val do not cover every row"
    assert len(np.unique(both)) == n, "train and val overlap"


def test_normalization_stats_use_train_rows_only() -> None:
    data = _synthetic_data()
    split = build_parent_split(data, seed=42, val_fraction=0.25)
    train_emb = data["embeddings"][split.train_idx].numpy()
    assert np.allclose(split.embedding_mean, train_emb.mean(axis=0), atol=1e-5)
    assert (split.embedding_std >= 1e-6).all(), "zero-variance dims must be floored, not 0"


def test_build_split_delegates_to_parent_split() -> None:
    data = _synthetic_data()
    a = build_split(data, seed=7)
    b = build_parent_split(data, seed=7)
    assert np.array_equal(a.val_idx, b.val_idx)
    assert a.split_method.startswith("parent_group_validation")


def test_wt_reference_maps_mutants_to_their_anchor() -> None:
    data = _synthetic_data(n_parents=3, muts_per_parent=4)
    ref = wt_reference_index(data)
    sids = data["source_sequence_ids"].numpy()
    nsub = data["num_substitutions"].numpy()
    assert (ref >= 0).all(), "every row should resolve a WT anchor"
    for i, r in enumerate(ref):
        assert sids[r] == sids[i], "anchor must share the source sequence id"
        assert nsub[r] == 0, "anchor must be the 0-substitution wild-type row"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
