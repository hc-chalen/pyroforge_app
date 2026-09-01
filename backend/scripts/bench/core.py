"""Scoring methods and metrics for the external benchmark.

Every method here consumes the same ``Variant`` records and returns one array of
predicted ΔΔG per direction, so the comparison is strictly like-for-like.

Methods
-------
``model_ddg``   the served path. Embeddings come from ``ESM2Service`` and the
                feature construction is copied from
                ``PredictionEngine._ddg_from_embeddings`` — ``[mut_norm ⊕
                (mut_norm − wt_norm)]`` under ``model_ddg``'s own norm stats.
                Benchmarking the served construction rather than a
                training-time reimplementation is the whole point.

``esm_zeroshot``  frozen ESM-2 masked-marginal log-odds (Meier et al., NeurIPS
                2021): ``log P(mt | x_masked_at_i) − log P(wt | x_masked_at_i)``.
                No trained head at all. This is the baseline that decides
                whether ``model_ddg`` is earning its place, and it reuses
                ``ESM2Service.residue_probabilities`` — the same masked pass the
                optimizer already runs.

``random``      shuffled scores, the floor. Seeded.

``DDGun`` and the other published tools are not recomputed: their predictions on
these exact variants ship inside the S669 CSV.

Sign convention. PyroForge reports ΔΔG > 0 = stabilizing. S669's convention is
verified empirically at run time (see ``detect_sign_convention``) rather than
assumed, because getting it backwards would invert every conclusion.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Embedding cache
# --------------------------------------------------------------------------- #
class EmbedCache:
    """Disk cache of mean-pooled ESM-2 embeddings, keyed by sequence hash.

    A site-saturation-style benchmark re-embeds the same wild type once per
    variant; without this the run is dominated by redundant 650M forward passes.
    """

    def __init__(self, cache_dir: Path):
        self.dir = cache_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _path(self, seq: str) -> Path:
        return self.dir / f"{hashlib.sha1(seq.encode()).hexdigest()}.npy"

    def get_many(self, sequences: list[str], esm2, batch_size: int = 4) -> dict[str, np.ndarray]:
        """Embed every sequence, reading from and writing to disk as it goes."""
        out: dict[str, np.ndarray] = {}
        todo: list[str] = []
        for seq in dict.fromkeys(sequences):
            path = self._path(seq)
            if path.exists():
                out[seq] = np.load(path)
                self.hits += 1
            else:
                todo.append(seq)
        if not todo:
            return out
        # Sort by length so each padded batch is near-uniform — padding to the
        # longest member is pure waste and attention cost is O(L^2).
        todo.sort(key=len)
        started = time.time()
        for i in range(0, len(todo), batch_size):
            batch = todo[i:i + batch_size]
            embeddings = esm2.batch_embed(batch, batch_size=batch_size)
            for seq, emb in zip(batch, embeddings):
                arr = emb.numpy()
                np.save(self._path(seq), arr)
                out[seq] = arr
                self.misses += 1
            if i % (batch_size * 20) == 0:
                done = i + len(batch)
                rate = done / max(time.time() - started, 1e-6)
                logger.info("  embedded %d/%d (%.1f seq/s)", done, len(todo), rate)
        return out


# --------------------------------------------------------------------------- #
# Methods
# --------------------------------------------------------------------------- #
@torch.no_grad()
def model_ddg_scores(variants, embeddings, store) -> tuple[np.ndarray, np.ndarray]:
    """Served ΔΔG for the direct and inverse direction of every variant.

    The inverse direction simply swaps which embedding plays the wild-type role,
    which is exactly what a user would get by pasting the mutant as their input.
    """
    device = store.device

    def run(wt_seqs, mut_seqs):
        out = np.empty(len(wt_seqs), dtype=np.float32)
        for i in range(0, len(wt_seqs), 256):
            wt = torch.tensor(np.stack([embeddings[s] for s in wt_seqs[i:i + 256]]))
            mut = torch.tensor(np.stack([embeddings[s] for s in mut_seqs[i:i + 256]]))
            wt_norm = store.normalize_ddg(wt.to(device))
            mut_norm = store.normalize_ddg(mut.to(device))
            feat = torch.cat([mut_norm, mut_norm - wt_norm], dim=-1)
            out[i:i + 256] = store.model_ddg(feat).cpu().numpy().ravel()
        return out

    wt_seqs = [v.wt_seq for v in variants]
    mut_seqs = [v.mut_seq for v in variants]
    return run(wt_seqs, mut_seqs), run(mut_seqs, wt_seqs)


@torch.no_grad()
def esm_zeroshot_scores(variants, esm2, cache_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Masked-marginal log-odds, both directions (Meier et al. 2021).

    One masked forward pass per (sequence, position) yields the full 20-way
    distribution, so the direct score and the *inverse* score computed on the
    mutant background each cost one pass. Results are cached on disk because
    this is the run's dominant cost.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    def marginals(seq: str, pos0: int) -> dict[str, float]:
        key = hashlib.sha1(f"{seq}|{pos0}".encode()).hexdigest()
        path = cache_dir / f"{key}.npz"
        if path.exists():
            data = np.load(path, allow_pickle=True)
            return {str(k): float(v) for k, v in zip(data["aa"], data["p"])}
        probs = esm2.residue_probabilities(seq, pos0)
        np.savez(path, aa=np.array(list(probs)), p=np.array(list(probs.values())))
        return probs

    direct = np.empty(len(variants), dtype=np.float32)
    inverse = np.empty(len(variants), dtype=np.float32)
    started = time.time()
    for i, v in enumerate(variants):
        wt_probs = marginals(v.wt_seq, v.pos0)
        mut_probs = marginals(v.mut_seq, v.pos0)
        eps = 1e-9
        # Direct: how much more plausible is the mutant residue than the wild type,
        # judged on the wild-type background.
        direct[i] = np.log(wt_probs[v.mt_aa] + eps) - np.log(wt_probs[v.wt_aa] + eps)
        # Inverse: the same question on the mutant background, reversed.
        inverse[i] = np.log(mut_probs[v.wt_aa] + eps) - np.log(mut_probs[v.mt_aa] + eps)
        if i and i % 100 == 0:
            logger.info("  zero-shot %d/%d (%.1f var/s)", i, len(variants),
                        i / max(time.time() - started, 1e-6))
    return direct, inverse


def random_scores(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    direct = rng.standard_normal(n).astype(np.float32)
    return direct, -direct


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def detect_sign_convention(ddg_dir: np.ndarray) -> dict:
    """Establish empirically whether S669's positive ΔΔG means stabilizing.

    Single-point mutations are overwhelmingly destabilizing, so the sign of the
    bulk of the distribution identifies the convention without trusting any
    model's output (which would be circular).
    """
    frac_negative = float(np.mean(ddg_dir < 0))
    return {
        "mean_ddg_dir": float(np.mean(ddg_dir)),
        "median_ddg_dir": float(np.median(ddg_dir)),
        "fraction_negative": frac_negative,
        "positive_means_stabilizing": frac_negative > 0.5,
    }


def _finite(*arrays):
    """Drop positions where any array is non-finite.

    Not every published tool predicted every S669 variant, so their columns carry
    NaN. Dropping pairwise (rather than globally) keeps each method scored on all
    the variants it actually covered; ``n`` is reported per method so unequal
    coverage stays visible.
    """
    numeric = [np.asarray(a) for a in arrays]
    mask = np.ones(len(numeric[0]), dtype=bool)
    for arr in numeric:
        if arr.dtype.kind in "fc":
            mask &= np.isfinite(arr)
    return [arr[mask] for arr in numeric]


def basic_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    from scipy.stats import pearsonr, spearmanr

    pred, true = _finite(pred, true)
    if len(pred) < 3:
        return {"n": int(len(pred))}
    return {
        "n": int(len(pred)),
        "pearson_r": float(pearsonr(pred, true)[0]),
        "spearman_rho": float(spearmanr(pred, true).correlation),
        "rmse": float(np.sqrt(np.mean((pred - true) ** 2))),
        "mae": float(np.mean(np.abs(pred - true))),
    }


def antisymmetry(pred_dir: np.ndarray, pred_inv: np.ndarray) -> dict:
    """Bias statistics from Pucci et al. 2018.

    A thermodynamically consistent predictor satisfies ΔΔG_dir = −ΔΔG_inv, so
    ``r_dir_inv`` should be −1 and ``mean_sum`` (the bias δ) should be 0. A
    method that ignores direction scores the same value twice and lands at
    r = +1, which is the failure mode Pancotti 2022 links to poor performance on
    *stabilizing* variants specifically.
    """
    from scipy.stats import pearsonr

    finite = np.isfinite(pred_dir) & np.isfinite(pred_inv)
    d, v = pred_dir[finite], pred_inv[finite]
    total = d + v
    return {
        "n": int(len(d)),
        "r_dir_inv": float(pearsonr(d, v)[0]) if len(d) > 2 else float("nan"),
        "bias_mean_sum": float(np.mean(total)),
        "bias_std_sum": float(np.std(total)),
        "max_abs_sum": float(np.max(np.abs(total))) if len(total) else float("nan"),
    }


def within_protein_spearman(pred, true, proteins, min_n: int = 5) -> dict:
    """Distribution of per-protein Spearman — the statistic the optimizer relies on.

    Ranking mutations *within* one protein is what the shortlist actually does;
    a pooled correlation across proteins is dominated by between-protein spread.
    """
    from scipy.stats import spearmanr

    pred, true, proteins = _finite(pred, true, np.asarray(proteins))
    values = []
    for protein in np.unique(proteins):
        mask = proteins == protein
        if mask.sum() >= min_n and np.std(true[mask]) > 0 and np.std(pred[mask]) > 0:
            rho = spearmanr(pred[mask], true[mask]).correlation
            if np.isfinite(rho):
                values.append(float(rho))
    values = np.array(values)
    if not len(values):
        return {"n_proteins": 0}
    return {
        "n_proteins": int(len(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "q25": float(np.percentile(values, 25)),
        "q75": float(np.percentile(values, 75)),
        "values": values.tolist(),
    }


def paired_bootstrap(pred_a, pred_b, true, proteins, statistic, n_boot=2000, seed=0) -> dict:
    """Paired CI on (A − B), resampling *proteins*, not mutations.

    Mutations inside one protein are not independent — they share a background
    and a wild-type embedding — so resampling rows would understate the interval
    substantially. Pairing keeps both methods on the same resampled proteins.
    """
    rng = np.random.default_rng(seed)
    proteins = np.asarray(proteins)
    unique = np.unique(proteins)
    index = {p: np.flatnonzero(proteins == p) for p in unique}

    observed = statistic(pred_a, true) - statistic(pred_b, true)
    deltas = []
    for _ in range(n_boot):
        picked = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([index[p] for p in picked])
        try:
            deltas.append(statistic(pred_a[rows], true[rows]) - statistic(pred_b[rows], true[rows]))
        except Exception:
            continue
    deltas = np.array([d for d in deltas if np.isfinite(d)])
    if not len(deltas):
        return {"delta": float(observed)}
    return {
        "delta": float(observed),
        "ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
        # Fraction of resamples where the sign flips — a direct read on whether
        # the ordering of the two methods is stable.
        "p_sign_flip": float(np.mean(np.sign(deltas) != np.sign(observed))),
        "n_boot": int(len(deltas)),
    }


def paired_bootstrap_within(pred_a, pred_b, true, proteins, n_boot=2000, seed=0, min_n=5) -> dict:
    """Paired CI on the difference in *median within-protein* Spearman.

    This is the statistic the optimizer actually depends on — it ranks mutations
    inside a single protein and never compares levels across proteins. Pooled
    correlation is dominated by between-protein spread, which the difference-
    embedding feature deliberately cancels, so the two statistics can and do
    disagree. Resampling is over proteins, which is also the unit of the median.
    """
    rng = np.random.default_rng(seed)
    pred_a, pred_b, true, proteins = _finite(pred_a, pred_b, true, np.asarray(proteins))
    unique = np.unique(proteins)
    index = {p: np.flatnonzero(proteins == p) for p in unique}

    def median_rho(pred, picked):
        from scipy.stats import spearmanr

        values = []
        for p in picked:
            rows = index[p]
            if len(rows) >= min_n and np.std(true[rows]) > 0 and np.std(pred[rows]) > 0:
                rho = spearmanr(pred[rows], true[rows]).correlation
                if np.isfinite(rho):
                    values.append(rho)
        return float(np.median(values)) if values else np.nan

    observed = median_rho(pred_a, unique) - median_rho(pred_b, unique)
    deltas = []
    for _ in range(n_boot):
        picked = rng.choice(unique, size=len(unique), replace=True)
        delta = median_rho(pred_a, picked) - median_rho(pred_b, picked)
        if np.isfinite(delta):
            deltas.append(delta)
    deltas = np.array(deltas)
    if not len(deltas):
        return {"delta": float(observed)}
    return {
        "delta": float(observed),
        "ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
        "p_sign_flip": float(np.mean(np.sign(deltas) != np.sign(observed))),
        "n_boot": int(len(deltas)),
    }


def spearman_stat(pred, true) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(pred, true).correlation)


def stabilizing_detection(pred, true, threshold: float = 0.0) -> dict:
    """How well does the method find *stabilizing* variants — the product's job.

    S669 is heavily destabilizing, so overall correlation can look respectable
    while the tool is useless at the only task PyroForge is built for.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    pred, true = _finite(pred, true)
    labels = (true > threshold).astype(int)
    out = {"n_positive": int(labels.sum()), "base_rate": float(labels.mean())}
    if 0 < labels.sum() < len(labels):
        out["roc_auc"] = float(roc_auc_score(labels, pred))
        out["average_precision"] = float(average_precision_score(labels, pred))
    return out


def top_k_precision(pred, true, proteins, k: int = 5, threshold: float = 0.0) -> dict:
    """Precision of a per-protein top-k shortlist — the number a user acts on."""
    pred, true, proteins = _finite(pred, true, np.asarray(proteins))
    per_protein = []
    for protein in np.unique(proteins):
        mask = np.flatnonzero(proteins == protein)
        if len(mask) < k:
            continue
        top = mask[np.argsort(-pred[mask])[:k]]
        per_protein.append(float(np.mean(true[top] > threshold)))
    if not per_protein:
        return {"n_proteins": 0}
    return {
        "n_proteins": len(per_protein),
        "k": k,
        "precision": float(np.mean(per_protein)),
        "values": per_protein,
    }
