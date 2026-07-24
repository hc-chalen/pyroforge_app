"""Canonical absolute-Tm regressor (single-database, Meltome-trained).

Absolute Tm is a per-protein baseline regression (its variance is ~93%
between-protein), so it wants a regularized small/mid-data model, not a deep net.
This is a PCA-100 → (Ridge + gradient-boosting) ensemble over frozen ESM2-650M
mean-pooled embeddings — the recipe that reaches r≈0.81 / MAE≈5.9 °C once trained
on thousands of Meltome proteins.

It is deliberately trained on ONE database (Meltome Atlas). The original
tiny-domain Tm labels are a different assay (folding Tm on designed domains) that
does not transfer, so mixing them in adds no accuracy and only confuses the
target. Mutant sequences are scored by running this same model on the mutant
embedding; the mutation's stability effect is reported separately as ΔΔG.

Self-contained (stores its own normalization + PCA), so ``predict`` takes a raw
1280-d ESM2 embedding and returns °C.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

# Residue-length window model_tm is trained on (see ``ingest_meltome.select``
# defaults). A served sequence outside this window is out-of-distribution for
# absolute Tm, so its prediction should not be reported as high-confidence.
MELTOME_TM_MIN_LEN = 20
MELTOME_TM_MAX_LEN = 400


class TmRegressor:
    def __init__(self, n_components: int = 100, alpha: float = 50.0):
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.pca = PCA(n_components=n_components, random_state=0)
        self.ridge = Ridge(alpha=alpha)
        self.gbm = HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=400, l2_regularization=1.0, random_state=0
        )
        self.source = "meltome"

    def _features(self, emb: np.ndarray, fit: bool) -> np.ndarray:
        z = (emb - self.mean) / self.std
        return self.pca.fit_transform(z) if fit else self.pca.transform(z)

    def fit(self, emb: np.ndarray, tm: np.ndarray) -> "TmRegressor":
        emb = np.asarray(emb, dtype=np.float32)
        self.mean = emb.mean(0)
        self.std = emb.std(0)
        self.std[self.std < 1e-6] = 1.0
        X = self._features(emb, fit=True)
        self.ridge.fit(X, tm)
        self.gbm.fit(X, tm)
        return self

    def predict(self, emb: np.ndarray) -> np.ndarray:
        """Absolute Tm (°C) for one or many 1280-d ESM2 embeddings."""
        emb = np.asarray(emb, dtype=np.float32)
        single = emb.ndim == 1
        if single:
            emb = emb[None, :]
        X = self._features(emb, fit=False)
        out = 0.5 * (self.ridge.predict(X) + self.gbm.predict(X))
        return float(out[0]) if single else out


class TmAutoMLRegressor:
    """AutoML-selected absolute-Tm regressor — a drop-in for :class:`TmRegressor`.

    Wraps a fitted scikit-learn ``Pipeline`` (StandardScaler → PCA → estimator or
    stacking ensemble) chosen by ``app.training.automl_tm``. Preprocessing lives
    inside the pipeline, so ``predict`` takes a raw 1280-d ESM2 embedding (one or
    many) and returns °C — the same contract ``model_store`` relies on. Kept in
    this module so joblib can import the class at load time.
    """

    def __init__(self, pipeline, recipe: str = "automl"):
        self.pipeline = pipeline
        self.recipe = recipe
        self.source = "meltome"

    def predict(self, emb: np.ndarray):
        emb = np.asarray(emb, dtype=np.float32)
        single = emb.ndim == 1
        if single:
            emb = emb[None, :]
        out = np.asarray(self.pipeline.predict(emb), dtype=np.float64)
        return float(out[0]) if single else out
