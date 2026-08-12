from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from app.core.config import settings
from app.models.regression import DeltaMLP, ModelThermoDg
from app.models.tm_regressor import MELTOME_TM_MAX_LEN, MELTOME_TM_MIN_LEN

logger = logging.getLogger(__name__)


class ModelStore:
    """Loads the production checkpoints.

    * ``model_thermo`` — absolute folding dG (its Tm head is retired; see below).
    * ``model_ddg``   — difference-embedding ΔΔG (the served mutation-effect
      signal). If ``model_ddg`` is missing, ΔΔG falls back to dG(mut)-dG(wt).
    * ``model_tm`` (improved/model_tm.joblib) — the absolute-Tm model, Meltome-trained.
      Whatever recipe ``automl_tm`` promoted is what ships (currently an AutoML
      stacking pipeline, ``recipe="automl:stack"``, not the original PCA+Ridge+GBM
      ``TmRegressor``); read ``recipe`` and ``test_metrics`` from the checkpoint rather
      than trusting a number in a docstring. Serves absolute Tm for both wild-type and
      mutant sequences (run directly on the mutant embedding). Note its test split is
      FLIP ``mixed_split`` with exact-sequence dedup only, so its metrics are
      optimistic for novel proteins — see ``ingest_meltome.select``.
    """

    def __init__(self) -> None:
        self.device = settings.resolved_device
        self.norm_stats: dict | None = None
        self.ddg_norm_stats: dict | None = None
        self.norm_stats_match: bool | None = None
        self._norm_cache: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}
        self.model_thermo: ModelThermoDg | None = None
        self.thermo_calibration: dict | None = None
        self.thermo_residual_std: dict | None = None
        self.model_ddg: DeltaMLP | None = None
        self.ddg_residual_std: dict | None = None
        # Fitted by app.training.calibrate_ddg on the parent-held-out split.
        self.ddg_calibration: dict | None = None
        # Measured top-k ranking precision — what the optimizer's shortlist is
        # actually worth, reported to the user in place of a σ claim.
        self.ddg_ranking: dict | None = None
        self.ddg_ranking_source: str | None = None
        self.model_tm = None  # app.models.tm_regressor.TmRegressor (reliable abs Tm)
        self.tm_residual_std: float | None = None
        self.tm_length_range: tuple[int, int] | None = None
        self._loaded = False

    @property
    def is_ready(self) -> bool:
        return self._loaded

    def load(self, checkpoint_dir: Path | None = None) -> None:
        checkpoint_dir = checkpoint_dir or settings.checkpoint_dir
        path_thermo = checkpoint_dir / "model_thermo.pt"
        if not path_thermo.exists():
            raise FileNotFoundError(
                f"Checkpoints not found in {checkpoint_dir}. Run training first."
            )

        ckpt_thermo = torch.load(path_thermo, map_location=self.device, weights_only=False)
        self.norm_stats = ckpt_thermo["norm_stats"]

        tcfg = ckpt_thermo.get("config") or {}
        self.model_thermo = ModelThermoDg(
            dg_widths=tcfg.get("dg_widths"),
            dropout=tcfg.get("dropout", 0.2),
            n_dg=tcfg.get("n_dg", 3),
        ).to(self.device)
        self.model_thermo.load_state_dict(ckpt_thermo["state_dict"])
        self.model_thermo.eval()
        self.thermo_calibration = ckpt_thermo.get("calibration")
        self.thermo_residual_std = ckpt_thermo.get("residual_std")

        path_ddg = checkpoint_dir / "model_ddg.pt"
        if path_ddg.exists():
            ckpt_ddg = torch.load(path_ddg, map_location=self.device, weights_only=False)
            cfg = ckpt_ddg["config"]
            self.model_ddg = DeltaMLP(cfg["input_dim"], cfg["widths"], cfg["dropout"]).to(self.device)
            self.model_ddg.load_state_dict(ckpt_ddg["state_dict"])
            self.model_ddg.eval()
            self.ddg_residual_std = ckpt_ddg.get("residual_std")
            # model_ddg was standardized with the stats from *its own* training split.
            # Using model_thermo's instead silently shifts every ΔΔG feature the
            # moment the two are trained on different data or seeds, with no error.
            self.ddg_norm_stats = ckpt_ddg.get("norm_stats") or self.norm_stats
            self._load_ddg_calibration(checkpoint_dir)
            self.norm_stats_match = self._stats_match(self.norm_stats, self.ddg_norm_stats)
            if not self.norm_stats_match:
                logger.warning(
                    "model_thermo and model_ddg carry different embedding norm stats "
                    "(different training split, seed or data file). ΔΔG will use its "
                    "own stats, as it must — but the two checkpoints are no longer a "
                    "matched pair; retrain both from the same split."
                )

        # Reliable absolute-Tm model (Meltome-trained). Optional: if absent,
        # absolute Tm is reported as "unavailable".
        path_tm = checkpoint_dir / "improved" / "model_tm.joblib"
        if path_tm.exists():
            import joblib

            ckpt_tm = joblib.load(path_tm)
            self.model_tm = ckpt_tm["model"]
            # Prefer the held-out test RMSE over the stored residual_std: older
            # checkpoints computed that residual on the training rows, which
            # understates the served interval
            # RMSE = sqrt(std² + bias²), a conservative and honest choice for a ±95% band
            self.tm_residual_std = (ckpt_tm.get("test_metrics") or {}).get("rmse") or (
                ckpt_tm.get("residual_std") or {}
            ).get("tm")
            # Training length window for the in-distribution confidence check; fall
            # back to the documented Meltome ingestion defaults if not stored.
            self.tm_length_range = tuple(
                ckpt_tm.get("train_length_range") or (MELTOME_TM_MIN_LEN, MELTOME_TM_MAX_LEN)
            )
        self._loaded = True

    def _load_ddg_calibration(self, checkpoint_dir: Path) -> None:
        """Load ``model_ddg_calibration.json`` (see app.training.calibrate_ddg).

        Carries two things the raw checkpoint cannot: whether a level
        recalibration is warranted (measured, and currently it is not), and the
        *measured* within-protein ranking quality that the optimizer reports to
        the user in place of the retired significance-gate claim.
        """
        path = checkpoint_dir / "model_ddg_calibration.json"
        if not path.exists():
            logger.warning(
                "No model_ddg_calibration.json in %s — the optimizer will report no "
                "measured ranking quality. Run `uv run pyroforge-calibrate-ddg`.",
                checkpoint_dir,
            )
            return
        try:
            report = json.loads(path.read_text())
        except json.JSONDecodeError:
            logger.exception("model_ddg_calibration.json is not valid JSON; ignoring it.")
            return

        cal = report.get("calibration") or {}
        if cal.get("applied") and "slope" in cal:
            self.ddg_calibration = {
                "slope": float(cal["slope"]),
                "intercept": float(cal["intercept"]),
            }
        # The residual measured around whichever prediction is actually served
        # supersedes the checkpoint's, which predates the calibration fit.
        residual = cal.get("residual_std" if cal.get("applied") else "raw_residual_std")
        if residual is not None:
            self.ddg_residual_std = {"ddg": float(residual)}
        self.ddg_ranking = report.get("validation_ranking")
        self.ddg_ranking_source = (report.get("split") or {}).get("split_method")

    def calibrate_ddg(self, value):
        """Apply the served ΔΔG level calibration, if one was found to be warranted.

        Accepts a float or an array. Identity when no calibration is applied —
        which is the current measured state: the fitted slope sits within noise of
        1.0, so there is no over-dispersion to correct and applying it would only
        add an unauditable transformation. See ``calibration.applied_reason``.
        """
        cal = self.ddg_calibration
        if not cal:
            return value
        return cal["slope"] * value + cal["intercept"]

    @staticmethod
    def _stats_match(a: dict | None, b: dict | None) -> bool:
        if a is None or b is None:
            return a is b
        for key in ("embedding_mean", "embedding_std"):
            va, vb = a.get(key), b.get(key)
            if va is None or vb is None or len(va) != len(vb):
                return False
            if not torch.allclose(
                torch.tensor(va, dtype=torch.float64),
                torch.tensor(vb, dtype=torch.float64),
                atol=1e-6,
            ):
                return False
        return True

    def _tensors(self, stats: dict, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
        """Cached (mean, std) tensors — rebuilt from a 1280-element Python list on
        every call otherwise, once per position of a saturation scan."""
        key = (id(stats), str(device))
        cached = self._norm_cache.get(key)
        if cached is None:
            cached = (
                torch.tensor(stats["embedding_mean"], device=device),
                torch.tensor(stats["embedding_std"], device=device),
            )
            self._norm_cache[key] = cached
        return cached

    def normalize(self, embedding: torch.Tensor) -> torch.Tensor:
        """Standardize with model_thermo's stats (the absolute-dG feature space)."""
        if self.norm_stats is None:
            raise RuntimeError("Models not loaded")
        mean, std = self._tensors(self.norm_stats, embedding.device)
        return (embedding - mean) / std

    def normalize_ddg(self, embedding: torch.Tensor) -> torch.Tensor:
        """Standardize with model_ddg's own stats (the ΔΔG feature space).

        Falls back to model_thermo's stats only when the ΔΔG checkpoint stores none.
        """
        stats = self.ddg_norm_stats or self.norm_stats
        if stats is None:
            raise RuntimeError("Models not loaded")
        mean, std = self._tensors(stats, embedding.device)
        return (embedding - mean) / std


_model_store: ModelStore | None = None


def get_model_store() -> ModelStore:
    global _model_store
    if _model_store is None:
        _model_store = ModelStore()
    return _model_store
