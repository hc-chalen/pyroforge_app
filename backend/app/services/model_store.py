from __future__ import annotations

from pathlib import Path

import torch

from app.core.config import settings
from app.models.regression import DeltaMLP, ModelThermoDg
from app.models.tm_regressor import MELTOME_TM_MAX_LEN, MELTOME_TM_MIN_LEN


class ModelStore:
    """Loads the production checkpoints.

    * ``model_thermo`` — absolute folding dG (its Tm head is retired; see below).
    * ``model_ddg``   — difference-embedding ΔΔG (the served mutation-effect
      signal). If ``model_ddg`` is missing, ΔΔG falls back to dG(mut)-dG(wt).
    * ``model_tm`` (improved/model_tm.joblib) — the reliable absolute-Tm model
      (Meltome-trained, r≈0.81). Serves absolute Tm for both wild-type and
      mutant sequences (run directly on the mutant embedding).
    """

    def __init__(self) -> None:
        self.device = settings.resolved_device
        self.norm_stats: dict | None = None
        self.model_thermo: ModelThermoDg | None = None
        self.thermo_calibration: dict | None = None
        self.thermo_residual_std: dict | None = None
        self.model_ddg: DeltaMLP | None = None
        self.ddg_residual_std: dict | None = None
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

        # Reliable absolute-Tm model (Meltome-trained). Optional: if absent,
        # absolute Tm is reported as "unavailable".
        path_tm = checkpoint_dir / "improved" / "model_tm.joblib"
        if path_tm.exists():
            import joblib

            ckpt_tm = joblib.load(path_tm)
            self.model_tm = ckpt_tm["model"]
            self.tm_residual_std = (ckpt_tm.get("residual_std") or {}).get("tm")
            # Training length window for the in-distribution confidence check; fall
            # back to the documented Meltome ingestion defaults if not stored.
            self.tm_length_range = tuple(
                ckpt_tm.get("train_length_range") or (MELTOME_TM_MIN_LEN, MELTOME_TM_MAX_LEN)
            )
        self._loaded = True

    def normalize(self, embedding: torch.Tensor) -> torch.Tensor:
        if self.norm_stats is None:
            raise RuntimeError("Models not loaded")
        mean = torch.tensor(self.norm_stats["embedding_mean"], device=embedding.device)
        std = torch.tensor(self.norm_stats["embedding_std"], device=embedding.device)
        return (embedding - mean) / std


_model_store: ModelStore | None = None


def get_model_store() -> ModelStore:
    global _model_store
    if _model_store is None:
        _model_store = ModelStore()
    return _model_store
