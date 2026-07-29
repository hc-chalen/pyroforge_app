from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from app.core.config import settings
from app.services.esm2 import ESM2Service, get_esm2_service
from app.services.model_store import get_model_store


@dataclass
class Mutation:
    position: int
    source_aa: str
    target_aa: str


class PredictionEngine:
    """Core prediction engine.

    Served outputs:
      * wild-type → absolute **Tm** (``model_tm``) and **dG** (``model_thermo``).
      * mutant    → absolute **Tm** (``model_tm`` on the mutant sequence) and
        **ΔΔG** (``model_ddg``, difference embedding).

    ΔΔG is the validated mutation-effect signal. The mutant's absolute Tm is read
    straight from ``model_tm`` on the mutant sequence — but a single point mutation
    barely moves the mean-pooled embedding, so read mutant Tm as "absolute Tm of
    this sequence", not as evidence of a thermal shift; use ΔΔG for the effect.
    """

    def __init__(self, esm2: ESM2Service | None = None) -> None:
        self.esm2 = esm2
        self.store = get_model_store()

    def _get_esm2(self) -> ESM2Service:
        if self.esm2 is None:
            self.esm2 = get_esm2_service()
        return self.esm2

    def ensure_ready(self) -> None:
        if not self.store.is_ready:
            self.store.load()

    def _tm_confidence(self, n_residues: int | None) -> dict[str, object]:
        """In-distribution flag for a model_tm prediction.

        model_tm only saw proteins within a fixed residue-length window (Meltome
        ingestion filter). A sequence outside that window is extrapolation, so it is
        flagged ``low`` rather than reported as ``high`` unconditionally. Length is a
        coarse proxy, but it is the honest signal available without retraining.
        """
        rng = self.store.tm_length_range
        if n_residues is None or rng is None:
            return {"tm_confidence": "high"}
        lo, hi = rng
        if lo <= n_residues <= hi:
            return {"tm_confidence": "high"}
        return {
            "tm_confidence": "low",
            "tm_confidence_reason": (
                f"sequence length {n_residues} is outside model_tm's training range "
                f"{lo}-{hi} aa; absolute Tm is extrapolated"
            ),
        }

    def _tm_from_embedding(
        self, raw_emb: torch.Tensor, n_residues: int | None = None
    ) -> dict[str, object]:
        """Absolute Tm (°C) from the Meltome-trained model_tm on a raw embedding."""
        if self.store.model_tm is not None:
            out: dict[str, object] = {
                "tm": float(self.store.model_tm.predict(raw_emb.cpu().numpy())),
            }
            out.update(self._tm_confidence(n_residues))
            if self.store.tm_residual_std is not None:
                out["tm_ci95"] = 1.96 * self.store.tm_residual_std
            return out
        # None, not NaN: NaN serializes to a bare `NaN` literal that is not valid
        # JSON and throws in the browser.
        return {"tm": None, "tm_confidence": "unavailable"}

    def predict_absolute(self, sequence: str) -> dict[str, object]:
        self.ensure_ready()
        esm2 = self._get_esm2()
        seq = esm2.validate_sequence(sequence)
        raw_emb = esm2.embed(seq)  # cpu tensor, 1280-d (needed raw for model_tm)
        emb = self.store.normalize(raw_emb.to(settings.resolved_device)).unsqueeze(0)
        with torch.no_grad():
            dg = self.store.model_thermo(emb)
        dg_v = float(dg.item())
        # Apply the val-fit linear recalibration to the absolute dG head.
        cal = self.store.thermo_calibration or {}
        if "dg" in cal:
            dg_v = cal["dg"]["slope"] * dg_v + cal["dg"]["intercept"]
        rs = self.store.thermo_residual_std or {}
        out: dict[str, object] = {"dg": dg_v}
        if "dg" in rs:
            out["dg_ci95"] = 1.96 * rs["dg"]
        # Absolute Tm from the Meltome-trained model_tm (reads the raw embedding).
        out.update(self._tm_from_embedding(raw_emb, n_residues=len(seq)))
        return out

    def _embed_wt_mut(
        self, wt_sequence: str, mutations: list[Mutation]
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        """Embed the wild-type and mutant sequences together (raw cpu tensors).

        Returns the validated wild-type sequence too, so callers can report length
        without re-validating.
        """
        esm2 = self._get_esm2()
        seq = esm2.validate_sequence(wt_sequence)
        mut_seq = esm2.apply_mutations(
            seq, [(m.position, m.source_aa, m.target_aa) for m in mutations]
        )
        # Embed WT and mutant together so the difference is on a consistent footing.
        wt_emb, mut_emb = esm2.batch_embed([seq, mut_seq])
        return wt_emb, mut_emb, seq

    def _ddg_from_embeddings(self, wt_emb: torch.Tensor, mut_emb: torch.Tensor) -> float:
        device = settings.resolved_device
        wt_norm = self.store.normalize(wt_emb.to(device)).unsqueeze(0)
        mut_norm = self.store.normalize(mut_emb.to(device)).unsqueeze(0)
        with torch.no_grad():
            if self.store.model_ddg is not None:
                # Δemb = (emb_mut - emb_wt)/std = normalize(mut) - normalize(wt).
                feat = torch.cat([mut_norm, mut_norm - wt_norm], dim=-1)
                return float(self.store.model_ddg(feat).item())
            # Fallback only if no ΔΔG model is installed.
            dg_wt = self.store.model_thermo(wt_norm)
            dg_mut = self.store.model_thermo(mut_norm)
            return float(dg_mut.item() - dg_wt.item())

    def _ddg_from_embeddings_batch(
        self, wt_emb: torch.Tensor, mut_embs: torch.Tensor
    ) -> np.ndarray:
        """Vectorized ΔΔG for many mutants scored against one wild-type.

        ``wt_emb`` is a (1280,) raw embedding, ``mut_embs`` an (N, 1280) batch. The
        feature construction matches :meth:`_ddg_from_embeddings` exactly — the shared
        WT reference is normalized once and broadcast, so a site-saturation scan need
        not re-embed the wild-type per candidate.
        """
        device = settings.resolved_device
        wt_norm = self.store.normalize(wt_emb.to(device)).unsqueeze(0)  # (1, D)
        mut_norm = self.store.normalize(mut_embs.to(device))            # (N, D)
        with torch.no_grad():
            if self.store.model_ddg is not None:
                feat = torch.cat([mut_norm, mut_norm - wt_norm], dim=-1)  # (N, 2D)
                return self.store.model_ddg(feat).cpu().numpy()
            # Fallback only if no ΔΔG model is installed.
            dg_wt = self.store.model_thermo(wt_norm)                      # (1,)
            dg_mut = self.store.model_thermo(mut_norm)                    # (N,)
            return (dg_mut - dg_wt).cpu().numpy()

    def predict_delta_batch_embeds(
        self, wt_emb: torch.Tensor, mut_embs: torch.Tensor
    ) -> np.ndarray:
        """Batched ΔΔG values from a precomputed WT embedding + mutant embeddings."""
        self.ensure_ready()
        return self._ddg_from_embeddings_batch(wt_emb, mut_embs)

    def _ddg_out(self, ddg: float) -> dict[str, object]:
        rs = self.store.ddg_residual_std or {}
        out: dict[str, object] = {"ddg": ddg}
        if "ddg" in rs:
            out["ddg_ci95"] = 1.96 * rs["ddg"]
        return out

    def predict_delta(
        self, wt_sequence: str, mutations: list[Mutation]
    ) -> dict[str, object]:
        """ΔΔG for a mutation set — used to rank candidate mutations."""
        self.ensure_ready()
        wt_emb, mut_emb, _ = self._embed_wt_mut(wt_sequence, mutations)
        return self._ddg_out(self._ddg_from_embeddings(wt_emb, mut_emb))

    def predict_mutant(
        self, wt_sequence: str, mutations: list[Mutation]
    ) -> dict[str, object]:
        """Mutant outputs: absolute Tm (model_tm on the mutant) + ΔΔG (model_ddg)."""
        self.ensure_ready()
        wt_emb, mut_emb, seq = self._embed_wt_mut(wt_sequence, mutations)
        out = self._ddg_out(self._ddg_from_embeddings(wt_emb, mut_emb))
        # A point mutation preserves length, so the mutant length equals len(seq).
        out.update(self._tm_from_embedding(mut_emb, n_residues=len(seq)))
        return out

    def predict_full(
        self, sequence: str, mutations: list[Mutation] | None = None
    ) -> dict[str, object]:
        if mutations:
            return self.predict_mutant(sequence, mutations)
        return self.predict_absolute(sequence)
