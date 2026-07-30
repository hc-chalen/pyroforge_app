from __future__ import annotations

from dataclasses import dataclass, field

from app.core.config import settings
from app.engines.prediction import Mutation, PredictionEngine
from app.models.regression import AA_VOCAB
from app.services.esm2 import ESM2Service, get_esm2_service


@dataclass
class CandidateMutation:
    position: int
    source_aa: str
    target_aa: str
    ddg: float
    residue_prob: float  # masked-marginal P(target_aa) from ESM-2 (0–1, higher = more plausible)
    score: float


@dataclass
class OptimizationResult:
    wild_type_sequence: str
    optimized_sequence: str
    mutations: list[Mutation]
    # Absolute Tm of the input (wild-type) protein; None when no Tm model is installed.
    baseline_tm: float | None
    ddg: float          # total ΔΔG of the selected set — positive = stabilizing
    ddg_ci95: float | None = None  # ±95% interval on ΔΔG from model_ddg's residual
    scan_summary: dict = field(default_factory=dict)
    # In-process only (raw ESM-2 embeddings, never serialized into the job result):
    # lets the caller run the structure comparison without re-embedding both sequences.
    wt_embedding: object | None = None
    variant_embedding: object | None = None


class OptimizationEngine:
    """In silico thermostabilization.

    Site-saturation scan → ESM-2 residue-probability guardrail → significance gate →
    keep the top-k most *stabilizing* mutations at distinct positions.

    **What the objective is, and is not.** The ranked signal is ΔΔG: the change in
    *folding free energy*, from the Tsuboyama 2023 small-designed-domain assay. The
    product's headline number is Tm, from the Meltome whole-protein assay. These are
    different measurements and they are only weakly related — on the 219 rows of
    esm2_test.pt carrying both labels, corr(dG, Tm) is +0.24 row-level and **+0.17
    per-protein across 21 proteins**. So a positive total ΔΔG must be read as
    "predicted more stable to unfolding", not as a predicted Tm gain. Only the
    wild-type baseline Tm is reported; no variant Tm is claimed.

    **Significance gate.** Candidates must clear ``settings.ddg_effect_size_sigma``
    standard deviations of model_ddg's own held-out residual, so the tool cannot
    report a stabilizing variant whose entire predicted gain sits inside the model's
    noise band. Returning zero mutations is a valid, honest outcome.
    """

    def __init__(
        self,
        prediction: PredictionEngine | None = None,
        esm2: ESM2Service | None = None,
    ) -> None:
        self.prediction = prediction or PredictionEngine()
        self.esm2 = esm2

    def _get_esm2(self) -> ESM2Service:
        if self.esm2 is None:
            self.esm2 = get_esm2_service()
        return self.esm2

    def _composite_score(self, ddg: float) -> float:
        # Objective: maximize thermostability. In the training data ΔΔG = dG(mut) −
        # dG(wt) with higher dG = more stable (verified: corr(dg, tm) > 0), so a MORE
        # POSITIVE ΔΔG is more stabilizing. Rank by +ΔΔG so the greedy step keeps the
        # most stabilizing mutations. (ΔTm was dropped — it was a strictly monotone
        # function of ΔΔG, so it added nothing to the ranking.)
        return ddg

    def saturation_scan(
        self,
        sequence: str,
        perplexity_threshold: float | None = None,
        progress_callback=None,
        wt_emb=None,
    ) -> list[CandidateMutation]:
        esm2 = self._get_esm2()
        seq = esm2.validate_sequence(sequence)
        threshold = perplexity_threshold or settings.perplexity_threshold
        self.prediction.ensure_ready()
        # Embed the wild-type once; every candidate's ΔΔG is scored against this same
        # reference embedding, so the WT is never re-embedded per mutation. Callers
        # that already hold the WT embedding pass it in to skip even this pass.
        if wt_emb is None:
            wt_emb = esm2.embed(seq)
        candidates: list[CandidateMutation] = []
        total = len(seq) * (len(AA_VOCAB) - 1)
        done = 0

        for pos in range(len(seq)):
            wt_aa = seq[pos]
            # One masked forward pass per position yields every substitution's
            # probability (the pass does not depend on the target residue).
            pos_probs = esm2.residue_probabilities(seq, pos)
            # Guardrail: keep only mutations the language model deems plausible
            # (masked-marginal probability of the target residue above the floor).
            surviving_aa = [
                aa for aa in AA_VOCAB if aa != wt_aa and pos_probs[aa] >= threshold
            ]
            done += (len(AA_VOCAB) - 1) - len(surviving_aa)  # count filtered-out pairs
            if progress_callback and done:
                progress_callback(done, total)
            if not surviving_aa:
                continue
            # Batch-embed this position's plausible mutants and score them together
            # against the shared WT embedding (one batched ΔΔG forward pass).
            mut_seqs = [esm2.apply_mutations(seq, [(pos, wt_aa, aa)]) for aa in surviving_aa]
            mut_embs = esm2.batch_embed(mut_seqs)
            ddgs = self.prediction.predict_delta_batch_embeds(wt_emb, mut_embs)
            for aa, ddg in zip(surviving_aa, ddgs):
                ddg = float(ddg)
                candidates.append(
                    CandidateMutation(
                        position=pos,
                        source_aa=wt_aa,
                        target_aa=aa,
                        ddg=ddg,
                        residue_prob=pos_probs[aa],
                        score=self._composite_score(ddg),
                    )
                )
                done += 1
                if progress_callback:
                    progress_callback(done, total)
        return candidates

    def ddg_ci95(self) -> float | None:
        """±95% interval on a single ΔΔG prediction, from model_ddg's residual σ."""
        rs = (self.prediction.store.ddg_residual_std or {}).get("ddg")
        return None if rs is None else 1.96 * float(rs)

    def min_effect_threshold(self) -> float:
        """Smallest ΔΔG that is distinguishable from model_ddg's own noise.

        ``ddg_effect_size_sigma`` × the model's held-out residual σ. A candidate below
        this is not evidence of stabilization, however positive it looks: for the
        shipped checkpoint σ = 0.776 kcal/mol, so the default 1.96σ gate is
        1.52 kcal/mol — and the model's *positional* spread is only ≈0.31 kcal/mol,
        so most raw ΔΔG > 0 candidates are noise. Set the sigma to 0 to restore the
        old keep-anything-positive behaviour.
        """
        sigma = settings.ddg_effect_size_sigma
        if sigma <= 0:
            return 0.0
        rs = (self.prediction.store.ddg_residual_std or {}).get("ddg")
        if rs is None:
            return settings.ddg_min_effect_fallback
        return sigma * float(rs)

    def greedy_select(
        self,
        candidates: list[CandidateMutation],
        max_mutations: int,
        min_score: float = 0.0,
    ) -> list[Mutation]:
        """Take up to ``max_mutations`` *stabilizing* mutations at distinct positions.

        Only mutations whose score exceeds ``min_score`` are selected. Callers should
        pass :meth:`min_effect_threshold` so the cut is at the model's noise floor
        rather than at 0 — a bare ΔΔG > 0 test admits candidates whose predicted gain
        is smaller than the model's own uncertainty. The objective is
        thermostabilization, so the tool must never pad the set with mutations it
        cannot distinguish from noise just to reach ``max_mutations``. When fewer
        than ``max_mutations`` candidates clear the bar, fewer are returned (possibly
        none, leaving the wild-type unchanged — a valid outcome, not a failure).

        Selection assumes single-mutation ΔΔG effects are additive (no re-scoring of
        the growing set), so it can miss epistasis between chosen positions. The final
        combined ΔΔG is recomputed once on the full variant in :meth:`optimize`, so the
        reported number is exact even though selection is additive.
        """
        sorted_cands = sorted(candidates, key=lambda c: c.score, reverse=True)
        selected: list[Mutation] = []
        used_positions: set[int] = set()

        for cand in sorted_cands:
            # Sorted descending, so once a score fails the bar every later one does
            # too — stop rather than pad the set with indistinguishable mutations.
            if cand.score <= min_score:
                break
            if cand.position in used_positions:
                continue
            selected.append(
                Mutation(
                    position=cand.position,
                    source_aa=cand.source_aa,
                    target_aa=cand.target_aa,
                )
            )
            used_positions.add(cand.position)
            if len(selected) >= max_mutations:
                break
        return selected

    def optimize(
        self,
        sequence: str,
        max_mutations: int | None = None,
        perplexity_threshold: float | None = None,
        progress_callback=None,
    ) -> OptimizationResult:
        max_mut = max_mutations or settings.max_mutations_default
        esm2 = self._get_esm2()
        seq = esm2.validate_sequence(sequence)

        # One ESM-2 pass for the wild type, shared by the scan, the baseline Tm/dG and
        # the final combined ΔΔG. Previously the WT was embedded 4× per optimization
        # (scan → predict_absolute → predict_delta → compare_embeddings) and the
        # variant twice; on a 400-residue protein each pass is a full 650M forward.
        wt_emb = esm2.embed(seq)

        candidates = self.saturation_scan(
            seq, perplexity_threshold, progress_callback, wt_emb=wt_emb
        )
        # Cut at model_ddg's noise floor, not at 0 — see min_effect_threshold.
        min_effect = self.min_effect_threshold()
        selected = self.greedy_select(candidates, max_mut, min_score=min_effect)
        n_positive = sum(1 for c in candidates if c.score > 0.0)
        n_significant = sum(1 for c in candidates if c.score > min_effect)
        optimized = esm2.apply_mutations(
            seq, [(m.position, m.source_aa, m.target_aa) for m in selected]
        )

        # Baseline thermostability of the input protein — model_tm is reliable at the
        # whole-protein level, so this is the meaningful Tm to report.
        baseline = self.prediction.predict_absolute_from_embedding(
            wt_emb, n_residues=len(seq)
        )
        baseline_tm = baseline["tm"]
        baseline_tm = None if baseline_tm is None else float(baseline_tm)
        # ΔΔG (positive = stabilizing) is the optimization outcome, recomputed once on
        # the full selected set. We deliberately do NOT report a separate "variant Tm":
        # model_tm barely moves for a handful of point mutations, so it would be
        # ≈ baseline and easily misread as a thermal shift — the effect lives in ΔΔG.
        if selected:
            variant_emb = esm2.embed(optimized)
            ddg_value = float(
                self.prediction.predict_delta_embeds(wt_emb, variant_emb)["ddg"]
            )
        else:
            variant_emb = wt_emb  # no mutations survived the gate → variant == WT
            ddg_value = 0.0

        return OptimizationResult(
            wild_type_sequence=seq,
            optimized_sequence=optimized,
            mutations=selected,
            baseline_tm=baseline_tm,
            ddg=ddg_value,
            ddg_ci95=self.ddg_ci95(),
            scan_summary={
                "total_candidates": len(candidates),
                "selected_mutations": len(selected),
                "perplexity_threshold": perplexity_threshold or settings.perplexity_threshold,
                # Significance accounting, so the gap between "looks positive" and
                # "is distinguishable from noise" is visible in the result.
                "min_ddg_effect": min_effect,
                "effect_size_sigma": settings.ddg_effect_size_sigma,
                "candidates_ddg_positive": n_positive,
                "candidates_above_noise": n_significant,
                "candidates_positive_but_within_noise": n_positive - n_significant,
            },
            wt_embedding=wt_emb,
            variant_embedding=variant_emb,
        )
