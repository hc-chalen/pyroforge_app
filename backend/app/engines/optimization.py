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
    scan_summary: dict = field(default_factory=dict)


class OptimizationEngine:
    """In silico thermostabilization.

    Site-saturation scan → ESM-2 residue-probability guardrail → keep the top-k
    most *stabilizing* mutations at distinct positions. The optimization objective
    is thermostability; the ranked signal is ΔΔG (folding-stability change), which
    is mutation-sensitive and correlates with Tm. Only the wild-type baseline Tm is
    reported (``model_tm`` barely moves for a single point mutation), so the
    improvement is expressed as ΔΔG, not as a model_tm difference.
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
    ) -> list[CandidateMutation]:
        esm2 = self._get_esm2()
        seq = esm2.validate_sequence(sequence)
        threshold = perplexity_threshold or settings.perplexity_threshold
        self.prediction.ensure_ready()
        # Embed the wild-type once; every candidate's ΔΔG is scored against this same
        # reference embedding, so the WT is never re-embedded per mutation.
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

    def greedy_select(
        self,
        candidates: list[CandidateMutation],
        max_mutations: int,
        min_score: float = 0.0,
    ) -> list[Mutation]:
        """Take up to ``max_mutations`` *stabilizing* mutations at distinct positions.

        Only mutations whose score exceeds ``min_score`` (default 0 → predicted
        stabilizing) are selected. The objective is thermostabilization, so adding a
        mutation with a non-positive ΔΔG would, under the additivity assumption below,
        *lower* the variant's total ΔΔG — the tool must never pad the set with
        destabilizing mutations just to reach ``max_mutations``. When fewer than
        ``max_mutations`` candidates are stabilizing, fewer are returned (possibly
        none, leaving the wild-type unchanged).

        Selection assumes single-mutation ΔΔG effects are additive (no re-scoring of
        the growing set), so it can miss epistasis between chosen positions. The final
        combined ΔΔG is recomputed once on the full variant in :meth:`optimize`, so the
        reported number is exact even though selection is additive.
        """
        sorted_cands = sorted(candidates, key=lambda c: c.score, reverse=True)
        selected: list[Mutation] = []
        used_positions: set[int] = set()

        for cand in sorted_cands:
            # Sorted descending, so once a score is non-stabilizing every later one is
            # too — stop rather than pad the set with destabilizing mutations.
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

        candidates = self.saturation_scan(seq, perplexity_threshold, progress_callback)
        selected = self.greedy_select(candidates, max_mut)
        optimized = esm2.apply_mutations(
            seq, [(m.position, m.source_aa, m.target_aa) for m in selected]
        )

        # Baseline thermostability of the input protein — model_tm is reliable at the
        # whole-protein level, so this is the meaningful Tm to report.
        baseline = self.prediction.predict_absolute(seq)
        baseline_tm = baseline["tm"]
        baseline_tm = None if baseline_tm is None else float(baseline_tm)
        # ΔΔG (positive = stabilizing) is the optimization outcome, recomputed once on
        # the full selected set. We deliberately do NOT report a separate "variant Tm":
        # model_tm barely moves for a handful of point mutations, so it would be
        # ≈ baseline and easily misread as a thermal shift — the effect lives in ΔΔG.
        ddg_value = (
            float(self.prediction.predict_delta(seq, selected)["ddg"]) if selected else 0.0
        )

        return OptimizationResult(
            wild_type_sequence=seq,
            optimized_sequence=optimized,
            mutations=selected,
            baseline_tm=baseline_tm,
            ddg=ddg_value,
            scan_summary={
                "total_candidates": len(candidates),
                "selected_mutations": len(selected),
                "perplexity_threshold": perplexity_threshold or settings.perplexity_threshold,
            },
        )
