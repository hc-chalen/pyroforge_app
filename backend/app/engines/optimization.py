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
    # Per-mutation predicted ΔΔG and rank. The shortlist is a ranking, so the rank
    # and the individual predictions must travel with it — a bare list of mutations
    # invites reading every entry as equally established.
    mutation_details: list[dict] = field(default_factory=list)
    # Held-out measurement of what this ranking is worth (see calibrate_ddg).
    ranking_quality: dict | None = None
    scan_summary: dict = field(default_factory=dict)
    # In-process only (raw ESM-2 embeddings, never serialized into the job result):
    # lets the caller run the structure comparison without re-embedding both sequences.
    wt_embedding: object | None = None
    variant_embedding: object | None = None


class OptimizationEngine:
    """In silico thermostabilization.

    Site-saturation scan → ESM-2 residue-probability guardrail → rank by predicted
    ΔΔG → keep the top-k at distinct positions.

    **This produces a ranked shortlist for screening, not a set of established
    stabilizing mutations.** That distinction is the whole design. Measured on
    parent-held-out proteins (``app.training.calibrate_ddg``), model_ddg ranks well
    within a protein — median Spearman ≈ 0.76, and 74% of top-5 shortlists are truly
    stabilizing against a 22% base rate — but per protein that precision ranges from
    0% to 100%, and it is below 50% for 5 of the 22 proteins measured. So the
    shortlist is worth screening and is not worth trusting item by item, and the
    result carries those numbers so the caller can say so too.

    An earlier design instead required each candidate to clear
    ``ddg_effect_size_sigma`` × the model's residual σ, described as proving the
    candidate was "distinguishable from noise". That was wrong twice over: a kσ
    threshold is a per-comparison claim being applied to the L×19 candidates of a
    full scan, so it does not control the selection bias it was introduced for; and
    at the shipped σ it sat so far above the prediction range that it returned
    nothing for 20 of 22 held-out proteins. The feature was inert, and where it did
    fire it was not making the claim it advertised.

    **What the objective is, and is not.** The ranked signal is ΔΔG: the change in
    *folding free energy*, from the Tsuboyama 2023 small-designed-domain assay. The
    product's headline number is Tm, from the Meltome whole-protein assay. These are
    different measurements and they are only weakly related — on the 219 rows of
    esm2_test.pt carrying both labels, corr(dG, Tm) is +0.24 row-level and **+0.17
    per-protein across 21 proteins**. So a positive total ΔΔG must be read as
    "predicted more stable to unfolding", not as a predicted Tm gain. Only the
    wild-type baseline Tm is reported; no variant Tm is claimed.

    **Level vs rank.** model_ddg's absolute output is biased low: on held-out data
    it predicts ΔΔG > 0 for ~6% of mutations where ~24% are truly stabilizing. A
    linear recalibration was fitted to correct that and deliberately *not* shipped —
    the measured slope is within noise of 1.0 and it did not improve error on an
    independent file (see ``calibration.applied_reason``). So the reported kcal/mol
    is a rank-ordering signal whose absolute level should be read conservatively,
    which is what the ±95% interval and the copy in the UI say.
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

    def ranking_quality(self) -> dict | None:
        """Measured within-protein ranking quality, or ``None`` if never measured.

        Produced by ``app.training.calibrate_ddg`` on parent-held-out proteins. This
        is what the optimizer's shortlist is actually worth, and it is reported to
        the user instead of a statistical claim about individual candidates.
        """
        return self.prediction.store.ddg_ranking

    def greedy_select(
        self,
        candidates: list[CandidateMutation],
        max_mutations: int,
        min_score: float = 0.0,
    ) -> list[Mutation]:
        """Take the top ``max_mutations`` ranked mutations at distinct positions.

        ``min_score`` is a *sign floor*, not a significance test: a candidate the
        model predicts to be destabilizing must never be offered as a stabilizing
        one, so the default cut is at 0. It is deliberately not set to a multiple of
        the model's residual σ — that was the previous design, and it was wrong in
        two ways. A kσ threshold is a per-comparison claim applied to the L×19
        candidates of a whole scan, so it does not control the selection bias it was
        meant to; and empirically it was so far above the prediction range that it
        returned nothing at all for 20 of 22 held-out proteins, making the feature
        inert rather than conservative.

        What the held-out measurement does support is *ranking*: median
        within-protein Spearman ≈ 0.76, and 74% of top-5 shortlists are truly
        stabilizing against a 22% base rate. So the tool ranks and hands back a
        shortlist, and reports that measured quality rather than implying each
        candidate is individually established.

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
        # Sign floor only — see greedy_select for why the old kσ gate was removed.
        selected = self.greedy_select(candidates, max_mut, min_score=0.0)
        n_positive = sum(1 for c in candidates if c.score > 0.0)
        by_position = {(c.position, c.target_aa): c for c in candidates}
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
            mutation_details=[
                {
                    "position": m.position,
                    "source_aa": m.source_aa,
                    "target_aa": m.target_aa,
                    "ddg": by_position[(m.position, m.target_aa)].ddg,
                    "residue_prob": by_position[(m.position, m.target_aa)].residue_prob,
                    "rank": i + 1,
                }
                for i, m in enumerate(selected)
            ],
            ranking_quality=self.ranking_quality(),
            scan_summary={
                "total_candidates": len(candidates),
                "selected_mutations": len(selected),
                "perplexity_threshold": perplexity_threshold or settings.perplexity_threshold,
                "candidates_ddg_positive": n_positive,
            },
            wt_embedding=wt_emb,
            variant_embedding=variant_emb,
        )
