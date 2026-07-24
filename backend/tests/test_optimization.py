"""Fast, data-free tests for the thermostabilization optimizer's control flow.

These guard two properties that don't need ESM-2 or trained checkpoints:

  * ``greedy_select`` only applies *stabilizing* mutations (ΔΔG > 0) — a
    thermostabilizer must never pad the set with destabilizing mutations to
    reach ``max_mutations``.
  * ``saturation_scan`` runs the masked language-model guardrail *once per
    position* (not once per candidate residue) and filters on its probability.

Both use tiny fakes for the ESM-2 service and the ΔΔG predictor, so no model
weights are loaded.

Run directly (``python tests/test_optimization.py``) or with ``pytest``.
"""
from __future__ import annotations

import torch

from app.engines.optimization import CandidateMutation, OptimizationEngine
from app.models.regression import AA_VOCAB


def _cand(pos: int, ddg: float) -> CandidateMutation:
    return CandidateMutation(
        position=pos, source_aa="A", target_aa="V", ddg=ddg, residue_prob=0.9, score=ddg
    )


def _engine(**kwargs) -> OptimizationEngine:
    # prediction/esm2 are injected per-test; default to None (unused by greedy_select).
    return OptimizationEngine(prediction=kwargs.get("prediction"), esm2=kwargs.get("esm2"))


def test_greedy_selects_top_stabilizing_at_distinct_positions() -> None:
    eng = _engine()
    picked = eng.greedy_select([_cand(0, 3.0), _cand(1, 2.0), _cand(2, 1.0)], max_mutations=2)
    assert [m.position for m in picked] == [0, 1], "should take the two most stabilizing"


def test_greedy_skips_nonpositive_scores() -> None:
    eng = _engine()
    picked = eng.greedy_select([_cand(0, 3.0), _cand(1, -1.0), _cand(2, 2.0)], max_mutations=5)
    assert [m.position for m in picked] == [0, 2], "destabilizing candidate must be dropped"


def test_greedy_returns_empty_when_nothing_stabilizes() -> None:
    eng = _engine()
    picked = eng.greedy_select([_cand(0, -1.0), _cand(1, -0.5), _cand(2, 0.0)], max_mutations=5)
    assert picked == [], "no ΔΔG>0 candidate → leave the wild-type unchanged"


def test_greedy_keeps_only_best_per_position() -> None:
    eng = _engine()
    picked = eng.greedy_select([_cand(0, 3.0), _cand(0, 5.0), _cand(1, 1.0)], max_mutations=5)
    assert [m.position for m in picked] == [0, 1]
    assert picked[0].target_aa == "V"


class _FakeESM:
    """Stand-in ESM-2 service: records guardrail + embed calls, favours one residue.

    ``batch_embed`` encodes each mutant's mutated position (index of the plausible
    residue in the string) so the fake predictor can produce a deterministic ΔΔG
    without any real model.
    """

    def __init__(self, plausible_aa: str = "L") -> None:
        self.plausible_aa = plausible_aa
        self.prob_calls: list[int] = []
        self.embed_calls = 0

    def validate_sequence(self, s: str) -> str:
        return s

    def residue_probabilities(self, seq: str, pos: int) -> dict[str, float]:
        self.prob_calls.append(pos)
        return {aa: (0.9 if aa == self.plausible_aa else 0.001) for aa in AA_VOCAB}

    def apply_mutations(self, seq: str, muts: list) -> str:
        chars = list(seq)
        for pos, _src, tgt in muts:
            chars[pos] = tgt
        return "".join(chars)

    def embed(self, seq: str) -> torch.Tensor:
        self.embed_calls += 1
        return torch.tensor([-1.0])  # shared wild-type reference

    def batch_embed(self, seqs: list[str]) -> torch.Tensor:
        return torch.tensor([[float(s.index(self.plausible_aa))] for s in seqs])


class _FakePred:
    """Stand-in ΔΔG predictor: ΔΔG = the encoded mutated position (no model)."""

    def ensure_ready(self) -> None:
        pass

    def predict_delta_batch_embeds(self, wt_emb, mut_embs):
        return mut_embs[:, 0].tolist()


def test_saturation_scan_batches_and_embeds_wild_type_once() -> None:
    esm = _FakeESM(plausible_aa="L")
    eng = OptimizationEngine(prediction=_FakePred(), esm2=esm)
    seq = "CCCC"  # 4 residues, none equal to the plausible target "L"

    cands = eng.saturation_scan(seq, perplexity_threshold=0.05)

    # One masked forward pass per position — not one per (position, residue) pair.
    assert esm.prob_calls == [0, 1, 2, 3]
    # The wild-type is embedded exactly once for the whole scan.
    assert esm.embed_calls == 1
    # Only the single plausible residue clears the 0.05 floor at each position.
    assert len(cands) == 4
    assert {c.target_aa for c in cands} == {"L"}
    assert [c.ddg for c in cands] == [0.0, 1.0, 2.0, 3.0]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
