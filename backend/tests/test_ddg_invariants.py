"""Structural invariants of the served ΔΔG model and the mutation-safety guards.

These are properties that must hold for *any* checkpoint, independent of accuracy:

  * ΔΔG(wt→wt) should be 0 — no mutation, no stability change.
  * ΔΔG(A→B) should equal −ΔΔG(B→A) (``train_ddg.antisymmetric_augment`` exists to
    enforce exactly this).
  * The batched scan path must agree with the single-pair path, because the
    optimizer ranks with the former and reports with the latter.
  * ``apply_mutations`` must reject a 1-indexed position rather than silently
    mutating the wrong residue.

Most checks need neither ESM-2 nor a dataset file — embeddings are synthesized
from the checkpoint's own ``norm_stats``, so only ``model_ddg.pt`` (3 MB, in the
repo) is required and the suite runs in seconds.

**Antisymmetry is the exception and is measured on real embeddings.** Synthetic
embeddings drawn independently from N(μ, σ) ignore the correlation structure of a
mean-pooled ESM-2 vector, and the reverse-mutation property is only trained on
that manifold: measured on real pairs the bias is −0.15, on isotropic synthetic
pairs it is +0.40 — a different question. That test therefore reads
``esm2_test.pt`` and skips when it is absent (the file is gitignored, so it skips
in CI and runs locally).

The numeric bounds are **regression guards on measured values, not targets**. The
shipped checkpoint has a real ΔΔG(wt→wt) offset of about −0.24 kcal/mol; it is not
corrected because the optimizer consumes rank, not level (see
``checkpoints/model_ddg_calibration.json``). The assertions pin these values so
they cannot silently drift — a retrain that changes them must update the numbers
deliberately.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from app.core.config import settings
from app.services.esm2 import ESM2Service
from app.services.model_store import ModelStore

# Measured on the shipped checkpoint (2026-07): |ΔΔG(wt→wt)| = 0.238 kcal/mol, and
# on real esm2_test.pt pairs antisymmetry bias = −0.152 with corr(fwd, −rev) = 0.950.
# The bounds leave headroom for a retrain but fail loudly if the antisymmetric
# augmentation were dropped — bias was −1.29 and corr 0.27 before it was added
# (see train_ddg's module docstring).
MAX_SELF_DDG = 0.40
MAX_ANTISYMMETRY_BIAS = 0.40
MIN_ANTISYMMETRY_CORR = 0.90

N_SAMPLES = 256


@pytest.fixture(scope="module")
def store() -> ModelStore:
    s = ModelStore()
    try:
        s.load()
    except FileNotFoundError as exc:  # pragma: no cover - depends on the checkout
        pytest.skip(f"checkpoints not available: {exc}")
    if s.model_ddg is None:
        pytest.skip("model_ddg.pt not installed; ΔΔG invariants do not apply")
    return s


@pytest.fixture(scope="module")
def embeddings(store: ModelStore) -> tuple[torch.Tensor, torch.Tensor]:
    """Realistic (wt, mutant) embedding pairs: N(μ, σ) from the checkpoint's stats.

    The mutant is the wild type plus a small perturbation, mirroring what a single
    residue substitution does to a mean-pooled embedding — a large random vector
    would be a different question than the one these invariants ask.
    """
    stats = store.ddg_norm_stats or store.norm_stats
    mean = np.asarray(stats["embedding_mean"], dtype=np.float32)
    std = np.asarray(stats["embedding_std"], dtype=np.float32)

    rng = np.random.default_rng(0)
    wt = mean + std * rng.standard_normal((N_SAMPLES, len(mean))).astype(np.float32)
    # A point mutation moves the mean-pooled embedding by a small fraction of σ.
    delta = 0.05 * std * rng.standard_normal((N_SAMPLES, len(mean))).astype(np.float32)
    return torch.from_numpy(wt), torch.from_numpy(wt + delta)


def _score(store: ModelStore, wt: torch.Tensor, mut: torch.Tensor) -> np.ndarray:
    """ΔΔG for a batch of (wt, mutant) pairs — the served feature construction."""
    device = settings.resolved_device
    with torch.no_grad():
        wt_n = store.normalize_ddg(wt.to(device))
        mut_n = store.normalize_ddg(mut.to(device))
        feat = torch.cat([mut_n, mut_n - wt_n], dim=-1)
        return store.model_ddg(feat).cpu().numpy()


def test_null_mutation_is_near_zero(store: ModelStore, embeddings) -> None:
    """ΔΔG(wt→wt) must be ~0: an unchanged sequence has no stability change.

    A nonzero value is a constant offset on every reported ΔΔG. It is currently
    about −0.24 kcal/mol and deliberately uncorrected; this pins it.
    """
    wt, _ = embeddings
    self_ddg = _score(store, wt, wt)
    offset = float(np.mean(self_ddg))
    assert abs(offset) < MAX_SELF_DDG, (
        f"ΔΔG(wt→wt) = {offset:+.3f} kcal/mol exceeds the recorded "
        f"±{MAX_SELF_DDG} bound. Every served ΔΔG carries this offset — either "
        f"correct it in ModelStore.calibrate_ddg or update this bound on purpose."
    )


@pytest.fixture(scope="module")
def real_pairs() -> tuple[torch.Tensor, torch.Tensor]:
    """Real (wild-type, mutant) ESM-2 embedding pairs from the held-out file.

    Antisymmetry only holds on the manifold the model was trained on, so it cannot
    be checked with synthesized vectors. ``esm2_test.pt`` is gitignored, so this
    skips in CI and runs wherever the evaluation data is present.
    """
    path = settings.test_data_path
    if not path.exists():
        pytest.skip(f"{path.name} not present; antisymmetry needs real embeddings")

    data = torch.load(path, map_location="cpu", weights_only=False)
    from app.data.dataset import wt_reference_index

    ref = wt_reference_index(data)
    ddg = data["ddg"].numpy()
    nsub = data["num_substitutions"].numpy()
    rows = np.where((nsub > 0) & ~np.isnan(ddg) & (ref >= 0))[0][:2000]
    if len(rows) < 100:
        pytest.skip("too few anchored mutant rows to measure antisymmetry")

    emb = data["embeddings"]
    return emb[ref[rows]], emb[rows]


def test_reverse_mutation_flips_the_sign(store: ModelStore, real_pairs) -> None:
    """ΔΔG(A→B) = −ΔΔG(B→A) — the property antisymmetric_augment exists to enforce.

    Guards against a retrain that drops the reverse-mutation augmentation, which
    left the model badly antisymmetric (bias −1.29, corr 0.27) before it was added.
    """
    wt, mut = real_pairs
    forward = _score(store, wt, mut)
    reverse = _score(store, mut, wt)

    bias = float(np.mean(forward + reverse))
    corr = float(np.corrcoef(forward, -reverse)[0, 1])

    assert abs(bias) < MAX_ANTISYMMETRY_BIAS, (
        f"antisymmetry bias <ΔΔG_fwd + ΔΔG_rev> = {bias:+.3f}; the forward and "
        f"reverse of the same mutation should cancel. Was antisymmetric_augment "
        f"dropped from train_ddg?"
    )
    assert corr > MIN_ANTISYMMETRY_CORR, (
        f"corr(forward, −reverse) = {corr:.3f} < {MIN_ANTISYMMETRY_CORR}"
    )


def test_batched_scan_matches_the_single_pair_path(store: ModelStore, embeddings) -> None:
    """The saturation scan's batched ΔΔG must equal the per-pair ΔΔG.

    ``_ddg_from_embeddings_batch`` broadcasts one normalized wild type across N
    mutants while ``_ddg_from_embeddings`` normalizes a single pair. The optimizer
    *ranks* with the batched path and *reports* the selected set with the single
    path, so a divergence between them would mean the shortlist and the headline
    number disagree.
    """
    from app.engines.prediction import PredictionEngine

    wt, mut = embeddings
    engine = PredictionEngine()
    engine.store = store

    batched = engine.predict_delta_batch_embeds(wt[0], mut[:16])
    singles = np.array(
        [engine.predict_delta_embeds(wt[0], mut[i])["ddg"] for i in range(16)]
    )
    np.testing.assert_allclose(batched, singles, rtol=1e-4, atol=1e-4)


def test_calibration_is_identity_when_not_applied(store: ModelStore) -> None:
    """``calibrate_ddg`` must not transform anything unless a fit was adopted.

    The shipped calibration records ``applied: false`` (the fitted slope sits
    within noise of 1.0), so the served value has to pass through untouched.
    """
    if store.ddg_calibration:
        pytest.skip("a calibration is applied; identity does not hold by design")
    for value in (-2.5, 0.0, 1.75):
        assert store.calibrate_ddg(value) == value


class TestMutationSafety:
    """``apply_mutations`` is the last guard against scoring the wrong residue."""

    SEQ = "MKVLAAGIVG"  # 10 residues, the minimum accepted length

    def test_applies_a_valid_substitution(self) -> None:
        out = ESM2Service.apply_mutations(self.SEQ, [(0, "M", "A")])
        assert out == "AKVLAAGIVG"
        assert len(out) == len(self.SEQ)

    def test_rejects_a_one_indexed_position(self) -> None:
        """The UI shows position+1; a client echoing that back must not be honoured.

        Position 1 holds 'K', so a client meaning "residue 1 = M" is caught by the
        source_aa check rather than silently mutating the neighbour.
        """
        with pytest.raises(ValueError, match="source mismatch"):
            ESM2Service.apply_mutations(self.SEQ, [(1, "M", "A")])

    def test_rejects_out_of_range_position(self) -> None:
        with pytest.raises(ValueError, match="out of range"):
            ESM2Service.apply_mutations(self.SEQ, [(len(self.SEQ), "G", "A")])

    def test_rejects_non_standard_target_residue(self) -> None:
        """A non-standard residue becomes an ESM-2 <unk> token and a junk embedding."""
        with pytest.raises(ValueError, match="Invalid target amino acid"):
            ESM2Service.apply_mutations(self.SEQ, [(0, "M", "X")])

    def test_rejects_sequence_with_fasta_artifacts(self) -> None:
        """Newlines/headers must be rejected (or cleaned) rather than embedded."""
        with pytest.raises(ValueError):
            ESM2Service.validate_sequence(">sp|P1|X\nMKVLAAGIVG")
