from __future__ import annotations

import logging
from threading import Lock

import torch

from app.core.config import settings
from app.models.regression import (
    AA_VOCAB,
    MAX_SEQUENCE_LENGTH,
    MIN_SEQUENCE_LENGTH,
)

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
logger = logging.getLogger(__name__)


class ESM2Service:
    _instance: "ESM2Service | None" = None
    _lock = Lock()

    def __init__(self) -> None:
        import esm

        # ESM-2 forward pass must not run on Apple MPS — it yields degenerate
        # embeddings that collapse every downstream prediction (see esm_device).
        self.device = settings.esm_device
        self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.model = self.model.to(self.device)
        self.model.eval()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.layer = 33
        logger.info("ESM-2 model loaded on %s", self.device)

    @classmethod
    def get_instance(cls) -> "ESM2Service":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def is_ready(cls) -> bool:
        return cls._instance is not None

    def _tokenize(self, sequences: list[str]) -> tuple:
        data = [(f"seq_{i}", seq) for i, seq in enumerate(sequences)]
        _, _, tokens = self.batch_converter(data)
        return tokens.to(self.device)

    @torch.no_grad()
    def embed(self, sequence: str) -> torch.Tensor:
        return self.batch_embed([sequence])[0]

    @torch.no_grad()
    def batch_embed(self, sequences: list[str], batch_size: int = 8) -> torch.Tensor:
        results: list[torch.Tensor] = []
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i : i + batch_size]
            tokens = self._tokenize(batch)
            out = self.model(tokens, repr_layers=[self.layer], return_contacts=False)
            reps = out["representations"][self.layer]
            for j, seq in enumerate(batch):
                # exclude BOS/EOS tokens
                seq_emb = reps[j, 1 : len(seq) + 1].mean(dim=0)
                results.append(seq_emb.cpu())
        return torch.stack(results)

    @torch.no_grad()
    def compare_embeddings(
        self,
        sequence_a: str,
        sequence_b: str,
        emb_a: torch.Tensor | None = None,
        emb_b: torch.Tensor | None = None,
    ) -> dict:
        """Compare ESM-2 mean-pooled embeddings for structural/sequence similarity.

        ``emb_a``/``emb_b`` let a caller that has already embedded these exact
        sequences skip the forward passes (the optimizer has both in hand).
        """
        seq_a = self.validate_sequence(sequence_a)
        seq_b = self.validate_sequence(sequence_b)
        if emb_a is None or emb_b is None:
            emb_a, emb_b = self.batch_embed([seq_a, seq_b])

        abs_diff = torch.abs(emb_a - emb_b)
        mean_abs_diff = float(abs_diff.mean().item())
        l1_distance = float(abs_diff.sum().item())
        max_abs_diff = float(abs_diff.max().item())

        mean_abs_a = float(torch.abs(emb_a).mean().item())
        mean_abs_b = float(torch.abs(emb_b).mean().item())
        norm_a = float(emb_a.norm().item())
        norm_b = float(emb_b.norm().item())

        cosine_similarity = float(
            torch.nn.functional.cosine_similarity(
                emb_a.unsqueeze(0), emb_b.unsqueeze(0)
            ).item()
        )
        # Map cosine from [-1, 1] to [0, 1] for display as similarity %
        similarity = (cosine_similarity + 1.0) / 2.0

        if len(seq_a) == len(seq_b):
            matches = sum(a == b for a, b in zip(seq_a, seq_b))
            sequence_identity = matches / len(seq_a)
            mutation_count = len(seq_a) - matches
        else:
            sequence_identity = None
            mutation_count = None

        return {
            "similarity": round(similarity, 4),
            "cosine_similarity": round(cosine_similarity, 4),
            "mean_absolute_difference": round(mean_abs_diff, 6),
            "l1_distance": round(l1_distance, 4),
            "max_absolute_difference": round(max_abs_diff, 6),
            "embedding_mean_abs_original": round(mean_abs_a, 6),
            "embedding_mean_abs_optimized": round(mean_abs_b, 6),
            "embedding_norm_original": round(norm_a, 4),
            "embedding_norm_optimized": round(norm_b, 4),
            "embedding_dim": emb_a.shape[0],
            "sequence_a_length": len(seq_a),
            "sequence_b_length": len(seq_b),
            "sequence_identity": round(sequence_identity, 4) if sequence_identity is not None else None,
            "mutation_count": mutation_count,
        }

    @torch.no_grad()
    def residue_probabilities(self, wt_sequence: str, position: int) -> dict[str, float]:
        """Masked-marginal P(aa) for *all 20* amino acids at 0-indexed ``position``.

        The masked forward pass depends only on the position, not on the candidate
        residue, so one pass yields every substitution's probability. Callers doing a
        site-saturation scan must use this (once per position) rather than calling
        :meth:`residue_probability` per target residue, which would recompute the same
        650M forward pass ~19× per position.

        Values are probabilities in [0, 1] (higher = more plausible substitution).
        """
        tokens = self._tokenize([wt_sequence])
        tokens[0, position + 1] = self.alphabet.mask_idx  # position + 1 for BOS offset
        out = self.model(tokens)
        logits = out["logits"][0, position + 1]
        probs = torch.softmax(logits, dim=-1)
        return {aa: float(probs[self.alphabet.get_idx(aa)].item()) for aa in AA_VOCAB}

    def residue_probability(self, wt_sequence: str, position: int, mut_aa: str) -> float:
        """Masked-marginal probability of a single ``mut_aa`` at ``position``.

        Thin wrapper over :meth:`residue_probabilities`; prefer the batched method
        when scanning every residue at a position.
        """
        return self.residue_probabilities(wt_sequence, position)[mut_aa]

    @staticmethod
    def validate_sequence(sequence: str) -> str:
        seq = sequence.strip().upper()
        invalid = set(seq) - STANDARD_AA
        if invalid:
            raise ValueError(f"Invalid amino acids: {sorted(invalid)}")
        if len(seq) < MIN_SEQUENCE_LENGTH:
            raise ValueError(
                f"Sequence must be at least {MIN_SEQUENCE_LENGTH} residues"
            )
        if len(seq) > MAX_SEQUENCE_LENGTH:
            raise ValueError(
                f"Sequence length {len(seq)} exceeds the {MAX_SEQUENCE_LENGTH}-residue limit"
            )
        return seq

    @staticmethod
    def apply_mutations(sequence: str, mutations: list[tuple[int, str, str]]) -> str:
        """Apply 0-indexed ``(position, source_aa, target_aa)`` substitutions.

        ``source_aa`` is *checked*, not ignored: it is the only guard against a client
        sending 1-indexed positions (the UI displays ``position + 1``), which would
        otherwise silently mutate the wrong residue and return a confident, wrong
        prediction. ``target_aa`` must be one of the 20 standard residues — anything
        else becomes an ESM-2 ``<unk>`` token and yields a meaningless embedding.
        """
        chars = list(sequence)
        for pos, src, tgt in mutations:
            if pos < 0 or pos >= len(chars):
                raise ValueError(
                    f"Mutation position {pos} out of range for a "
                    f"{len(chars)}-residue sequence (positions are 0-indexed)"
                )
            if tgt not in STANDARD_AA:
                raise ValueError(
                    f"Invalid target amino acid {tgt!r} at position {pos}; "
                    f"expected one of {AA_VOCAB}"
                )
            if src and src != chars[pos]:
                raise ValueError(
                    f"Mutation source mismatch at position {pos}: expected "
                    f"{src!r} but the sequence has {chars[pos]!r}. Positions are "
                    f"0-indexed — a 1-indexed position is the usual cause."
                )
            chars[pos] = tgt
        return "".join(chars)


def get_esm2_service() -> ESM2Service:
    return ESM2Service.get_instance()
