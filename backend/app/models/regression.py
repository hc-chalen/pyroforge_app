import torch
import torch.nn as nn

AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_VOCAB)}
# Width of the mutation columns in the exported datasets (mutation_positions et al.
# are (N, 10)); it is NOT a cap on how many mutations the optimizer may select —
# OptimizeRequest.max_mutations governs that.
MAX_MUTATIONS = 10
# Longest sequence accepted for embedding. Previously written as MAX_MUTATIONS * 100,
# which coupled a request limit to an unrelated dataset column width.
MAX_SEQUENCE_LENGTH = 1000
MIN_SEQUENCE_LENGTH = 10


DEFAULT_DG_WIDTHS = [1024, 512, 256, 128]  # dG has ~500k labels → large tower


def _mlp_trunk(input_dim: int, widths: list[int], dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = [nn.BatchNorm1d(input_dim)]
    prev = input_dim
    for w in widths:
        layers.extend([nn.Linear(prev, w), nn.GELU(), nn.Dropout(dropout)])
        prev = w
    return nn.Sequential(*layers)


class _Tower(nn.Module):
    def __init__(self, input_dim: int, widths: list[int], dropout: float) -> None:
        super().__init__()
        self.trunk = _mlp_trunk(input_dim, widths, dropout)
        self.head = nn.Linear(widths[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x)).squeeze(-1)


class ModelThermoDg(nn.Module):
    """model_thermo: absolute dG predictor — an averaged ensemble of towers.

    Originally a dual Tm+dG model, but the Tm head was retired: absolute Tm is
    served by the Meltome-trained ``model_tm`` (``TmRegressor``), which supersedes
    a mean-pooled-embedding Tm head. What remains is the dG side — a small
    **ensemble** of independent towers, averaged, which lowers run-to-run variance
    and improves both correlation and within-protein ranking. ``ddG`` is served by
    ``model_ddg``.

    Tower widths and the ensemble size are stored in the checkpoint.
    """

    def __init__(
        self,
        input_dim: int = 1280,
        dg_widths: list[int] | None = None,
        dropout: float = 0.2,
        n_dg: int = 3,
    ) -> None:
        super().__init__()
        dg_widths = dg_widths or DEFAULT_DG_WIDTHS
        self.dg_towers = nn.ModuleList(_Tower(input_dim, dg_widths, dropout) for _ in range(n_dg))

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return torch.stack([t(embedding) for t in self.dg_towers], dim=0).mean(dim=0)


class DeltaMLP(nn.Module):
    """Configurable MLP for direct ΔΔG regression on difference embeddings.

    Input is the concatenation of the normalized mutant embedding and the
    normalized difference embedding Δemb = emb(mut) − emb(wt). The difference
    channel is what carries the single-residue effect that mean pooling would
    otherwise bury, and it lifts held-out ΔΔG correlation from ~0.5 to ~0.8.

    Architecture (``widths``, ``dropout``) is stored in the checkpoint so the
    autoPyTorch-tuned configuration can be reconstructed at load time.
    """

    def __init__(self, input_dim: int, widths: list[int], dropout: float = 0.2) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.BatchNorm1d(input_dim)]
        prev = input_dim
        for w in widths:
            layers.extend([nn.Linear(prev, w), nn.GELU(), nn.Dropout(dropout)])
            prev = w
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x)).squeeze(-1)
