from typing import ClassVar

from pydantic import BaseModel, Field, field_validator


_AA_PATTERN = r"^[ACDEFGHIKLMNPQRSTVWY]$"


class MutationSchema(BaseModel):
    position: int = Field(..., ge=0, description="0-indexed residue position")
    # Verified against the sequence server-side, not decorative: it is the guard
    # against 1-indexed positions. Must be one of the 20 standard residues.
    source_aa: str = Field(..., pattern=_AA_PATTERN)
    target_aa: str = Field(..., pattern=_AA_PATTERN)

    @field_validator("source_aa", "target_aa", mode="before")
    @classmethod
    def _upper(cls, v: object) -> object:
        return v.upper() if isinstance(v, str) else v


class PredictRequest(BaseModel):
    sequence: str
    mutations: list[MutationSchema] | None = None


class PredictResponse(BaseModel):
    # ``tm`` is optional: when no model_tm checkpoint is installed the engine reports
    # it as unavailable rather than emitting NaN (which is not valid JSON).
    tm: float | None = None
    dg: float | None = None
    ddg: float | None = None
    tm_ci95: float | None = Field(
        default=None, description="±95% interval on absolute Tm (°C)"
    )
    dg_ci95: float | None = Field(
        default=None, description="±95% interval on absolute ΔG (kcal/mol)"
    )
    ddg_ci95: float | None = Field(
        default=None, description="±95% interval on ΔΔG (kcal/mol)"
    )
    tm_confidence: str | None = Field(
        default=None,
        description=(
            "'high' when the sequence length is inside model_tm's training window, "
            "'low' when the absolute Tm is extrapolated, 'unavailable' when no Tm "
            "model is installed"
        ),
    )
    tm_confidence_reason: str | None = Field(
        default=None, description="Why tm_confidence is not 'high'"
    )
    dg_confidence: str | None = Field(
        default=None,
        description=(
            "'high' when the sequence length is inside the small-domain regime the "
            "dG/ΔΔG models were trained on, 'low' when both are extrapolated"
        ),
    )
    dg_confidence_reason: str | None = Field(
        default=None, description="Why dg_confidence is not 'high'"
    )

    # Fields the prediction engine may emit. Kept in one place so the sync route and
    # the Celery task serialize the same payload.
    RESULT_FIELDS: ClassVar[tuple[str, ...]] = (
        "tm",
        "dg",
        "ddg",
        "tm_ci95",
        "dg_ci95",
        "ddg_ci95",
        "tm_confidence",
        "tm_confidence_reason",
        "dg_confidence",
        "dg_confidence_reason",
    )

    @classmethod
    def from_result(cls, result: dict) -> "PredictResponse":
        """Build the response from a ``PredictionEngine.predict_*`` dict."""
        return cls(**{k: result.get(k) for k in cls.RESULT_FIELDS})


class OptimizeRequest(BaseModel):
    sequence: str
    max_mutations: int = Field(default=5, ge=1, le=20)
    # Plausibility floor: ESM-2 masked-marginal probability of the substituted
    # residue must be ≥ this to be considered (0–1, higher = stricter). Despite the
    # legacy name this is a probability, not a perplexity.
    perplexity_threshold: float = Field(default=0.05, ge=0.0, le=1.0)


class OptimizeResponse(BaseModel):
    job_id: str
    status: str


class MutationResult(BaseModel):
    position: int
    source_aa: str
    target_aa: str


class RankedMutation(BaseModel):
    """One entry of the shortlist, with the evidence behind its position in it."""

    position: int
    source_aa: str
    target_aa: str
    ddg: float = Field(..., description="Predicted ΔΔG for this single substitution")
    residue_prob: float = Field(
        ..., description="ESM-2 masked-marginal probability of the substituted residue"
    )
    rank: int = Field(..., description="1-based rank within the shortlist")


class RankingQuality(BaseModel):
    """Held-out measurement of what the shortlist ordering is worth.

    Reported with every optimization so the shortlist is never presented as a set
    of established results. Measured by ``app.training.calibrate_ddg`` on
    parent-held-out proteins.
    """

    n_proteins: int | None = None
    base_rate_stabilizing: float | None = Field(
        default=None, description="Fraction of all mutations that are truly stabilizing"
    )
    median_within_protein_spearman: float | None = None
    headline_k: int | None = None
    top_k_precision: dict[str, float] | None = Field(
        default=None, description="Fraction of the top-k that are truly stabilizing"
    )
    headline_precision_ci95: list[float] | None = None
    headline_precision_min: float | None = None
    headline_precision_max: float | None = None
    proteins_below_half_precision: int | None = Field(
        default=None,
        description="Proteins where fewer than half the top-k were truly stabilizing",
    )


class OptimizationResultResponse(BaseModel):
    wild_type_sequence: str
    optimized_sequence: str
    mutations: list[MutationResult]
    # No variant `tm` is reported: model_tm barely moves for a handful of point
    # mutations, so it would sit at ≈ baseline and be misread as a thermal shift.
    baseline_tm: float | None = None
    ddg: float
    ddg_ci95: float | None = None
    # Same length gate /predict applies to ΔΔG — the optimizer's ranking comes from
    # the same model, so it carries the same flag.
    dg_confidence: str | None = None
    dg_confidence_reason: str | None = None
    mutation_details: list[RankedMutation] = Field(default_factory=list)
    ranking_quality: RankingQuality | None = None
    scan_summary: dict


class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: float | None = None
    result: dict | None = None
    error: str | None = None


class FoldRequest(BaseModel):
    sequence: str


class FoldResponse(BaseModel):
    job_id: str
    status: str


class HeatmapRequest(BaseModel):
    sequence: str
    residue_scores: dict[int, float]


class HeatmapResponse(BaseModel):
    residues: list[dict]


class RegistrySearchRequest(BaseModel):
    sequence: str
    max_hits: int = Field(default=10, ge=1, le=50)


class RegistryMatch(BaseModel):
    part_id: str
    identity: float
    # None on the pairwise fallback path, which has no database model to compute an
    # E-value from. Reporting 0.0 there would read as a perfect hit.
    evalue: float | None = None
    bits: float
    description: str
    biobrick_url: str
    search_method: str | None = None


class RegistryDatabaseInfo(BaseModel):
    """What was actually searched.

    Reported with every result because without it an empty match list is
    unreadable: "this protein has no homolog in the Registry" and "the local
    snapshot holds three sequences" look identical to the user otherwise.
    """

    status: str = Field(..., description="'ready' or 'not_provisioned'")
    n_parts: int
    source: str | None = None
    built_at: str | None = None
    detail: str | None = None


class RegistrySearchResponse(BaseModel):
    matches: list[RegistryMatch]
    query_length: int
    database: RegistryDatabaseInfo
    # Non-null when no search could be run at all, which is a different answer
    # from a search that returned nothing.
    unavailable_reason: str | None = None
    unavailable_detail: str | None = None


class ProcureRequest(BaseModel):
    protein_sequence: str
    organism: str = "ecoli"


class ProcureResponse(BaseModel):
    protein_sequence: str
    dna_sequence: str
    organism: str
    cai: float
    cai_reference: str = Field(
        ...,
        description=(
            "The codon usage reference set the CAI was scored against. A CAI is "
            "only interpretable relative to a stated reference, so it is always "
            "reported alongside the number."
        ),
    )
    gc_content: float = Field(..., description="GC content as a percentage (0–100)")
    feasibility_flags: list[str]
    price_usd: float | None = None
    lead_time_days: int | None = None
    twist_status: str
    vendor: str | None = None
    # A price with no key configured is a local estimate, not a Twist quote. The
    # flag is structural so the UI cannot render the two identically by omission.
    price_is_estimate: bool = True
    price_basis: str | None = Field(
        default=None, description="Exactly where the price came from"
    )
    synthesis_product: str | None = None
    order_url: str | None = Field(
        default=None, description="Twist's order page — where a real order is placed"
    )
    pricing_url: str | None = None


class TrainRequest(BaseModel):
    epochs: int = Field(default=30, ge=1, le=200)
    batch_size: int = Field(default=256, ge=16, le=1024)
    lr: float = Field(default=1e-3, gt=0)


class TrainResponse(BaseModel):
    job_id: str
    status: str


class EmbedCompareRequest(BaseModel):
    sequence_original: str
    sequence_optimized: str


class EmbedCompareResponse(BaseModel):
    similarity: float = Field(
        ...,
        description="Normalized similarity in [0, 1] derived from cosine similarity",
    )
    cosine_similarity: float
    mean_absolute_difference: float
    l1_distance: float
    max_absolute_difference: float
    embedding_mean_abs_original: float
    embedding_mean_abs_optimized: float
    embedding_norm_original: float
    embedding_norm_optimized: float
    embedding_dim: int
    sequence_a_length: int
    sequence_b_length: int
    sequence_identity: float | None = None
    mutation_count: int | None = None


class NasValMetrics(BaseModel):
    mae: float | None = None
    rmse: float | None = None
    pearson_r: float | None = None


class NasModelWeighting(BaseModel):
    loss_task_weights: str
    weight_decay: float | None = None
    weight_decay_label: str
    search_time_share: float | None = None


class NasAlgorithmNames(BaseModel):
    search: str
    architecture: str
    activation: str
    optimizer: str
    normalization: str
    loss: str


class NasHyperparameters(BaseModel):
    activation: str | None = None
    optimizer: str | None = None
    input_normalization: str | None = None
    loss: str | None = None
    dropout: float | None = None
    learning_rate: float | None = None
    weight_decay: float | None = None
    batch_size: int | None = None
    hidden_widths: list[int] | None = None


class NasModelSummary(BaseModel):
    key: str
    label: str
    unit: str
    architecture: str
    n_layers: int
    best_budget_epochs: int | None = None
    time_limit_secs: int | None = None
    algorithm_names: NasAlgorithmNames
    weighting: NasModelWeighting
    val_metrics: NasValMetrics
    hyperparameters: NasHyperparameters


class DdgRankingSummary(BaseModel):
    pooled_pearson_r: float | None = None
    pooled_mae: float | None = None
    within_protein_median_spearman: float | None = None
    within_protein_n_proteins: int | None = None
    note: str | None = None


class NasReportResponse(BaseModel):
    available: bool
    ddg_ranking: DdgRankingSummary | None = None
    method: str | None = None
    design: str | None = None
    note: str | None = None
    total_time_limit_secs: int | None = None
    time_budget: dict[str, int] | None = None
    search_budget_weights: dict[str, float] | None = None
    production_weighting: dict | None = None
    seed: int | None = None
    device: str | None = None
    models: list[NasModelSummary] = Field(default_factory=list)
    comparison: dict | None = None
