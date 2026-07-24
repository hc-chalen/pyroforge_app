from pydantic import BaseModel, Field


class MutationSchema(BaseModel):
    position: int = Field(..., ge=0, description="0-indexed residue position")
    source_aa: str = Field(..., min_length=1, max_length=1)
    target_aa: str = Field(..., min_length=1, max_length=1)


class PredictRequest(BaseModel):
    sequence: str
    mutations: list[MutationSchema] | None = None


class PredictResponse(BaseModel):
    tm: float
    dg: float | None = None
    ddg: float | None = None


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


class OptimizationResultResponse(BaseModel):
    wild_type_sequence: str
    optimized_sequence: str
    mutations: list[MutationResult]
    baseline_tm: float
    tm: float
    ddg: float
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
    evalue: float
    bits: float
    description: str
    biobrick_url: str


class RegistrySearchResponse(BaseModel):
    matches: list[RegistryMatch]
    query_length: int


class ProcureRequest(BaseModel):
    protein_sequence: str
    organism: str = "ecoli"


class ProcureResponse(BaseModel):
    protein_sequence: str
    dna_sequence: str
    organism: str
    cai: float
    gc_content: float
    feasibility_flags: list[str]
    price_usd: float | None = None
    lead_time_days: int | None = None
    twist_status: str


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


class NasReportResponse(BaseModel):
    available: bool
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
