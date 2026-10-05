from fastapi import APIRouter, Depends, HTTPException
from celery.result import AsyncResult

from app.api.schemas import (
    EmbedCompareRequest,
    EmbedCompareResponse,
    FoldRequest,
    FoldResponse,
    HeatmapRequest,
    HeatmapResponse,
    JobStatus,
    NasReportResponse,
    OptimizeRequest,
    OptimizeResponse,
    PredictRequest,
    PredictResponse,
    ProcureRequest,
    ProcureResponse,
    RegistryDatabaseInfo,
    RegistrySearchRequest,
    RegistrySearchResponse,
    TrainRequest,
    TrainResponse,
)
from app.services.esm2 import ESM2Service, get_esm2_service
from app.services.nas_report import load_nas_report
from app.services.result_cache import get_result_cache
from app.api.errors import format_task_error
from app.core.celery_app import celery_app
from app.core.config import settings
from app.engines.prediction import Mutation
from app.engines.registry import EngineRegistry, get_engine_registry
from app.tasks.jobs import (
    fold_structure,
    optimize_sequence,
    predict_sequence,
    registry_search,
    train_models,
)

router = APIRouter(prefix="/api/v1")


def engines() -> EngineRegistry:
    return get_engine_registry()


@router.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, reg: EngineRegistry = Depends(engines)) -> PredictResponse:
    try:
        mutations = None
        if body.mutations:
            mutations = [
                Mutation(position=m.position, source_aa=m.source_aa, target_aa=m.target_aa)
                for m in body.mutations
            ]
        result = reg.prediction.predict_full(body.sequence, mutations)
        return PredictResponse.from_result(result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/predict/job", response_model=OptimizeResponse)
def predict_job(body: PredictRequest) -> OptimizeResponse:
    # Validated here, not in the worker, so bad input is a 400 rather than a job that
    # is accepted and then fails.
    try:
        seq = ESM2Service.validate_sequence(body.sequence)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    mutations = None
    if body.mutations:
        mutations = [m.model_dump() for m in body.mutations]
    task = predict_sequence.delay(seq, mutations)
    return OptimizeResponse(job_id=task.id, status="pending")


@router.post("/optimize", response_model=OptimizeResponse)
def optimize(body: OptimizeRequest) -> OptimizeResponse:
    # Validate before enqueueing: otherwise a bad sequence is only rejected inside
    # the worker and surfaces as a failed job rather than a 400. The scan is also
    # O(L × 20) serial ESM-2 passes, so an over-long sequence would hold the single
    # worker slot for hours — reject it here.
    try:
        seq = ESM2Service.validate_sequence(body.sequence)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if len(seq) > settings.optimize_max_length:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Sequence length {len(seq)} exceeds the optimization limit "
                f"({settings.optimize_max_length} residues). The site-saturation scan "
                f"runs ~{len(seq) * 20} ESM-2 forward passes serially; shorten the "
                f"sequence or raise OPTIMIZE_MAX_LENGTH."
            ),
        )
    task = optimize_sequence.delay(
        seq,
        body.max_mutations,
        body.perplexity_threshold,
    )
    return OptimizeResponse(job_id=task.id, status="pending")


@router.get("/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str) -> JobStatus:
    result = AsyncResult(job_id, app=celery_app)
    state = result.state
    if state == "PENDING":
        return JobStatus(job_id=job_id, status="pending")
    if state == "PROGRESS":
        meta = result.info or {}
        return JobStatus(
            job_id=job_id,
            status="running",
            progress=meta.get("progress", 0),
        )
    if state == "SUCCESS":
        return JobStatus(job_id=job_id, status="completed", result=result.result)
    if state == "FAILURE":
        return JobStatus(
            job_id=job_id,
            status="failed",
            error=format_task_error(result.result),
        )
    return JobStatus(job_id=job_id, status=state.lower())


@router.post("/visualize/fold", response_model=FoldResponse)
def fold(body: FoldRequest, reg: EngineRegistry = Depends(engines)) -> FoldResponse:
    try:
        reg.visualization.validate_sequence(body.sequence)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    task = fold_structure.delay(body.sequence)
    return FoldResponse(job_id=task.id, status="pending")


@router.post("/visualize/heatmap", response_model=HeatmapResponse)
def heatmap(
    body: HeatmapRequest,
    reg: EngineRegistry = Depends(engines),
) -> HeatmapResponse:
    try:
        residues = reg.visualization.build_heatmap(body.sequence, body.residue_scores)
        return HeatmapResponse(residues=residues)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/registry/status", response_model=RegistryDatabaseInfo)
def registry_status(reg: EngineRegistry = Depends(engines)) -> RegistryDatabaseInfo:
    """Whether this server has a parts snapshot at all.

    Cheap, synchronous and Celery-free, so the card can say "unavailable on this
    server" before the user submits a sequence and waits on a job that was never
    going to search anything.
    """
    return RegistryDatabaseInfo(**reg.registry.db.manifest())


@router.post("/registry/search", response_model=OptimizeResponse)
def registry(body: RegistrySearchRequest) -> OptimizeResponse:
    # Validated up front so malformed input is a 400, not a failed background job.
    try:
        seq = ESM2Service.validate_sequence(body.sequence)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    task = registry_search.delay(seq, body.max_hits)
    return OptimizeResponse(job_id=task.id, status="pending")


@router.post("/procure/quote", response_model=ProcureResponse)
def procure(
    body: ProcureRequest,
    reg: EngineRegistry = Depends(engines),
) -> ProcureResponse:
    try:
        result = reg.procurement.quote(body.protein_sequence, body.organism)
        return ProcureResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/admin/train", response_model=TrainResponse)
def train(body: TrainRequest) -> TrainResponse:
    task = train_models.delay(body.epochs, body.batch_size, body.lr)
    return TrainResponse(job_id=task.id, status="pending")


@router.get("/models/nas-report", response_model=NasReportResponse)
def nas_report() -> NasReportResponse:
    report = load_nas_report()
    if report is None:
        return NasReportResponse(available=False)
    return NasReportResponse(**report)


@router.post("/embed/compare", response_model=EmbedCompareResponse)
def compare_embeddings(body: EmbedCompareRequest) -> EmbedCompareResponse:
    # Two ESM-2 forward passes inside the API process — ~21 s for a pair of 84-mers,
    # and no Celery queue in front of it. It is also the card a whole room reaches at
    # once as soon as optimize starts coming back from cache.
    cache = get_result_cache("embed_compare")
    cache_key = {
        "original": body.sequence_original.strip().upper(),
        "optimized": body.sequence_optimized.strip().upper(),
    }
    def compute() -> dict:
        esm2 = get_esm2_service()
        result = esm2.compare_embeddings(
            body.sequence_original,
            body.sequence_optimized,
        )
        # Store the validated payload, not the raw engine output: that one carries
        # numpy scalars, which do not survive json.dumps.
        return EmbedCompareResponse(**result).model_dump()

    try:
        payload = cache.get_or_compute(cache_key, compute)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return EmbedCompareResponse(**payload)
