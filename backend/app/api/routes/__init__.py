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
    RegistrySearchRequest,
    RegistrySearchResponse,
    TrainRequest,
    TrainResponse,
)
from app.services.esm2 import get_esm2_service
from app.services.nas_report import load_nas_report
from app.api.errors import format_task_error
from app.core.celery_app import celery_app
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
        return PredictResponse(
            tm=result["tm"],
            dg=result.get("dg"),
            ddg=result.get("ddg"),
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/predict/job", response_model=OptimizeResponse)
def predict_job(body: PredictRequest) -> OptimizeResponse:
    mutations = None
    if body.mutations:
        mutations = [m.model_dump() for m in body.mutations]
    task = predict_sequence.delay(body.sequence, mutations)
    return OptimizeResponse(job_id=task.id, status="pending")


@router.post("/optimize", response_model=OptimizeResponse)
def optimize(body: OptimizeRequest) -> OptimizeResponse:
    task = optimize_sequence.delay(
        body.sequence,
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


@router.post("/registry/search", response_model=OptimizeResponse)
def registry(body: RegistrySearchRequest) -> OptimizeResponse:
    task = registry_search.delay(body.sequence, body.max_hits)
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
    try:
        esm2 = get_esm2_service()
        result = esm2.compare_embeddings(
            body.sequence_original,
            body.sequence_optimized,
        )
        return EmbedCompareResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
