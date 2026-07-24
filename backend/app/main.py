from contextlib import asynccontextmanager
import logging
from threading import Thread

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router
from app.core.config import settings
from app.engines.registry import get_engine_registry
from app.services.esm2 import ESM2Service, get_esm2_service
from app.services.model_store import get_model_store

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    get_engine_registry()
    store = get_model_store()
    try:
        store.load()
        logger.info("Model checkpoints loaded from %s", settings.checkpoint_dir)
    except FileNotFoundError:
        logger.warning(
            "Checkpoints not found. POST /api/v1/admin/train to train models."
        )
    if settings.preload_esm2_on_startup and not ESM2Service.is_ready():
        logger.info(
            "Loading ESM-2 model in background (first run downloads ~2.5 GB)..."
        )
        Thread(target=get_esm2_service, daemon=True).start()
    yield


app = FastAPI(title="PyroForge API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_origin_regex=settings.cors_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/health")
def health() -> dict[str, str]:
    store = get_model_store()
    return {
        "status": "ok",
        "models_ready": str(store.is_ready).lower(),
        "esm_ready": str(ESM2Service.is_ready()).lower(),
    }
