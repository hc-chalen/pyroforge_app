from __future__ import annotations

from app.api.schemas import PredictResponse
from app.core.celery_app import celery_app
from app.engines.prediction import Mutation
from app.engines.registry import get_engine_registry
from app.services.esm2 import get_esm2_service
from app.training.train import run_training


@celery_app.task(bind=True, name="app.tasks.predict_sequence")
def predict_sequence(
    self,
    sequence: str,
    mutations: list[dict] | None = None,
) -> dict:
    self.update_state(state="PROGRESS", meta={"progress": 0.1})
    reg = get_engine_registry()
    parsed_mutations = None
    if mutations:
        parsed_mutations = [
            Mutation(
                position=m["position"],
                source_aa=m["source_aa"],
                target_aa=m["target_aa"],
            )
            for m in mutations
        ]
    result = reg.prediction.predict_full(sequence, parsed_mutations)
    self.update_state(state="PROGRESS", meta={"progress": 0.9})
    # Same payload as the synchronous /predict route, uncertainty + confidence
    # flags included.
    return PredictResponse.from_result(result).model_dump()


@celery_app.task(bind=True, name="app.tasks.optimize_sequence")
def optimize_sequence(
    self,
    sequence: str,
    max_mutations: int = 5,
    perplexity_threshold: float = 0.05,
) -> dict:
    reg = get_engine_registry()

    def progress(done: int, total: int) -> None:
        self.update_state(state="PROGRESS", meta={"progress": done / max(total, 1)})

    result = reg.optimization.optimize(
        sequence,
        max_mutations=max_mutations,
        perplexity_threshold=perplexity_threshold,
        progress_callback=progress,
    )
    # Reuse the embeddings the optimizer already computed — these are the same two
    # sequences, so re-embedding them would be two redundant 650M forward passes.
    structure_comparison = get_esm2_service().compare_embeddings(
        result.wild_type_sequence,
        result.optimized_sequence,
        emb_a=result.wt_embedding,
        emb_b=result.variant_embedding,
    )
    return {
        "wild_type_sequence": result.wild_type_sequence,
        "optimized_sequence": result.optimized_sequence,
        "mutations": [
            {"position": m.position, "source_aa": m.source_aa, "target_aa": m.target_aa}
            for m in result.mutations
        ],
        "baseline_tm": result.baseline_tm,
        "ddg": result.ddg,
        "ddg_ci95": result.ddg_ci95,
        "scan_summary": result.scan_summary,
        "structure_comparison": structure_comparison,
    }


@celery_app.task(bind=True, name="app.tasks.fold_structure")
def fold_structure(self, sequence: str) -> dict:
    reg = get_engine_registry()
    self.update_state(state="PROGRESS", meta={"progress": 0.1})
    fold = reg.visualization.fold_structure(sequence)
    rmsf = reg.visualization.approximate_rmsf(fold["pdb_path"])
    self.update_state(state="PROGRESS", meta={"progress": 0.9})
    return {
        "pdb": fold["pdb"],
        "pdb_path": fold["pdb_path"],
        "sequence_length": fold["sequence_length"],
        "rmsf": rmsf,
    }


@celery_app.task(bind=True, name="app.tasks.registry_search")
def registry_search(self, sequence: str, max_hits: int = 10) -> dict:
    reg = get_engine_registry()
    self.update_state(state="PROGRESS", meta={"progress": 0.5})
    return reg.registry.search(sequence, max_hits=max_hits)


@celery_app.task(bind=True, name="app.tasks.train_models")
def train_models(
    self,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> dict:
    self.update_state(state="PROGRESS", meta={"progress": 0.1})
    report = run_training(epochs=epochs, batch_size=batch_size, lr=lr)
    self.update_state(state="PROGRESS", meta={"progress": 0.95})
    from app.services.model_store import get_model_store

    store = get_model_store()
    store.load()
    return report
