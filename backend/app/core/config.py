from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_ROOT = Path(__file__).resolve().parents[2]
_REPO_ROOT = _BACKEND_ROOT.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    redis_url: str = "redis://localhost:6379/0"

    data_path: Path = _REPO_ROOT / "esm2_train.pt"
    test_data_path: Path = _REPO_ROOT / "esm2_test.pt"
    checkpoint_dir: Path = _BACKEND_ROOT / "checkpoints"
    cache_dir: Path = _BACKEND_ROOT / "cache"
    igem_parts_fasta_path: Path = _BACKEND_ROOT / "data" / "igem_parts.fasta"
    mmseqs_db_path: Path = _BACKEND_ROOT / "data" / "igem_mmseqs_db"

    device: str = "auto"  # auto → CUDA, else Apple-Silicon MPS, else CPU
    esm_model_name: str = "esm2_t33_650M_UR50D"
    preload_esm2_on_startup: bool = True

    esmfold_api_url: str = "https://api.esmatlas.com/foldSequence/v1/pdb/"
    igem_api_base: str = "https://parts.igem.org/cgi"
    twist_api_key: str = ""
    twist_api_base: str = "https://api.twistbioscience.com"

    mmseqs_bin: str = "mmseqs"
    perplexity_threshold: float = 0.05
    max_mutations_default: int = 5
    celery_result_expires: int = 86400

    # NOTE: ddg_effect_size_sigma / ddg_min_effect_fallback were removed. They
    # configured a kσ "significance gate" on each candidate mutation, which was a
    # per-comparison threshold applied to the L×19 candidates of a whole scan — it
    # did not control the selection bias it was introduced for, and at the shipped
    # σ it returned nothing at all for 20 of 22 held-out proteins. Selection is now
    # a ranked top-k shortlist with a sign floor; see OptimizationEngine.

    # Length regime the dG/ΔΔG models were actually trained on. Measured on
    # esm2_train.pt: over the 513,689 dG-labelled rows, a lower-bound length proxy
    # (highest mutated position + 1, since no sequence column is stored) has
    # p50 = 60 and p95 = 72 residues — the Tsuboyama 2023 small-designed-domain
    # regime. Outside it, absolute dG and ΔΔG are extrapolation and are reported as
    # low confidence, mirroring the model_tm length gate.
    dg_trusted_min_len: int = 10
    dg_trusted_max_len: int = 72

    # Hard ceiling for the site-saturation scan. The scan is O(L × 20) ESM-2 650M
    # forward passes run serially, so a long sequence occupies the single worker
    # slot for hours. Reject up front instead.
    optimize_max_length: int = 400

    # Celery task limits. The worker pool is single-slot ("solo"), so an unbounded
    # task is a denial of service for every other user. Soft limit first so the task
    # can report a real error; hard limit as the backstop.
    task_soft_time_limit: int = 3600
    task_time_limit: int = 3900

    cors_origins: list[str] = ["http://localhost:5173"]
    # Vite picks the next free port (5174, 5175, …) when 5173 is taken.
    cors_origin_regex: str = r"http://localhost:\d+"

    @property
    def resolved_device(self) -> str:
        import torch

        if self.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            # Apple Silicon (M-series) GPU — much faster than CPU for ESM-2.
            if torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        # An explicitly-set device must still be honest about what's available:
        # returning "cuda"/"mps" on a machine without it makes torch.load(
        # map_location=...) raise "torch.cuda.is_available() is False" instead of
        # loading the checkpoint. Degrade to CPU rather than crash — e.g. a stray
        # DEVICE=cuda leaking into a CPU-only box (a teammate trying the repo).
        if self.device == "cuda" and not torch.cuda.is_available():
            return "cpu"
        if self.device == "mps" and not torch.backends.mps.is_available():
            return "cpu"
        return self.device

    @property
    def esm_device(self) -> str:
        """Device for the ESM-2 forward pass.

        ESM-2 650M produces numerically degenerate embeddings on the Apple MPS
        backend (near-constant mean-pooled vectors, cosine ~0.4-0.6 vs CPU), which
        silently collapses every downstream Tm/dG/ΔΔG prediction to the training
        mean. Training embeddings are made on CPU/CUDA, so serving must match. Keep
        CUDA (correct + fast); never run ESM-2 on MPS — fall back to CPU instead.
        """
        dev = self.resolved_device
        return "cpu" if dev == "mps" else dev


settings = Settings()
