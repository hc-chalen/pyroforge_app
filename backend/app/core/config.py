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

    cors_origins: list[str] = ["http://localhost:5173"]
    # Vite picks the next free port (5174, 5175, …) when 5173 is taken.
    cors_origin_regex: str = r"http://localhost:\d+"

    @property
    def resolved_device(self) -> str:
        if self.device == "auto":
            import torch

            if torch.cuda.is_available():
                return "cuda"
            # Apple Silicon (M-series) GPU — much faster than CPU for ESM-2.
            if torch.backends.mps.is_available():
                return "mps"
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
