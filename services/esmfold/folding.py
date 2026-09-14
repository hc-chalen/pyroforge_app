"""ESMFold model loading, kept apart from server.py.

The HTTP layer imports only the names defined here, never torch or transformers
at module level, so its tests run without either installed.
"""

from __future__ import annotations

from typing import Callable

# A short, ordinary protein used to warm the CUDA kernels before the service
# reports healthy — the first fold after load is several seconds slower.
_WARMUP_SEQUENCE = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"


class OutOfMemory(RuntimeError):
    """The fold did not fit in GPU memory. Retryable: the CUDA cache is freed."""


def load_esmfold(chunk_size: int) -> Callable[[str], str]:
    """Load ESMFold onto the visible GPU and return a sequence -> PDB function."""
    import torch
    from transformers import EsmForProteinFolding

    if not torch.cuda.is_available():
        raise RuntimeError("The ESMFold service needs a CUDA device; none is visible.")

    model = EsmForProteinFolding.from_pretrained(
        "facebook/esmfold_v1",
        low_cpu_mem_usage=True,
    )
    model = model.cuda().eval()

    # The settings benchmark.py measured on the RTX 2080 Ti (9.0 GB peak at 400
    # residues): fp16 ESM-2 stem, fp32 folding trunk. Turing has no bfloat16 —
    # do not switch the stem to it.
    model.esm = model.esm.half()
    torch.backends.cuda.matmul.allow_tf32 = True  # Ampere+ only; a no-op on Turing
    model.trunk.set_chunk_size(chunk_size)

    def fold(sequence: str) -> str:
        try:
            with torch.no_grad():
                return model.infer_pdb(sequence)
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise OutOfMemory(
                f"Out of GPU memory folding {len(sequence)} residues.",
            ) from exc

    fold(_WARMUP_SEQUENCE)
    return fold
