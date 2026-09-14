"""Self-hosted ESMFold, speaking the ESM Atlas foldSequence contract.

PyroForge's worker folds by POSTing a bare sequence to ESMFOLD_API_URL and
reading PDB text back — the contract of api.esmatlas.com/foldSequence/v1/pdb/.
This service answers the same request, so pointing that URL here is the entire
integration.

One process, one model, one GPU, one fold at a time. A GPU does not time-slice a
3.5B-parameter model the way a CPU shares cores: a second concurrent fold only
competes for the same card. Requests therefore wait for a single slot, and the
line is bounded (ESMFOLD_MAX_WAITING below).

Run with exactly one uvicorn worker — each extra worker loads another ~8 GB copy
of the model onto the card.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from fastapi.responses import PlainTextResponse

from folding import OutOfMemory, load_esmfold

VALID_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")

# Same ceiling the backend validates against before it ever calls this service
# (ESMFOLD_MAX_LENGTH in backend/app/engines/visualization.py).
MAX_LENGTH = int(os.environ.get("ESMFOLD_MAX_LENGTH", "400"))

# Requests allowed to wait behind the running fold before new ones get a 503.
# Sized against the worker's 300 s read timeout (FOLD_READ_TIMEOUT_S): the
# measured worst case on the RTX 2080 Ti is 22.8 s for 400 residues, so 8 waiting
# + 1 running is ~205 s and every admitted request finishes before its caller
# gives up. Redo that arithmetic before raising it.
MAX_WAITING = int(os.environ.get("ESMFOLD_MAX_WAITING", "8"))

# Folding-trunk chunk size. 64 peaked at 9.0 GB for 400 residues on a 22 GB card;
# lower trades speed for memory.
CHUNK_SIZE = int(os.environ.get("ESMFOLD_CHUNK_SIZE", "64"))

RETRY_AFTER_S = 30


class QueueFull(Exception):
    """More requests are already waiting than the queue admits."""


class FoldQueue:
    """One GPU slot, and a bounded line of requests waiting for it."""

    def __init__(self, fold: Callable[[str], str], max_waiting: int) -> None:
        self._fold = fold
        self._slot = asyncio.Lock()
        self.max_waiting = max_waiting
        self.waiting = 0
        self.busy = False

    async def submit(
        self,
        sequence: str,
        caller_gone: Callable[[], Awaitable[bool]],
    ) -> str | None:
        """Fold once the slot is free. None if the caller left while queued."""
        if self._slot.locked() and self.waiting >= self.max_waiting:
            raise QueueFull

        self.waiting += 1
        try:
            await self._slot.acquire()
        finally:
            self.waiting -= 1

        try:
            # A caller that timed out in the queue has already retried or moved on
            # to the fallback API. Folding for it would only delay everyone behind.
            if await caller_gone():
                return None
            self.busy = True
            return await asyncio.to_thread(self._fold, sequence)
        finally:
            self.busy = False
            self._slot.release()


def _problem_with(sequence: str, max_length: int) -> str | None:
    if not sequence:
        return "Empty sequence."
    if len(sequence) > max_length:
        return f"Sequence length {len(sequence)} exceeds the {max_length}-residue limit."
    invalid = sorted(set(sequence) - VALID_AMINO_ACIDS)
    if invalid:
        return f"Invalid amino acid characters: {''.join(invalid)}"
    return None


def _retry_later(message: str) -> PlainTextResponse:
    return PlainTextResponse(
        message,
        status_code=503,
        headers={"Retry-After": str(RETRY_AFTER_S)},
    )


def create_app(
    load_fold: Callable[[], Callable[[str], str]] | None = None,
    *,
    max_length: int = MAX_LENGTH,
    max_waiting: int = MAX_WAITING,
) -> FastAPI:
    if load_fold is None:
        def load_fold() -> Callable[[str], str]:
            return load_esmfold(CHUNK_SIZE)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Blocks for the model load (~1 min). uvicorn accepts no connections until
        # this returns, so the healthcheck stays red until folds can actually run.
        app.state.queue = FoldQueue(load_fold(), max_waiting)
        yield

    app = FastAPI(title="PyroForge ESMFold", lifespan=lifespan)

    @app.post("/foldSequence/v1/pdb/")
    async def fold(request: Request) -> Response:
        sequence = (await request.body()).decode(errors="replace").strip().upper()
        problem = _problem_with(sequence, max_length)
        if problem:
            return PlainTextResponse(problem, status_code=422)

        try:
            pdb = await request.app.state.queue.submit(sequence, request.is_disconnected)
        except QueueFull:
            return _retry_later("The fold queue is full. Retry shortly.")
        except OutOfMemory as exc:
            return _retry_later(str(exc))

        if pdb is None:
            # nginx's "client closed request". Nobody is left to read it; it only
            # makes skipped folds recognisable in the access log.
            return Response(status_code=499)
        return PlainTextResponse(pdb)

    @app.get("/health")
    async def health(request: Request) -> dict:
        queue = request.app.state.queue
        return {
            "status": "ok",
            "busy": queue.busy,
            "waiting": queue.waiting,
            "max_waiting": queue.max_waiting,
        }

    return app


app = create_app()
