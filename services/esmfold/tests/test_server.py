"""The fold service's contract: the request/response shape of
api.esmatlas.com/foldSequence/v1/pdb/, one fold at a time, and a bounded queue.

The model is replaced by a plain function, so these run without torch or
transformers. The backend's environment has everything else they need:

    cd backend && uv run pytest ../services/esmfold/tests -q
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest
from fastapi.testclient import TestClient

from folding import OutOfMemory
from server import FoldQueue, QueueFull, create_app

# The 84-residue protein the public API returned 504 for.
SEQUENCE = (
    "MSGPNGDLGMPVEAGAEGEEDGFGEAEYAAINSMLDQINSCLDHLEEKND"
    "HLHARLQELLESNRQTRLEFQQQLGEAPSDASP"
)
FOLD_URL = "/foldSequence/v1/pdb/"


async def still_connected() -> bool:
    return False


async def already_gone() -> bool:
    return True


async def wait_until(condition, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.005)


def client_for(fold, **kwargs) -> TestClient:
    return TestClient(create_app(lambda: fold, **kwargs))


def test_folds_a_sequence_with_the_esm_atlas_contract() -> None:
    received = []

    def fold(sequence: str) -> str:
        received.append(sequence)
        return "PDB TEXT"

    with client_for(fold) as client:
        response = client.post(
            FOLD_URL,
            content=f"  {SEQUENCE.lower()}\n",
            headers={"Content-Type": "text/plain"},
        )

    assert response.status_code == 200
    assert response.text == "PDB TEXT"
    assert received == [SEQUENCE]


@pytest.mark.parametrize("body", ["", "MKTAYIAKQX", "MKT AYI"])
def test_invalid_sequences_are_rejected_with_422(body: str) -> None:
    with client_for(lambda sequence: "PDB") as client:
        response = client.post(FOLD_URL, content=body)

    assert response.status_code == 422


def test_sequence_over_the_length_limit_is_rejected() -> None:
    with client_for(lambda sequence: "PDB", max_length=50) as client:
        response = client.post(FOLD_URL, content="A" * 51)

    assert response.status_code == 422


def test_out_of_memory_is_a_retryable_503() -> None:
    def fold(sequence: str) -> str:
        raise OutOfMemory("Out of GPU memory folding 400 residues.")

    with client_for(fold) as client:
        response = client.post(FOLD_URL, content=SEQUENCE)

    assert response.status_code == 503
    assert response.headers["Retry-After"]


def test_health_reports_the_queue() -> None:
    with client_for(lambda sequence: "PDB", max_waiting=3) as client:
        body = client.get("/health").json()

    assert body == {"status": "ok", "busy": False, "waiting": 0, "max_waiting": 3}


def test_folds_run_one_at_a_time() -> None:
    lock = threading.Lock()
    running = {"now": 0, "most": 0}

    def fold(sequence: str) -> str:
        with lock:
            running["now"] += 1
            running["most"] = max(running["most"], running["now"])
        time.sleep(0.02)
        with lock:
            running["now"] -= 1
        return sequence

    async def scenario() -> list[str | None]:
        queue = FoldQueue(fold, max_waiting=10)
        return await asyncio.gather(
            *(queue.submit(s, still_connected) for s in ["A", "B", "C", "D"]),
        )

    assert asyncio.run(scenario()) == ["A", "B", "C", "D"]
    assert running["most"] == 1


def test_queue_rejects_requests_beyond_max_waiting() -> None:
    release = threading.Event()

    def fold(sequence: str) -> str:
        release.wait(timeout=5)
        return sequence

    async def scenario() -> None:
        queue = FoldQueue(fold, max_waiting=1)
        first = asyncio.create_task(queue.submit("A", still_connected))
        await wait_until(lambda: queue.busy)
        second = asyncio.create_task(queue.submit("B", still_connected))
        await wait_until(lambda: queue.waiting == 1)

        try:
            with pytest.raises(QueueFull):
                await queue.submit("C", still_connected)
        finally:
            release.set()

        assert await first == "A"
        assert await second == "B"

    asyncio.run(scenario())


def test_request_whose_caller_left_is_not_folded() -> None:
    folded = []

    async def scenario() -> str | None:
        queue = FoldQueue(lambda s: folded.append(s) or "PDB", max_waiting=1)
        return await queue.submit(SEQUENCE, already_gone)

    assert asyncio.run(scenario()) is None
    assert folded == []
