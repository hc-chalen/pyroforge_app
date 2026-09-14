"""Folding moves to the fallback endpoint only when the primary is unavailable.

On the GPU stack ESMFOLD_API_URL points at the self-hosted service
(services/esmfold) and ESMFOLD_FALLBACK_API_URL at the public ESM Atlas API.

The incident behind these: the public API returned 504 for every sequence it had
not folded before, and the app told the user to try a sequence under 400
residues — for an 84-residue protein.
"""

from __future__ import annotations

from typing import Callable

import httpx
import pytest

from app.api.errors import format_task_error
from app.core.config import settings
from app.engines import visualization
from app.engines.visualization import FoldServiceUnavailable, VisualizationEngine

PRIMARY = "http://esmfold:8000/foldSequence/v1/pdb/"
FALLBACK = "https://api.esmatlas.com/foldSequence/v1/pdb/"
SEQUENCE = (
    "MSGPNGDLGMPVEAGAEGEEDGFGEAEYAAINSMLDQINSCLDHLEEKND"
    "HLHARLQELLESNRQTRLEFQQQLGEAPSDASP"
)
PDB = "ATOM      1  N   MET A   1\n"

Handler = Callable[[httpx.Request], httpx.Response]


def ok(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, text=PDB)


def refused(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("connection refused", request=request)


def gateway_timeout(request: httpx.Request) -> httpx.Response:
    return httpx.Response(504, json={"message": "Endpoint request timed out"})


def rejected(request: httpx.Request) -> httpx.Response:
    return httpx.Response(422, text="Invalid amino acid characters: X")


@pytest.fixture
def endpoints(monkeypatch, tmp_path) -> tuple[dict[str, Handler], list[str]]:
    """Route each fold host to a handler the test sets, recording every call."""
    handlers: dict[str, Handler] = {}
    calls: list[str] = []

    def dispatch(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.host)
        return handlers[request.url.host](request)

    transport = httpx.MockTransport(dispatch)
    real_client = httpx.Client
    monkeypatch.setattr(
        visualization.httpx,
        "Client",
        lambda **kwargs: real_client(transport=transport, **kwargs),
    )
    monkeypatch.setattr(visualization.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(settings, "cache_dir", tmp_path)
    monkeypatch.setattr(settings, "esmfold_api_url", PRIMARY)
    monkeypatch.setattr(settings, "esmfold_fallback_api_url", FALLBACK)
    return handlers, calls


def test_primary_answers_and_the_fallback_is_never_called(endpoints) -> None:
    handlers, calls = endpoints
    handlers["esmfold"] = ok
    handlers["api.esmatlas.com"] = ok

    assert VisualizationEngine().fold_structure(SEQUENCE)["pdb"] == PDB
    assert calls == ["esmfold"]


def test_unreachable_primary_falls_back(endpoints) -> None:
    handlers, calls = endpoints
    handlers["esmfold"] = refused
    handlers["api.esmatlas.com"] = ok

    assert VisualizationEngine().fold_structure(SEQUENCE)["pdb"] == PDB
    assert calls == ["esmfold"] * visualization.MAX_FOLD_RETRIES + ["api.esmatlas.com"]


def test_a_rejected_sequence_is_not_sent_to_the_fallback(endpoints) -> None:
    handlers, calls = endpoints
    handlers["esmfold"] = rejected
    handlers["api.esmatlas.com"] = ok

    with pytest.raises(ValueError, match="rejected this sequence") as excinfo:
        VisualizationEngine().fold_structure(SEQUENCE)

    assert not isinstance(excinfo.value, FoldServiceUnavailable)
    assert calls == ["esmfold"]


def test_both_unavailable_raises_and_caches_nothing(endpoints, tmp_path) -> None:
    handlers, calls = endpoints
    handlers["esmfold"] = refused
    handlers["api.esmatlas.com"] = gateway_timeout

    with pytest.raises(FoldServiceUnavailable, match="504"):
        VisualizationEngine().fold_structure(SEQUENCE)

    assert list((tmp_path / "structures").iterdir()) == []


def test_without_a_fallback_only_the_primary_is_tried(endpoints, monkeypatch) -> None:
    handlers, calls = endpoints
    monkeypatch.setattr(settings, "esmfold_fallback_api_url", "")
    handlers["esmfold"] = gateway_timeout

    with pytest.raises(FoldServiceUnavailable):
        VisualizationEngine().fold_structure(SEQUENCE)

    assert set(calls) == {"esmfold"}


def test_a_gateway_timeout_is_not_blamed_on_the_sequence() -> None:
    message = format_task_error(
        "ESMFold API error (504). The service may be overloaded — "
        "please retry in a few minutes.",
    )

    assert "shorter sequence" not in message
    assert "not your sequence" in message


def test_504_inside_a_longer_number_is_not_a_gateway_timeout() -> None:
    message = "Sequence length 5041 exceeds the ESMFold limit (400 residues)."

    assert format_task_error(message) == message
