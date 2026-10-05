"""The app must at least import.

Nothing in the suite reached `app.api.routes` before this, so a syntax error
there passed 86 green tests and only surfaced as a container that would not
start. Importing the FastAPI app exercises every route module, schema and
service import in one line.
"""

from __future__ import annotations


def test_the_fastapi_app_imports() -> None:
    from app.main import app

    assert app.routes


def test_every_api_route_is_registered() -> None:
    from app.main import app

    # The OpenAPI schema, not app.routes: since FastAPI 0.138 include_router keeps
    # its routes inside an _IncludedRouter instead of flattening them, so walking
    # app.routes finds only /health and the docs.
    paths = set(app.openapi()["paths"])

    # One per card the UI drives; a route lost to a bad merge is a dead feature.
    for path in (
        "/api/v1/predict/job",
        "/api/v1/optimize",
        "/api/v1/visualize/fold",
        "/api/v1/visualize/heatmap",
        "/api/v1/registry/search",
        "/api/v1/procure/quote",
        "/api/v1/embed/compare",
        "/api/v1/jobs/{job_id}",
    ):
        assert path in paths, f"{path} is not registered"
