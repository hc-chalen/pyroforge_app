"""Pytest guard: prove the tests import *this* checkout's ``app`` package.

More than one PyroForge checkout can exist on a development machine (this one was
previously laid out under a different directory name, and a stale editable install
kept resolving ``app`` there). When that happens pytest still reports ``rootdir``
as this repository while exercising code from somewhere else.

An ``ImportError`` is the lucky outcome — it is visible. The dangerous case is a
sibling checkout that defines the same names: the suite passes, against code that
is not the code under review. The same hazard applies to
``scripts/benchmark_grouped_cv.py``, which would emit a perfectly well-formed
report full of numbers from the wrong codebase (hence its ``code_path`` field).

This deliberately **checks without repairing**. Prepending the right path here
would make pytest pass while every other entry point — the scripts, the app, the
Celery worker — still loaded the wrong package: a green suite would then be
evidence of nothing. The environment itself has to be correct, so a mismatch is
raised with the command that fixes it (``uv sync --reinstall``).
"""
from __future__ import annotations

from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent


def _assert_app_is_local() -> None:
    import app

    resolved = Path(app.__file__).resolve().parent
    expected = BACKEND_ROOT / "app"
    if resolved != expected:
        raise RuntimeError(
            "pytest is importing a DIFFERENT PyroForge checkout.\n"
            f"  expected: {expected}\n"
            f"  actually: {resolved}\n\n"
            "Test results from that package say nothing about this working tree.\n\n"
            "Usual cause: the project's editable install still points at an older "
            "location of this repository. Rebuild the environment with\n"
            "    uv sync --reinstall\n"
            "Otherwise check for a PYTHONPATH entry naming another checkout:\n"
            "    uv run python -c \"import os,sys; print(os.environ.get('PYTHONPATH')); "
            "print([p for p in sys.path if 'pyroforge' in p])\""
        )


_assert_app_is_local()
