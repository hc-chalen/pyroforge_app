#!/usr/bin/env python3
"""Thin shim — the implementation lives in ``app.data.igem_parts`` so that it
ships inside the installed package (only ``app`` is packaged; see pyproject).

Prefer the console script: ``uv run pyroforge-fetch-parts``.
"""

from app.data.igem_parts import main

if __name__ == "__main__":
    raise SystemExit(main())
