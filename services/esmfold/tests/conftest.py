"""Import server.py and folding.py as top-level modules, as the container does
(`uvicorn server:app` from /app)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
