"""Disk cache for deterministic, expensive engine results.

A class UAT is thirty people putting the *same* protein through the same cards.
`optimize_sequence` is a site-saturation scan — 134 s measured on an 84-mer — and
it is deterministic: same sequence, same parameters, same checkpoints, same
answer. Thirty runs of it is the single `heavy` worker spending 67 minutes
recomputing one result while everyone waits in line.

Keys cover the inputs *and* a fingerprint of the served checkpoints, so
retraining invalidates every entry instead of serving yesterday's answer under
today's model. Writes are atomic, because more than one worker can finish the
same computation at once.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import uuid
from pathlib import Path
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)

# Only the files the served models are actually loaded from. Globbing the whole
# checkpoint directory would also stat the training plots, which change without
# changing a single prediction.
_CHECKPOINT_PATTERNS = ("*.pt", "*.joblib", "*.json")


def checkpoint_fingerprint() -> str:
    """Size and mtime of every served checkpoint, hashed."""
    directory = settings.checkpoint_dir
    parts: list[str] = []
    if directory.exists():
        paths: list[Path] = []
        for pattern in _CHECKPOINT_PATTERNS:
            paths.extend(directory.rglob(pattern))
        for path in sorted(paths):
            try:
                stat = path.stat()
            except OSError:
                continue
            parts.append(f"{path.relative_to(directory)}:{stat.st_size}:{int(stat.st_mtime)}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


class ResultCache:
    """Keyed JSON results under ``cache_dir/<name>``."""

    def __init__(self, name: str) -> None:
        self.directory = settings.cache_dir / name
        # One lock per key, so a thundering herd on the same input computes once.
        # Bounded in practice by the number of distinct requests a process sees;
        # each entry is an empty Lock, so the footprint is negligible.
        self._key_locks: dict[str, threading.Lock] = {}
        self._key_locks_guard = threading.Lock()

    def _path(self, key: dict[str, Any]) -> Path:
        material = json.dumps(
            {**key, "_checkpoints": checkpoint_fingerprint()},
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(material.encode()).hexdigest()[:24]
        return self.directory / f"{digest}.json"

    def get(self, key: dict[str, Any]) -> dict | None:
        if not settings.result_cache_enabled:
            return None
        path = self._path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            # A truncated file from an interrupted write is a miss, not a failure.
            logger.warning("Discarding unreadable cache entry %s", path)
            return None

    def put(self, key: dict[str, Any], value: dict) -> None:
        if not settings.result_cache_enabled:
            return
        path = self._path(key)
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            # Atomic, and unique per writer: another worker — or another thread in
            # this process, which is how the API serves its synchronous routes —
            # may be writing the same key right now, and a reader must never see
            # half a file. A name keyed on the pid alone let two threads share one
            # temp file, so whichever renamed second failed on a file already gone.
            temporary = path.with_name(f"{path.stem}.{uuid.uuid4().hex}.tmp")
            temporary.write_text(json.dumps(value))
            temporary.replace(path)
        except (OSError, TypeError, ValueError):
            logger.warning("Could not cache result at %s", path, exc_info=True)


    def get_or_compute(self, key: dict[str, Any], compute) -> dict:
        """Return the cached result, computing it once if several callers race.

        A room told to press the same button arrives together, so every caller
        misses the empty cache and starts the same 19 s computation. The second
        check — after the lock, when the winner has already written — is what turns
        thirty identical computations into one plus twenty-nine reads.

        Single-flight within this process only. Another worker process computing
        the same key in parallel is wasteful but correct.
        """
        cached = self.get(key)
        if cached is not None:
            return cached

        with self._lock_for(key):
            cached = self.get(key)
            if cached is not None:
                return cached
            value = compute()
            self.put(key, value)
            return value

    def _lock_for(self, key: dict[str, Any]) -> threading.Lock:
        name = self._path(key).name
        with self._key_locks_guard:
            return self._key_locks.setdefault(name, threading.Lock())


_caches: dict[str, ResultCache] = {}


def get_result_cache(name: str) -> ResultCache:
    if name not in _caches:
        _caches[name] = ResultCache(name)
    return _caches[name]
