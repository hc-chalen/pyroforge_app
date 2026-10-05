"""What the result cache must not get wrong.

It exists because a room of testers on one protein made the single heavy worker
recompute the same 134 s optimization scan once per person. Caching a *model's*
output is only safe while the answer really is a function of the key, so the
checkpoint fingerprint is as much a part of the contract as the round trip.
"""

from __future__ import annotations

import time

import pytest

from app.core.config import settings
from app.services.result_cache import ResultCache

KEY = {"sequence": "MKTAYIAK", "max_mutations": 5, "perplexity_threshold": 0.05}
RESULT = {"ddg": 1.62, "mutations": [{"position": 65, "target_aa": "I"}]}


@pytest.fixture
def cache(monkeypatch, tmp_path) -> ResultCache:
    monkeypatch.setattr(settings, "cache_dir", tmp_path / "cache")
    monkeypatch.setattr(settings, "checkpoint_dir", tmp_path / "checkpoints")
    monkeypatch.setattr(settings, "result_cache_enabled", True)
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "model_thermo.pt").write_text("weights")
    return ResultCache("optimizations")


def test_a_stored_result_comes_back(cache) -> None:
    assert cache.get(KEY) is None
    cache.put(KEY, RESULT)

    assert cache.get(KEY) == RESULT


def test_different_parameters_are_a_different_result(cache) -> None:
    cache.put(KEY, RESULT)

    assert cache.get({**KEY, "max_mutations": 6}) is None
    assert cache.get({**KEY, "sequence": "MKTAYIAKQ"}) is None


def test_retraining_invalidates_everything(cache, tmp_path) -> None:
    """A new checkpoint must not serve the old model's answer under its name."""
    cache.put(KEY, RESULT)
    assert cache.get(KEY) == RESULT

    (tmp_path / "checkpoints" / "model_thermo.pt").write_text("retrained, different size")

    assert cache.get(KEY) is None


def test_plots_do_not_invalidate_anything(cache, tmp_path) -> None:
    """Training writes figures next to the checkpoints; they change no prediction."""
    cache.put(KEY, RESULT)
    (tmp_path / "checkpoints" / "plots").mkdir()
    (tmp_path / "checkpoints" / "plots" / "loss.png").write_text("not a checkpoint")

    assert cache.get(KEY) == RESULT


def test_disabled_cache_neither_reads_nor_writes(cache, monkeypatch) -> None:
    cache.put(KEY, RESULT)
    monkeypatch.setattr(settings, "result_cache_enabled", False)

    assert cache.get(KEY) is None

    cache.put({**KEY, "sequence": "NEWSEQ"}, RESULT)
    monkeypatch.setattr(settings, "result_cache_enabled", True)
    assert cache.get({**KEY, "sequence": "NEWSEQ"}) is None


def test_a_truncated_entry_is_a_miss_not_a_crash(cache) -> None:
    cache.put(KEY, RESULT)
    entry = next(cache.directory.glob("*.json"))
    entry.write_text('{"ddg": 1.6')

    assert cache.get(KEY) is None


def test_missing_checkpoint_directory_is_survivable(monkeypatch, tmp_path) -> None:
    """A serving install with no checkpoints yet must not fail on every lookup."""
    monkeypatch.setattr(settings, "cache_dir", tmp_path / "cache")
    monkeypatch.setattr(settings, "checkpoint_dir", tmp_path / "absent")
    monkeypatch.setattr(settings, "result_cache_enabled", True)
    cache = ResultCache("optimizations")

    cache.put(KEY, RESULT)
    assert cache.get(KEY) == RESULT


def test_a_herd_on_one_key_computes_once(cache) -> None:
    """Thirty testers pressing the same button must not start thirty computations."""
    import threading

    calls: list[int] = []
    started = threading.Barrier(8)

    def compute() -> dict:
        calls.append(1)
        time.sleep(0.05)
        return RESULT

    def worker(results: list) -> None:
        started.wait()
        results.append(cache.get_or_compute(KEY, compute))

    results: list = []
    threads = [threading.Thread(target=worker, args=(results,)) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(calls) == 1
    assert results == [RESULT] * 8
