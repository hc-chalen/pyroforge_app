"""The registry engine's contract: never answer for a database it does not have.

The regression these cover is the one that shipped: a three-sequence placeholder
was written on first use, and every query was answered against it while the UI
said "iGEM Parts Registry". A real protein came back "No matching parts found",
which reads as a fact about the Registry rather than about this server.
"""

from __future__ import annotations

import subprocess

import pytest

from app.engines.registry_engine import MMseqsService, PartsDatabase, RegistryEngine

GFP = (
    "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDH"
    "MKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMA"
    "DKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITH"
    "GMDELYK"
)
LYSOZYME = (
    "MKALIVLGLVLLSVTVQGKVFERCELARTLKRLGMDGYRGISLANWMCLAKWESGYNTRATNYNAGDRSTDYGIFQI"
    "NSRYWCNDGKTPGAVNACHLSCSALLQDNIADAVACAKRVVRDPQGIRAWVAWRNRCQNRDVRQYVQGCGV"
)


@pytest.fixture
def db(tmp_path) -> PartsDatabase:
    return PartsDatabase(tmp_path / "igem_parts.fasta")


def test_absent_database_is_reported_not_invented(db) -> None:
    engine = RegistryEngine(db)
    result = engine.search(LYSOZYME)

    assert result["database"]["status"] == "not_provisioned"
    assert result["database"]["n_parts"] == 0
    assert result["unavailable_reason"] == "no_database"
    # The key property: an unrunnable search is not reported as a search that
    # found nothing.
    assert result["matches"] == []
    # What the user is told must not blame their input, and must not be a shell
    # command they cannot run — that stays in admin_hint.
    assert "Nothing is wrong with your sequence" in result["unavailable_detail"]
    assert "pyroforge-fetch-parts" not in result["database"]["detail"]
    assert "pyroforge-fetch-parts" in result["database"]["admin_hint"]


def test_search_reports_what_it_searched(db) -> None:
    db.write([("BBa_K176000", "GFP", GFP)], source="unit test")
    result = RegistryEngine(db).search(LYSOZYME)

    assert result["unavailable_reason"] is None
    assert result["database"]["status"] == "ready"
    assert result["database"]["n_parts"] == 1
    assert result["database"]["source"] == "unit test"
    assert result["database"]["built_at"]


def test_self_hit_carries_description_and_url_without_network(db) -> None:
    db.write([("BBa_K176000", "GFP reporter", GFP)], source="unit test")
    result = RegistryEngine(db).search(GFP)

    (match,) = result["matches"]
    assert match["part_id"] == "BBa_K176000"
    assert match["identity"] > 99.0
    # Descriptions come from the snapshot. The old per-hit call to
    # parts.igem.org returned 403 behind a CDN — an empty description after up
    # to 15 s on the queue that also serves predictions.
    assert match["description"] == "GFP reporter"
    assert match["biobrick_url"].endswith("Part:BBa_K176000")


def test_fallback_reports_no_evalue_rather_than_a_perfect_one(db) -> None:
    db.write([("BBa_K176000", "GFP", GFP)], source="unit test")
    hits = MMseqsService(db)._fallback_search(GFP, max_hits=5)

    assert hits[0]["search_method"] == "pairwise_blosum62_fallback"
    # 0.0 is the strongest possible E-value; there is none computable here.
    assert hits[0]["evalue"] is None


def test_unrelated_sequence_is_not_reported_as_a_hit(db) -> None:
    db.write([("BBa_K176000", "GFP", GFP)], source="unit test")
    hits = MMseqsService(db)._fallback_search("W" * 120, max_hits=5)
    assert hits == []


def test_mmseqs_failure_degrades_instead_of_failing_the_job(db, monkeypatch) -> None:
    """A mismatched mmseqs build dies with SIGILL on its first AVX2 instruction.

    That used to surface as a failed Celery job carrying a CalledProcessError.
    """
    db.write([("BBa_K176000", "GFP", GFP)], source="unit test")
    svc = MMseqsService(db)
    monkeypatch.setattr(svc, "_mmseqs_available", lambda: True)
    monkeypatch.setattr(
        svc,
        "_mmseqs_search",
        lambda *a, **k: (_ for _ in ()).throw(
            subprocess.CalledProcessError(-4, "mmseqs")
        ),
    )

    hits, unavailable = svc.search(GFP, max_hits=5)
    assert unavailable is None
    assert hits and hits[0]["search_method"] == "pairwise_blosum62_fallback"


@pytest.mark.parametrize(("raw", "expected"), [(0.92, 92.0), (92.0, 92.0), (1.0, 100.0)])
def test_mmseqs_identity_is_normalized_to_a_percentage(raw, expected) -> None:
    """mmseqs reports pident as a 0-1 fraction on some builds and 0-100 on others.

    The UI renders it with a % sign, so an un-normalized fraction displays a 92%
    match as "0.9%".
    """
    assert MMseqsService._normalize_identity(raw) == expected


def test_pairwise_fallback_refuses_a_database_it_cannot_scan(db, monkeypatch) -> None:
    """The fallback is O(database) full alignments — it cannot stand in for mmseqs
    on a real snapshot without parking the single-slot worker for minutes."""
    db.write(
        [(f"BBa_{i:06d}", "part", GFP) for i in range(3)], source="unit test"
    )
    monkeypatch.setattr(MMseqsService, "FALLBACK_MAX_PARTS", 2)
    svc = MMseqsService(db)
    monkeypatch.setattr(svc, "_mmseqs_available", lambda: False)

    hits, unavailable = svc.search(GFP, max_hits=5)
    assert hits == []
    assert unavailable == "mmseqs_required"
