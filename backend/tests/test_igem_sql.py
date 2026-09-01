"""Reading the Registry's `parts` table out of a mysqldump.

The bulk FASTA endpoint (`parts.igem.org/fasta/parts/All_Parts`) now answers 403
from iGEM's own Apache, so the SQL dump is the route that actually builds a
snapshot. These cover the properties that decide whether the snapshot is real:
the sequence column is found by name rather than by position, the dump's own
escaping survives, and — the regression that shipped zero parts — lower-case DNA
is recognised as DNA.
"""

from __future__ import annotations

import gzip

import pytest

from app.data.igem_parts import to_protein_records
from app.data.igem_sql import column_index, iter_parts, parse_value_tuples

# A miniature dump with the real column *names* in a deliberately different order
# and count from the live table. If anything reads columns positionally instead of
# by name, these tests fail.
SCHEMA = """DROP TABLE IF EXISTS `parts`;
CREATE TABLE `parts` (
  `part_id` int NOT NULL AUTO_INCREMENT,
  `part_name` varchar(255) DEFAULT NULL,
  `sequence` longtext NOT NULL,
  `short_desc` varchar(100) DEFAULT NULL,
  `part_type` varchar(20) DEFAULT NULL,
  `status` varchar(20) DEFAULT NULL,
  `discontinued` int DEFAULT '0',
  PRIMARY KEY (`part_id`),
  KEY `parts_name_ix` (`part_name`)
) ENGINE=InnoDB AUTO_INCREMENT=99 DEFAULT CHARSET=latin1;
"""

# A 40-codon ORF: ATG, 38 sense codons, TAA. Long enough to clear the 30 aa floor
# in to_protein_records.
ORF = "atg" + "gcg" * 38 + "taa"


def _dump(rows: list[str]) -> str:
    return SCHEMA + "INSERT INTO `parts` VALUES " + ",".join(rows) + ";\n"


def _row(
    part_id: int,
    name: str,
    sequence: str,
    desc: str = "a part",
    part_type: str = "Coding",
    status: str = "Available",
    discontinued: int = 0,
) -> str:
    return (
        f"({part_id},'{name}','{sequence}','{desc}','{part_type}',"
        f"'{status}',{discontinued})"
    )


def _write(tmp_path, text: str, gz: bool = False):
    if gz:
        path = tmp_path / "parts.sql.gz"
        path.write_bytes(gzip.compress(text.encode("latin-1")))
        return path
    path = tmp_path / "parts.sql"
    path.write_text(text, encoding="latin-1")
    return path


def test_columns_are_located_by_name_not_position() -> None:
    cols = column_index(SCHEMA)
    assert cols["part_name"] == 1
    assert cols["sequence"] == 2
    # The KEY / PRIMARY KEY block is not a column.
    assert "parts_name_ix" not in cols
    assert len(cols) == 7


def test_lowercase_dna_survives_to_a_protein(tmp_path) -> None:
    """The regression that made the whole dump yield zero parts.

    The Registry stores sequences lower-case. `looks_like_dna` tests membership in
    upper-case "ACGTUN", so un-cased input is not recognised as DNA, skips
    translation, and is then stripped to nothing by the amino-acid filter — 75,672
    parts in, 0 out, with the only symptom being a `too_short` counter.
    """
    path = _write(tmp_path, _dump([_row(1, "BBa_K000001", ORF)]))
    records, stats = iter_parts(path)

    assert stats["kept"] == 1
    assert records[0][2].isupper()

    proteins, pstats = to_protein_records(records)
    assert pstats["translated"] == 1
    assert pstats["too_short"] == 0
    assert proteins[0][2].startswith("MA")


def test_part_name_is_the_id_so_urls_resolve(tmp_path) -> None:
    """`RegistryEngine.part_url` builds parts.igem.org/Part:<id> from this field,
    so the numeric primary key would produce a dead link on every hit."""
    path = _write(tmp_path, _dump([_row(31213, "BBa_K1149051", ORF)]))
    records, _ = iter_parts(path)

    assert records[0][0] == "BBa_K1149051"


def test_deleted_and_discontinued_parts_are_dropped(tmp_path) -> None:
    path = _write(
        tmp_path,
        _dump([
            _row(1, "BBa_LIVE", ORF),
            _row(2, "BBa_DEAD", ORF, status="Deleted"),
            _row(3, "BBa_GONE", ORF, discontinued=1),
        ]),
    )
    records, stats = iter_parts(path)

    assert [r[0] for r in records] == ["BBa_LIVE"]
    assert stats["deleted"] == 2


def test_empty_sequence_is_counted_not_translated(tmp_path) -> None:
    path = _write(tmp_path, _dump([_row(1, "BBa_EMPTY", ""), _row(2, "BBa_OK", ORF)]))
    records, stats = iter_parts(path)

    assert stats["no_sequence"] == 1
    assert [r[0] for r in records] == ["BBa_OK"]


@pytest.mark.parametrize(
    ("literal", "expected"),
    [
        (r"It\'s complicated", "It's complicated"),
        (r"back\\slash", "back\\slash"),
        (r"line\nbreak", "line break"),  # collapsed by the header sanitiser
        (r"tab\tsep", "tab sep"),
    ],
)
def test_dump_escaping_is_decoded(tmp_path, literal: str, expected: str) -> None:
    path = _write(tmp_path, _dump([_row(1, "BBa_X", ORF, desc=literal)]))
    records, _ = iter_parts(path)

    assert records[0][1] == f"[Coding] {expected}"


def test_description_never_contains_whitespace_that_breaks_the_fasta(tmp_path) -> None:
    """`PartsDatabase.write` puts the description on the header line and
    `_parse_descriptions` splits on whitespace, so an embedded newline would
    corrupt the FASTA into unparseable records."""
    path = _write(tmp_path, _dump([_row(1, "BBa_X", ORF, desc=r"two\nlines\r\nhere")]))
    records, _ = iter_parts(path)

    assert "\n" not in records[0][1]
    assert "\r" not in records[0][1]


def test_extended_insert_yields_every_tuple() -> None:
    line = (
        "INSERT INTO `parts` VALUES (1,'BBa_A','atg','d','Coding','Available',0),"
        "(2,'BBa_B','atg','d','Coding','Available',0);"
    )
    tuples = list(parse_value_tuples(line))

    assert len(tuples) == 2
    assert tuples[0][1] == "BBa_A"
    assert tuples[1][1] == "BBa_B"


def test_null_is_distinguished_from_empty_string() -> None:
    line = "INSERT INTO `parts` VALUES (1,NULL,'',NULL,'Coding','Available',0);"
    values = next(parse_value_tuples(line))

    assert values[1] is None
    assert values[2] == ""


def test_gzip_with_trailing_garbage_is_read_in_full(tmp_path) -> None:
    """The file the Registry serves has bytes appended after the gzip stream ends.
    `gzip.open` reads them as a second member's header and raises BadGzipFile
    partway through — after most rows, so it looks like a corrupt row."""
    path = tmp_path / "parts.sql.gz"
    path.write_bytes(
        gzip.compress(_dump([_row(1, "BBa_A", ORF), _row(2, "BBa_B", ORF)]).encode())
        + b"\n"
    )
    records, stats = iter_parts(path)

    assert stats["rows"] == 2
    assert [r[0] for r in records] == ["BBa_A", "BBa_B"]


def test_a_dump_without_the_parts_table_fails_loudly(tmp_path) -> None:
    path = _write(tmp_path, "CREATE TABLE `other` (\n  `x` int\n) ENGINE=InnoDB;\n")

    with pytest.raises(ValueError, match="does not look like a Registry parts dump"):
        iter_parts(path)
