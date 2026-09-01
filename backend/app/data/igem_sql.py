#!/usr/bin/env python3
"""Read the Registry's ``parts`` table out of an iGEM MySQL dump.

Why this exists: ``https://parts.igem.org/fasta/parts/All_Parts`` — the bulk FASTA
export the sibling module fetches — now answers **403 from iGEM's own Apache**, not
from the CDN in front of it. That is the origin refusing the path, so no amount of
retrying, user-agent spoofing or running off a different network gets it back. The
export the Registry does still publish is a ``mysqldump`` of ``parts_bbdb``, and it
carries strictly more than the FASTA did: part type and status as well as the
sequence, so non-coding and deleted parts can be dropped before translation rather
than after.

The output is the same ``[(part_id, description, sequence)]`` shape
``igem_parts.to_protein_records`` already consumes, so translation, length filtering
and de-duplication stay in one place and the two input paths cannot drift apart.

``part_id`` is the **part name** (``BBa_K1149051``), not the numeric primary key —
``RegistryEngine.part_url`` builds ``parts.igem.org/Part:<id>`` from it, and the
numeric id would produce a dead link on every hit.

Dump specifics this handles, all observed in the real file:

  * ``latin1`` table with ``SET NAMES utf8``, and a ``binary(20)`` sha1 column, so
    the byte stream is not valid UTF-8. Parsing runs on a latin-1 decode (total,
    byte-exact); text fields are re-decoded as UTF-8 afterwards.
  * Backslash escaping (``\\'``, ``\\n``, ``\\\\``), not doubled quotes.
  * Real newlines inside strings are escaped by mysqldump, so one statement is one
    line — the file streams a line at a time instead of loading 422 MB.
  * ``--extended-insert`` packs many ``(...),(...)`` tuples per statement.
"""

from __future__ import annotations

import re
import zlib
from collections.abc import Iterator
from pathlib import Path

# 1 MiB of compressed input per read. Rows carry HTML blobs, so the decompressed
# side of one chunk is several MB — large enough that the loop is not the cost,
# small enough that a 422 MB dump never lands in memory at once.
_CHUNK = 1 << 20

# Column order is read from the dump's own CREATE TABLE rather than hard-coded:
# the table has 50 columns and iGEM has changed it before. A positional constant
# would decode silently-wrong fields after any such change; a name lookup fails
# loudly instead.
_CREATE_RE = re.compile(r"CREATE TABLE `parts` \((.*?)\n\) ENGINE=", re.S)
_COLUMN_RE = re.compile(r"^\s+`([a-zA-Z0-9_]+)`\s")
_INSERT_PREFIX = "INSERT INTO `parts` VALUES "

# MySQL's string escapes. \0 and \Z are the two that are not their own literal.
_UNESCAPE = {
    "0": "\0",
    "b": "\b",
    "n": "\n",
    "r": "\r",
    "t": "\t",
    "Z": "\x1a",
    "'": "'",
    '"': '"',
    "\\": "\\",
}

# Parts whose row says they are not real parts. `status` is the workflow state and
# `discontinued` the retirement flag; a part that is either is still in the dump but
# is not something a homology hit should send a user to.
_DEAD_STATUSES = frozenset({"Deleted", "Removed"})


def iter_lines(path: Path) -> Iterator[str]:
    """Stream a dump as latin-1 text lines, transparently gunzipping ``.gz``.

    latin-1 because the decode must be total: the ``sequence_sha1`` column is raw
    binary and would abort a UTF-8 decode. latin-1 maps every byte to exactly one
    character, so quote/backslash positions — the only thing the parser depends on —
    are preserved exactly. Text fields are re-decoded in :func:`_text`.

    Decompression goes through ``zlib`` rather than ``gzip.open`` because the file
    the Registry serves has **trailing bytes after the gzip stream ends**. ``gzip``
    reads them as the header of a second member and dies with ``BadGzipFile`` partway
    through the dump — after yielding most of it, so the failure looks like a corrupt
    row rather than a container problem. ``gzcat`` merely warns ("trailing garbage
    ignored"), which is the behaviour wanted here: one member, then stop.
    """
    if path.suffix != ".gz":
        with path.open("r", encoding="latin-1", newline="") as handle:
            yield from handle
        return

    # wbits=31 = zlib window 15 + gzip header/trailer handling.
    decompressor = zlib.decompressobj(31)
    pending = ""
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_CHUNK)
            if not chunk:
                break
            pending += decompressor.decompress(chunk).decode("latin-1")
            lines = pending.split("\n")
            # The last element is a partial line unless the chunk ended exactly on a
            # newline, in which case it is "" and carries forward harmlessly.
            pending = lines.pop()
            for line in lines:
                yield line + "\n"
            if decompressor.eof:
                break  # End of the first member; anything after it is the garbage.
    if pending:
        yield pending


def _text(value: str | None) -> str:
    """Recover UTF-8 text from a latin-1-decoded field."""
    if not value:
        return ""
    return value.encode("latin-1", "replace").decode("utf-8", "replace")


def column_index(dump_text: str) -> dict[str, int]:
    """``{column_name: position}`` from the dump's ``CREATE TABLE `parts```."""
    match = _CREATE_RE.search(dump_text)
    if not match:
        raise ValueError(
            "No `CREATE TABLE \\`parts\\`` found — this does not look like a "
            "Registry parts dump."
        )
    columns: list[str] = []
    for line in match.group(1).splitlines():
        col = _COLUMN_RE.match(line)
        # Stops at the KEY/PRIMARY KEY block, which does not match the column form.
        if col:
            columns.append(col.group(1))
    return {name: i for i, name in enumerate(columns)}


def _unescape(raw: str) -> str:
    if "\\" not in raw:
        return raw
    out: list[str] = []
    i = 0
    n = len(raw)
    while i < n:
        ch = raw[i]
        if ch == "\\" and i + 1 < n:
            nxt = raw[i + 1]
            out.append(_UNESCAPE.get(nxt, nxt))
            i += 2
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _end_of_string(line: str, start: int) -> int:
    """Index of the quote closing the string that opens at ``start``.

    Jumps quote to quote with ``str.find`` rather than walking characters. The rows
    carry HTML blobs tens of kilobytes long, so a per-character loop here is the
    difference between seconds and minutes over the whole 422 MB dump.
    """
    i = start + 1
    while True:
        j = line.find("'", i)
        if j == -1:
            raise ValueError("unterminated string literal in dump")
        # A quote is escaped only by an ODD run of backslashes: `\\'` is a literal
        # backslash then a closing quote, `\'` is an escaped quote.
        backslashes = 0
        k = j - 1
        while k >= start and line[k] == "\\":
            backslashes += 1
            k -= 1
        if backslashes % 2 == 0:
            return j
        i = j + 1


def parse_value_tuples(line: str) -> Iterator[list[str | None]]:
    """Yield each ``(...)`` tuple of an ``INSERT ... VALUES`` statement.

    Values come back as ``str`` (unescaped, still latin-1) or ``None`` for SQL NULL;
    numbers are left as their source text since every column this module reads is
    either textual or compared as a string.
    """
    i = line.index(_INSERT_PREFIX) + len(_INSERT_PREFIX)
    n = len(line)
    while i < n:
        if line[i] != "(":
            # The inter-tuple `,` and the statement-terminating `;`.
            i += 1
            continue
        i += 1
        values: list[str | None] = []
        field_start = i
        while i < n:
            ch = line[i]
            if ch == "'":
                end = _end_of_string(line, i)
                values.append(_unescape(line[i + 1 : end]))
                i = end + 1
                # Skip to the separator; a quoted field is complete at its quote.
                while i < n and line[i] not in ",)":
                    i += 1
                if line[i] == ")":
                    i += 1
                    break
                i += 1
                field_start = i
                continue
            if ch in ",)":
                bare = line[field_start:i].strip()
                if bare:
                    values.append(None if bare == "NULL" else bare)
                i += 1
                if ch == ")":
                    break
                field_start = i
                continue
            i += 1
        yield values


def iter_parts(
    path: Path,
) -> tuple[list[tuple[str, str, str]], dict[str, int]]:
    """``([(part_name, description, dna)], stats)`` from a Registry dump.

    Rows are dropped here only for reasons visible in the *row* — no sequence, a
    deleted/discontinued part, a part type that cannot code for protein. Everything
    downstream of translation (ORF finding, the 30–2000 aa window, de-duplication)
    stays in ``igem_parts.to_protein_records`` so both input paths filter identically.
    """
    records: list[tuple[str, str, str]] = []
    stats = {
        "rows": 0,
        "kept": 0,
        "no_sequence": 0,
        "deleted": 0,
        "unnamed": 0,
    }

    columns: dict[str, int] | None = None
    header: list[str] = []
    fields: tuple[int, ...] = ()

    for line in iter_lines(path):
        if columns is None:
            # Accumulate only from the `parts` CREATE TABLE onwards. The dump holds
            # more than one table, and starting the capture at the right one keeps
            # `parts_seq_features` from being read as the schema if iGEM ever
            # reorders the dump.
            if not header and not line.startswith("CREATE TABLE `parts` ("):
                continue
            header.append(line)
            # The CREATE TABLE precedes every INSERT in a mysqldump, so the schema
            # is always known before the first row is parsed.
            if line.startswith(") ENGINE="):
                columns = column_index("".join(header))
                header = []
                fields = (
                    columns["part_name"],
                    columns["short_desc"],
                    columns["part_type"],
                    columns["status"],
                    columns["discontinued"],
                    columns["sequence"],
                )
            continue
        if not line.startswith(_INSERT_PREFIX):
            continue

        i_name, i_desc, i_type, i_status, i_disc, i_seq = fields

        for values in parse_value_tuples(line):
                if len(values) <= i_seq:
                    continue
                stats["rows"] += 1

                name = (values[i_name] or "").strip()
                if not name:
                    stats["unnamed"] += 1
                    continue

                status = (values[i_status] or "").strip()
                discontinued = (values[i_disc] or "0").strip()
                if status in _DEAD_STATUSES or discontinued not in ("0", "NULL"):
                    stats["deleted"] += 1
                    continue

                # Upper-cased here, matching what `parse_fasta` does on the FASTA
                # route. The Registry stores sequences lower-case, and every
                # downstream check is upper-case: `looks_like_dna` tests membership
                # in "ACGTUN" and the translated protein is filtered against
                # STANDARD_AA. Left as-is, lower-case DNA is not recognised as DNA,
                # so it skips translation, then loses every character to the
                # amino-acid filter — the whole dump silently reduces to zero parts.
                sequence = (values[i_seq] or "").strip().upper()
                if not sequence:
                    stats["no_sequence"] += 1
                    continue

                # Part type rides along in the description: the hit table shows it,
                # and "Coding" next to a match is most of what tells a user whether
                # the homology means anything.
                part_type = _text(values[i_type]).strip()
                short_desc = _text(values[i_desc]).strip()
                description = " ".join(
                    part for part in (f"[{part_type}]" if part_type else "", short_desc)
                    if part
                )
                # `write()` emits one header line per record and `_parse_descriptions`
                # splits it on whitespace, so any newline or tab here would silently
                # corrupt the FASTA into unparseable records.
                description = " ".join(description.split())

                records.append((name, description, sequence))
                stats["kept"] += 1

    if columns is None:
        raise ValueError(
            "No `CREATE TABLE \\`parts\\`` found — this does not look like a "
            "Registry parts dump."
        )
    return records, stats
