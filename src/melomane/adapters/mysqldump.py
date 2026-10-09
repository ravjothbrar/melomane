"""Streaming parser for `mysqldump` files (extended INSERTs).  Pure Python, constant memory, Windows-safe.

Handles: .sql / .sql.gz / .zip (first .sql member); quoted strings with backslash escapes and '' escapes; NULL; ints/floats;
parentheses and commas inside strings; very long INSERT lines (mysqldump wraps at ~1 MB).
Encoding: files are decoded as UTF-8 with replacement; `fix_mojibake` repairs the common latin1-table/utf8-client double-encoding.
"""
from __future__ import annotations
import gzip, io, re, zipfile
from pathlib import Path
from typing import Iterator, Callable

_INSERT = re.compile(r"^INSERT INTO `([^`]+)` (?:\(([^)]*)\) )?VALUES ")
_TUPLE = re.compile(r"\((?:[^()']|'(?:[^'\\]|\\.)*')*\)", re.S)
_VALUE = re.compile(r"'((?:[^'\\]|\\.)*)'|(NULL)|(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", re.S)
_ESC = re.compile(r"\\(.)", re.S)
_ESC_MAP = {"n": "\n", "r": "\r", "t": "\t", "0": "\0", "b": "\b", "Z": "\x1a", "'": "'", '"': '"', "\\": "\\"}
_CREATE = re.compile(r"^CREATE TABLE `([^`]+)` \(")
_COL = re.compile(r"^\s+`([^`]+)` ")


def unescape(s: str) -> str:
    return _ESC.sub(lambda m: _ESC_MAP.get(m.group(1), m.group(1)), s) if "\\" in s else s


def fix_mojibake(s: str) -> str:
    """Repair UTF-8 that was stored/dumped through latin1 ('CafÃ©' -> 'Café').  Leaves clean text untouched."""
    if not s or s.isascii():
        return s
    if "Ã" in s or "â€" in s or "Â" in s:
        try:
            return s.encode("cp1252", errors="strict").decode("utf-8")
        except UnicodeError:
            try:
                return s.encode("latin-1", errors="strict").decode("utf-8")
            except UnicodeError:
                return s
    return s


def open_text(path: Path) -> io.TextIOBase:
    path = Path(path)
    if path.suffix == ".gz":
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", errors="replace", newline="\n")
    if path.suffix == ".zip":
        z = zipfile.ZipFile(path)
        name = next(n for n in z.namelist() if n.lower().endswith(".sql"))
        return io.TextIOWrapper(z.open(name), encoding="utf-8", errors="replace", newline="\n")
    return open(path, "r", encoding="utf-8", errors="replace", newline="\n", buffering=1 << 24)


def parse_row(tup: str) -> list:
    """'(1,'a',NULL)' -> [1, 'a', None]  (strings unescaped, ints/floats converted)."""
    out = []
    for m in _VALUE.finditer(tup, 1, len(tup) - 1):
        s, null, num = m.groups()
        if s is not None:
            out.append(unescape(s))
        elif null is not None:
            out.append(None)
        else:
            out.append(float(num) if ("." in num or "e" in num.lower()) else int(num))
    return out


def iter_statements(path: Path) -> Iterator[tuple[str, str, str | None]]:
    """Yield (kind, table, payload): ('create', table, ddl_text) and ('insert', table, values_text) in file order."""
    ddl_table, ddl = None, []
    with open_text(path) as f:
        for line in f:
            if ddl_table is not None:
                ddl.append(line)
                if line.startswith(")"):
                    yield "create", ddl_table, "".join(ddl)
                    ddl_table, ddl = None, []
                continue
            if line.startswith("INSERT INTO"):
                m = _INSERT.match(line)
                if m:
                    yield "insert", m.group(1), line[m.end():]
            elif line.startswith("CREATE TABLE"):
                m = _CREATE.match(line)
                if m:
                    ddl_table, ddl = m.group(1), [line]


def table_columns(ddl: str) -> list[str]:
    cols = []
    for ln in ddl.splitlines()[1:]:
        m = _COL.match(ln)
        if m:
            cols.append(m.group(1))
    return cols


def iter_rows(values_text: str) -> Iterator[list]:
    for m in _TUPLE.finditer(values_text):
        yield parse_row(m.group(0))


def scan(path: Path, handlers: dict[str, Callable[[list[list], list[str]], None]], batch_rows: int = 200_000,
         progress: Callable[[str, int], None] | None = None) -> dict[str, int]:
    """One pass over the dump.  handlers[table](rows, columns) is called with batches of parsed rows (columns from the
    CREATE TABLE in the same file when present).  Tables without a handler are skipped without parsing."""
    cols: dict[str, list[str]] = {}
    counts: dict[str, int] = {}
    for kind, table, payload in iter_statements(path):
        if kind == "create":
            cols[table] = table_columns(payload)
            continue
        h = handlers.get(table)
        if h is None:
            continue
        rows = list(iter_rows(payload))
        if rows:
            h(rows, cols.get(table, []))
            counts[table] = counts.get(table, 0) + len(rows)
            if progress:
                progress(table, counts[table])
    return counts
