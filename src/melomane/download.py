"""Resumable HTTP downloader (httpx) with progress logging, plus a tiny Zenodo client.

Design (all Windows-safe: pathlib only, file handles are always closed before hashing/renaming, the final rename retries
on transient ``PermissionError`` caused by virus scanners / indexers):

* data is streamed into ``<dest><ext>.part`` and only renamed to ``dest`` once size/md5 verify -> a file at ``dest`` is
  always complete, a ``.part`` file is always resumable;
* resume uses ``Range: bytes=<size>-``; servers that ignore ``Range`` (HTTP 200) make us restart from zero, a 416 means
  "already complete" when the range start equals the resource size, otherwise the partial file is discarded;
* transient failures (transport errors, 408/425/429/5xx, short reads) are retried with exponential backoff; the retry
  counter only counts *consecutive failures without progress*, so a flaky link that keeps making progress never gives up;
* on md5/size mismatch the partial file is deleted and the file fetched once more before ``DownloadError`` is raised.

UNVERIFIED (sandbox has no route to zenodo.org): the exact JSON shape of Zenodo's API today.  ``zenodo_record_files``
understands the legacy shape (``files[].key/size/checksum/links.self``) and the InvenioRDM ``entries`` shapes, and falls
back to ``/api/records/<id>/files``; if none match it raises ``DownloadError`` with the top-level keys it saw.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Callable

import httpx

from .runtime import disk_free_gb, get_logger, md5_file, write_json

log = get_logger("melomane.download")

ZENODO_API = "https://zenodo.org/api/records/{id}"
ZENODO_FILES_API = "https://zenodo.org/api/records/{id}/files"
_RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524}
_PROGRESS_EVERY_S = 10.0


class DownloadError(RuntimeError):
    """Non-recoverable download failure (bad status, retries exhausted, persistent checksum mismatch)."""


class _Transient(Exception):
    def __init__(self, msg: str, retry_after: float | None = None, backoff: bool = True):
        super().__init__(msg)
        self.retry_after, self.backoff = retry_after, backoff


def _part_path(dest: Path) -> Path:
    return dest.with_suffix(dest.suffix + ".part")


def _parse_content_range(value: str | None) -> tuple[int | None, int | None, int | None]:
    """'bytes 100-199/1000' -> (100, 199, 1000);  'bytes */1000' -> (None, None, 1000)."""
    if not value:
        return None, None, None
    m = re.match(r"\s*bytes\s+(?:(\d+)-(\d+)|\*)\s*/\s*(\d+|\*)", value, re.I)
    if not m:
        return None, None, None
    start = int(m.group(1)) if m.group(1) is not None else None
    end = int(m.group(2)) if m.group(2) is not None else None
    total = int(m.group(3)) if m.group(3).isdigit() else None
    return start, end, total


def _retry_after(resp: httpx.Response) -> float | None:
    v = resp.headers.get("Retry-After")
    try:
        return float(v) if v is not None else None
    except ValueError:
        return None


def _fmt_mb(n: float) -> str:
    return f"{n / 1e6:,.1f} MB"


def _replace_with_retry(src: Path, dst: Path, tries: int = 6, sleep: Callable[[float], None] = time.sleep) -> None:
    for i in range(tries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == tries - 1:
                raise
            sleep(0.5 * (i + 1))


def _verify(path: Path, expected_md5: str | None, expected_size: int | None) -> str | None:
    """None if OK, else a human-readable reason."""
    size = path.stat().st_size
    if expected_size is not None and size != expected_size:
        return f"size {size} != expected {expected_size}"
    if expected_md5:
        got = md5_file(path)
        if got.lower() != expected_md5.lower():
            return f"md5 {got} != expected {expected_md5.lower()}"
    return None


def _fetch_once(client: httpx.Client, url: str, part: Path, headers: dict, chunk: int,
                expected_size: int | None, label: str) -> bool:
    """One HTTP attempt appending to ``part``.  Returns True when the body is complete (or was already complete)."""
    offset = part.stat().st_size if part.exists() else 0
    if expected_size is not None:
        if offset > expected_size:
            log.warning("%s: partial file larger than expected size, restarting", label)
            part.unlink()
            offset = 0
        elif offset == expected_size:
            return True
    h = dict(headers)
    if offset:
        h["Range"] = f"bytes={offset}-"
    t0 = last = time.monotonic()
    with client.stream("GET", url, headers=h) as r:
        st = r.status_code
        if st == 416:
            _, _, total = _parse_content_range(r.headers.get("Content-Range"))
            if offset and (total == offset or (total is None and expected_size in (None, offset))):
                log.info("%s: server says range not satisfiable at %d bytes -> already complete", label, offset)
                return True
            part.unlink(missing_ok=True)
            raise _Transient("416 with inconsistent size; restarting from zero", backoff=False)
        if st in _RETRY_STATUS:
            raise _Transient(f"HTTP {st}", _retry_after(r))
        if st >= 400 or st < 200:
            raise DownloadError(f"{label}: HTTP {st} for {url}")
        if st == 206:
            start, _, total = _parse_content_range(r.headers.get("Content-Range"))
            if start != offset:
                part.unlink(missing_ok=True)
                raise _Transient(f"206 starts at {start}, wanted {offset}; restarting", backoff=False)
            mode = "ab"
        else:
            if offset:
                log.warning("%s: server ignored Range (HTTP %d); restarting from zero", label, st)
            offset, mode = 0, "wb"
            cl = r.headers.get("Content-Length")
            total = int(cl) if cl and cl.isdigit() else None
        total = total if total is not None else expected_size
        if offset:
            log.info("%s: resuming at %s%s", label, _fmt_mb(offset), f" of {_fmt_mb(total)}" if total else "")
        got = 0
        with open(part, mode) as f:
            for data in r.iter_raw(chunk_size=chunk):
                f.write(data)
                got += len(data)
                now = time.monotonic()
                if now - last >= _PROGRESS_EVERY_S:
                    done = offset + got
                    rate = got / max(now - t0, 1e-9)
                    eta = f", ETA {(total - done) / rate / 60:.1f} min" if total and rate > 0 else ""
                    pct = f" ({100 * done / total:.0f}%)" if total else ""
                    log.info("%s: %s%s at %.1f MB/s%s", label, _fmt_mb(done), pct, rate / 1e6, eta)
                    last = now
    if total is not None and offset + got < total:
        raise _Transient(f"short read: {offset + got} of {total} bytes")
    return True


def download(url: str, dest: Path, expected_md5: str | None = None, expected_size: int | None = None,
             retries: int = 8, chunk: int = 8 << 20, headers: dict[str, str] | None = None,
             transport: httpx.BaseTransport | None = None, sleep: Callable[[float], None] = time.sleep) -> Path:
    """Download ``url`` to ``dest`` (resumable, verified, atomic).  Returns ``dest``.

    * If ``dest`` exists and verifies against the given md5/size, nothing is fetched.  If it exists and *no* expectation
      was given it is trusted (a file at ``dest`` is only ever produced by a completed download).
    * ``transport`` lets tests inject ``httpx.MockTransport``; ``sleep`` lets them skip backoff.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = _part_path(dest)
    label = dest.name
    if dest.exists():
        if expected_md5 is None and expected_size is None:
            log.info("%s: already present, nothing to verify -> skipped", label)
            return dest
        why = _verify(dest, expected_md5, expected_size)
        if why is None:
            log.info("%s: already present and verified -> skipped", label)
            return dest
        log.warning("%s: existing file failed verification (%s); re-downloading", label, why)
        dest.unlink()
    hdrs = {"Accept-Encoding": "identity", **(headers or {})}      # identity: Range offsets refer to the raw bytes
    refetched = False
    failures = 0
    timeout = httpx.Timeout(60.0, connect=30.0)
    with httpx.Client(transport=transport, follow_redirects=True, timeout=timeout) as client:
        while True:
            before = part.stat().st_size if part.exists() else 0
            try:
                _fetch_once(client, url, part, hdrs, chunk, expected_size, label)
            except (_Transient, httpx.TransportError) as e:
                after = part.stat().st_size if part.exists() else 0
                failures = 0 if after > before else failures + 1
                backoff = getattr(e, "backoff", True)
                if failures > retries:
                    raise DownloadError(f"{label}: giving up after {retries} retries without progress: {e}") from e
                delay = min(60.0, 2.0 ** max(failures - 1, 0)) if backoff else 0.0
                ra = getattr(e, "retry_after", None)
                if ra:
                    delay = max(delay, min(ra, 300.0))
                log.warning("%s: %s (%s); retry %d/%d in %.0fs", label, type(e).__name__, e, failures, retries, delay)
                if delay:
                    sleep(delay)
                continue
            why = _verify(part, expected_md5, expected_size)
            if why is None:
                break
            part.unlink(missing_ok=True)
            if refetched:
                raise DownloadError(f"{label}: verification failed twice ({why})")
            refetched = True
            log.warning("%s: verification failed (%s); fetching once more from scratch", label, why)
    _replace_with_retry(part, dest, sleep=sleep)
    log.info("%s: done (%s)", label, _fmt_mb(dest.stat().st_size))
    return dest


# ----------------------------------------------------------------------------------------------------------------------
# Zenodo
# ----------------------------------------------------------------------------------------------------------------------
def _get_json(url: str, transport: httpx.BaseTransport | None, retries: int = 5,
              sleep: Callable[[float], None] = time.sleep):
    last: Exception | None = None
    with httpx.Client(transport=transport, follow_redirects=True, timeout=httpx.Timeout(60.0, connect=30.0)) as c:
        for i in range(retries + 1):
            try:
                r = c.get(url, headers={"Accept": "application/json"})
                if r.status_code in _RETRY_STATUS:
                    raise _Transient(f"HTTP {r.status_code}", _retry_after(r))
                if r.status_code >= 400:
                    raise DownloadError(f"GET {url}: HTTP {r.status_code}")
                return r.json()
            except (_Transient, httpx.TransportError, ValueError) as e:
                last = e
                if i < retries:
                    sleep(min(30.0, 2.0 ** i))
    raise DownloadError(f"GET {url} failed after {retries} retries: {last}")


def _md5_of(checksum) -> str | None:
    if not checksum or not isinstance(checksum, str):
        return None
    c = checksum.strip().lower()
    if ":" in c:
        algo, _, val = c.partition(":")
        return val if algo == "md5" else None
    return c if re.fullmatch(r"[0-9a-f]{32}", c) else None


def _file_entries(payload) -> list[dict] | None:
    """Locate the list of file dicts in either API shape; None if the payload has no file listing."""
    if not isinstance(payload, dict):
        return None
    for key in ("entries", "files"):
        v = payload.get(key)
        if isinstance(v, list):
            return v
        if isinstance(v, dict):                    # InvenioRDM record: files = {"enabled":.., "entries": {name: {...}}}
            e = v.get("entries")
            if isinstance(e, dict):
                return [{"key": k, **x} if isinstance(x, dict) and "key" not in x else x for k, x in e.items()]
            if isinstance(e, list):
                return e
    return None


def zenodo_record_files(record_id: str | int, transport: httpx.BaseTransport | None = None) -> list[dict]:
    """List a Zenodo record's files as ``[{name, size, md5, url}]`` (sorted by name; md5 may be None)."""
    rid = str(record_id).strip()
    payload = _get_json(ZENODO_API.format(id=rid), transport)
    entries = _file_entries(payload)
    if entries is None:
        payload = _get_json(ZENODO_FILES_API.format(id=rid), transport)
        entries = _file_entries(payload)
    if entries is None:
        raise DownloadError(f"Zenodo record {rid}: no file listing found (top-level keys: "
                            f"{sorted(payload) if isinstance(payload, dict) else type(payload).__name__})")
    legacy = isinstance(payload, dict) and isinstance(payload.get("files"), list)
    out = []
    for e in entries:
        name = e.get("key") or e.get("filename") or e.get("name")
        if not name:
            continue
        links = e.get("links") or {}
        url = links.get("content") or links.get("download")
        if not url:
            selfl = links.get("self")
            # legacy shape: links.self *is* the download URL; in the entries shape it is the metadata URL
            if selfl and (legacy or selfl.rstrip("/").endswith("/content")):
                url = selfl
        if not url:
            url = f"https://zenodo.org/records/{rid}/files/{name}?download=1"
        size = e.get("size", e.get("filesize"))
        out.append({"name": str(name), "size": int(size) if size is not None else None,
                    "md5": _md5_of(e.get("checksum")), "url": url})
    return sorted(out, key=lambda d: d["name"])


def _safe_relpath(name: str) -> Path:
    p = Path(*[s for s in re.split(r"[\\/]+", name) if s])
    if p.is_absolute() or ".." in p.parts or not p.parts:
        raise DownloadError(f"unsafe file name in record: {name!r}")
    return p


def fetch_zenodo(record_id: str | int, dest_dir: Path, only: Callable[[str], bool] | None = None,
                 transport: httpx.BaseTransport | None = None, retries: int = 8, chunk: int = 8 << 20,
                 sleep: Callable[[float], None] = time.sleep) -> list[Path]:
    """Download every (selected) file of a Zenodo record into ``dest_dir``; files already present with matching
    md5/size are skipped.  Also writes ``ZENODO_<id>_FILES.json`` (name/size/md5/url) next to the data."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    files = zenodo_record_files(record_id, transport=transport)
    if only is not None:
        files = [f for f in files if only(f["name"])]
    total = sum(f["size"] or 0 for f in files)
    log.info("Zenodo %s: %d files, %.2f GB to fetch (%.1f GB free in %s)", record_id, len(files), total / 1e9,
             disk_free_gb(dest_dir), dest_dir)
    if total and disk_free_gb(dest_dir) * 1e9 < total * 1.02:
        log.warning("free disk space looks smaller than the download size (%.1f GB needed)", total / 1e9)
    write_json(dest_dir / f"ZENODO_{record_id}_FILES.json", files)
    out: list[Path] = []
    for i, f in enumerate(files, 1):
        log.info("[%d/%d] %s (%s)", i, len(files), f["name"], _fmt_mb(f["size"] or 0))
        out.append(download(f["url"], dest_dir / _safe_relpath(f["name"]), expected_md5=f["md5"],
                            expected_size=f["size"], retries=retries, chunk=chunk, transport=transport, sleep=sleep))
    return out
