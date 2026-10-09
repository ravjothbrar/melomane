"""Downloader tests: in-process fake HTTP server via httpx.MockTransport (no network)."""
import hashlib
import re

import httpx
import pytest

from melomane import download as dl

PAYLOAD = bytes(range(256)) * 400          # 102,400 bytes, non-repeating-at-boundaries enough for offset checks
MD5 = hashlib.md5(PAYLOAD).hexdigest()
NOSLEEP = lambda s: None


class _IterStream(httpx.SyncByteStream):
    def __init__(self, it):
        self._it = it

    def __iter__(self):
        yield from self._it


def _resp(code, headers=None, body=b"", stream=None):
    """Response whose body is an *unread* stream, like a real transport produces (content=bytes would be pre-read)."""
    return httpx.Response(code, headers=headers or {}, stream=stream if stream is not None else httpx.ByteStream(body))


class FakeServer:
    """Serves PAYLOAD at /f.bin with optional Range support and scripted failures."""

    def __init__(self, data=PAYLOAD, support_range=True, fail_after=None, fail_times=0, status_script=None,
                 wrong_first=False):
        self.data, self.support_range = data, support_range
        self.fail_after, self.fail_times = fail_after, fail_times
        self.status_script = list(status_script or [])
        self.wrong_first = wrong_first
        self.requests: list[dict] = []

    def _stream(self, body: bytes, interrupt: bool):
        def gen():
            step = 1000
            for i in range(0, len(body), step):
                if interrupt and i >= self.fail_after:
                    raise httpx.ReadError("connection reset")
                yield body[i:i + step]
        return gen()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        rng = request.headers.get("range")
        self.requests.append({"range": rng, "url": str(request.url), "ae": request.headers.get("accept-encoding")})
        if self.status_script:
            code = self.status_script.pop(0)
            if code:
                return _resp(code, {"Retry-After": "0"} if code == 429 else {})
        data = self.data
        interrupt = self.fail_times > 0
        start = 0
        m = re.match(r"bytes=(\d+)-", rng or "")
        if m and self.support_range:
            start = int(m.group(1))
            if start >= len(data):
                return _resp(416, {"Content-Range": f"bytes */{len(data)}"})
            body = data[start:]
            hdr = {"Content-Range": f"bytes {start}-{len(data) - 1}/{len(data)}", "Content-Length": str(len(body))}
            code = 206
        else:
            body, hdr, code = data, {"Content-Length": str(len(data))}, 200
        if interrupt:
            self.fail_times -= 1
            return _resp(code, hdr, stream=_IterStream(self._stream(body, True)))
        return _resp(code, hdr, body)


def _dl(tmp_path, server, **kw):
    kw.setdefault("sleep", NOSLEEP)
    kw.setdefault("chunk", 4096)
    return dl.download("https://example.org/f.bin", tmp_path / "out" / "f.bin", transport=httpx.MockTransport(server), **kw)


def test_plain_download_verifies_and_is_atomic(tmp_path):
    s = FakeServer()
    p = _dl(tmp_path, s, expected_md5=MD5, expected_size=len(PAYLOAD))
    assert p.read_bytes() == PAYLOAD
    assert not p.with_suffix(".bin.part").exists()
    assert s.requests[0]["range"] is None and s.requests[0]["ae"] == "identity"


def test_resume_after_interruption_uses_range(tmp_path):
    s = FakeServer(fail_after=30_000, fail_times=1)
    p = _dl(tmp_path, s, expected_md5=MD5)
    assert p.read_bytes() == PAYLOAD
    assert len(s.requests) == 2
    start = int(s.requests[1]["range"].split("=")[1].rstrip("-"))
    assert 0 < start <= 30_000                              # resumed from what was flushed, not from zero


def test_resume_from_preexisting_part_file(tmp_path):
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "f.bin.part").write_bytes(PAYLOAD[:12345])
    s = FakeServer()
    p = _dl(tmp_path, s, expected_md5=MD5)
    assert p.read_bytes() == PAYLOAD and s.requests[0]["range"] == "bytes=12345-"


def test_server_ignoring_range_restarts(tmp_path):
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "f.bin.part").write_bytes(PAYLOAD[:5000])
    s = FakeServer(support_range=False)
    p = _dl(tmp_path, s, expected_md5=MD5)
    assert p.read_bytes() == PAYLOAD          # not corrupted by appending a full body to a partial prefix


def test_416_already_complete_part(tmp_path):
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "f.bin.part").write_bytes(PAYLOAD)      # complete but never renamed (crash before rename)
    s = FakeServer()
    p = _dl(tmp_path, s, expected_md5=MD5)                       # no expected_size: relies on the 416 path
    assert p.read_bytes() == PAYLOAD
    assert s.requests[0]["range"] == f"bytes={len(PAYLOAD)}-"


def test_416_with_wrong_size_restarts(tmp_path):
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "f.bin.part").write_bytes(PAYLOAD + b"junk")
    s = FakeServer()
    p = _dl(tmp_path, s, expected_md5=MD5)
    assert p.read_bytes() == PAYLOAD


def test_md5_mismatch_refetches_once_then_fails(tmp_path):
    s = FakeServer()
    with pytest.raises(dl.DownloadError, match="verification failed twice"):
        _dl(tmp_path, s, expected_md5="0" * 32)
    assert len(s.requests) == 2
    assert not (tmp_path / "out" / "f.bin").exists() and not (tmp_path / "out" / "f.bin.part").exists()


def test_corrupt_first_fetch_then_good(tmp_path):
    bad = bytearray(PAYLOAD); bad[100] ^= 0xFF
    servers = [FakeServer(data=bytes(bad)), FakeServer()]
    calls = []

    def handler(req):
        calls.append(1)
        return servers[0 if len(calls) == 1 else 1](req)
    p = dl.download("https://example.org/f.bin", tmp_path / "f.bin", expected_md5=MD5, transport=httpx.MockTransport(handler),
                    sleep=NOSLEEP)
    assert p.read_bytes() == PAYLOAD and len(calls) == 2


def test_retries_transient_statuses_with_backoff(tmp_path):
    sleeps = []
    s = FakeServer(status_script=[503, 429, 500])
    p = _dl(tmp_path, s, expected_md5=MD5, sleep=sleeps.append)
    assert p.read_bytes() == PAYLOAD and len(s.requests) == 4
    assert sleeps == [1.0, 2.0, 4.0]


def test_gives_up_after_retries_and_fatal_status(tmp_path):
    s = FakeServer(status_script=[503] * 10)
    with pytest.raises(dl.DownloadError, match="giving up"):
        _dl(tmp_path, s, retries=3)
    s = FakeServer(status_script=[404])
    with pytest.raises(dl.DownloadError, match="404"):
        _dl(tmp_path, s)


def test_short_body_counts_as_interruption(tmp_path):
    calls = []

    def handler(req):
        calls.append(req.headers.get("range"))
        if len(calls) == 1:     # promises the whole file, delivers half, closes cleanly
            return _resp(200, {"Content-Length": str(len(PAYLOAD))}, PAYLOAD[:50_000])
        return FakeServer()(req)
    p = dl.download("https://example.org/f.bin", tmp_path / "f.bin", expected_md5=MD5, transport=httpx.MockTransport(handler),
                    sleep=NOSLEEP)
    assert p.read_bytes() == PAYLOAD and calls[1] == "bytes=50000-"


def test_existing_file_skipped_or_replaced(tmp_path):
    (tmp_path / "out").mkdir()
    f = tmp_path / "out" / "f.bin"
    f.write_bytes(PAYLOAD)
    s = FakeServer()
    _dl(tmp_path, s, expected_md5=MD5)
    assert s.requests == []                                   # verified -> no request
    f.write_bytes(b"stale")
    _dl(tmp_path, s, expected_md5=MD5)
    assert f.read_bytes() == PAYLOAD and len(s.requests) == 1


# ----------------------------------------------------------------------------------------------------------------------
def _record_server(files: dict[str, bytes], shape="legacy", record_id="777"):
    def handler(req: httpx.Request) -> httpx.Response:
        path = req.url.path
        if path == f"/api/records/{record_id}":
            if shape == "legacy":
                return httpx.Response(200, json={"files": [
                    {"key": k, "size": len(v), "checksum": "md5:" + hashlib.md5(v).hexdigest(),
                     "links": {"self": f"https://zenodo.org/api/records/{record_id}/files/{k}/content"}}
                    for k, v in files.items()]})
            if shape == "rdm":
                return httpx.Response(200, json={"files": {"enabled": True, "entries": {
                    k: {"key": k, "size": len(v), "checksum": "md5:" + hashlib.md5(v).hexdigest(),
                        "links": {"self": f"https://zenodo.org/api/records/{record_id}/files/{k}",
                                  "content": f"https://zenodo.org/api/records/{record_id}/files/{k}/content"}}
                    for k, v in files.items()}}})
            return httpx.Response(200, json={"id": 777, "metadata": {}})           # 'nofiles': force /files fallback
        if path == f"/api/records/{record_id}/files":
            return httpx.Response(200, json={"entries": [
                {"key": k, "size": len(v), "checksum": "md5:" + hashlib.md5(v).hexdigest(),
                 "links": {"content": f"https://zenodo.org/api/records/{record_id}/files/{k}/content"}}
                for k, v in files.items()]})
        m = re.fullmatch(rf"/api/records/{record_id}/files/(.+)/content", path)
        if m and m.group(1) in files:
            return FakeServer(files[m.group(1)])(req)
        return httpx.Response(404)
    return handler


@pytest.mark.parametrize("shape", ["legacy", "rdm", "nofiles"])
def test_zenodo_listing_shapes(shape):
    files = {"id_essentia.tsv.bz2": b"a" * 50, "userid_trackid_timestamp.tsv.bz2": b"b" * 70}
    listing = dl.zenodo_record_files(777, transport=httpx.MockTransport(_record_server(files, shape)))
    assert [f["name"] for f in listing] == sorted(files)
    assert all(f["md5"] == hashlib.md5(files[f["name"]]).hexdigest() and f["size"] == len(files[f["name"]]) for f in listing)
    assert all(f["url"].endswith(f"/files/{f['name']}/content") for f in listing)


def test_fetch_zenodo_downloads_skips_and_filters(tmp_path):
    files = {"a.tsv": b"x" * 5000, "b.tsv": b"y" * 3000, "skip.zip": b"z" * 10}
    t = httpx.MockTransport(_record_server(files))
    out = dl.fetch_zenodo(777, tmp_path, only=lambda n: n.endswith(".tsv"), transport=t, sleep=NOSLEEP)
    assert sorted(p.name for p in out) == ["a.tsv", "b.tsv"]
    assert not (tmp_path / "skip.zip").exists()
    assert (tmp_path / "a.tsv").read_bytes() == files["a.tsv"]
    # second run: nothing is fetched (only the API listing is requested)
    seen = []

    def spy(req):
        seen.append(req.url.path)
        return _record_server(files)(req)
    dl.fetch_zenodo(777, tmp_path, only=lambda n: n.endswith(".tsv"), transport=httpx.MockTransport(spy), sleep=NOSLEEP)
    assert seen == ["/api/records/777"]


def test_fetch_zenodo_rejects_path_traversal(tmp_path):
    files = {"../evil.txt": b"1"}
    with pytest.raises(dl.DownloadError, match="unsafe"):
        dl.fetch_zenodo(777, tmp_path, transport=httpx.MockTransport(_record_server(files)), sleep=NOSLEEP)
