#!/usr/bin/env python
"""Read-only Model Context Protocol server over stdio for a Melomane release bundle.

Implements JSON-RPC 2.0 by hand (newline-delimited messages, protocol 2024-11-05 compatible) with NO dependency on the
``mcp`` SDK.  Tools: ``neighbors``, ``slerp``, ``list_modes``, ``mode_members``, ``describe_track``, ``list_poles``.
Tool results are one text content block containing JSON.  The server never writes files, never touches the network and
only exposes metadata (artist/title/album/year/category/mode labels) - never lyrics or audio.

Run:  ``python apps/mcp_server/server.py --bundle path/to/core [--bundle path/to/core-small ...]``
(or set ``MELOMANE_BUNDLE`` to one path).  Several bundles can be named: ``--bundle core=path/to/core``; tools take an
optional ``model`` argument (default: the first bundle).  stdout carries protocol messages only; logs go to stderr.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, TextIO

try:
    from melomane.api import Melomane, TrackNotFound
except ImportError:                                            # running from a checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from melomane.api import Melomane, TrackNotFound

log = logging.getLogger("melomane.mcp")

SUPPORTED_VERSIONS = ("2024-11-05", "2025-03-26", "2025-06-18")
DEFAULT_VERSION = "2024-11-05"
SERVER_INFO = {"name": "melomane-mcp", "version": "0.1.0"}
MAX_K = 50
MAX_STR = 300
PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS, INTERNAL_ERROR = -32700, -32600, -32601, -32602, -32603


class ToolError(Exception):
    """A tool failed in a way the model should see (returned as ``isError: true`` content, not a protocol error)."""


def _schema(props: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props, "required": required or [], "additionalProperties": False}


_MODEL = {"type": "string", "description": "Which loaded bundle to query (default: the first)."}
_NAME = {"type": "string", "maxLength": MAX_STR,
         "description": "Track as 'Artist - Title' (fuzzy, normalised), or 'cid:<n>'."}
_K = {"type": "integer", "minimum": 1, "maximum": MAX_K, "default": 10}
TOOLS: list[dict] = [
    {"name": "neighbors", "description": "Nearest tracks (cosine) to a seed track in the Melomane embedding; artist-disjoint by default.",
     "inputSchema": _schema({"name": _NAME, "k": _K, "exclude_artist": {"type": "boolean", "default": True}, "model": _MODEL}, ["name"])},
    {"name": "slerp", "description": ("Rotate a seed track theta degrees toward a pole (or several, composed with weights) and return the nearest "
                                      "tracks. Poles: supervised names ('category:jazz'), 'mode:<id>', 'factor:<i>', 'track:<name>'. "
                                      "cos(query, seed) = cos(theta)."),
     "inputSchema": _schema({"seed": _NAME,
                             "direction": {"description": "One pole name or a list of pole names.",
                                           "oneOf": [{"type": "string", "maxLength": MAX_STR},
                                                     {"type": "array", "items": {"type": "string", "maxLength": MAX_STR}, "minItems": 1, "maxItems": 8}]},
                             "theta_deg": {"type": "number", "minimum": 0, "maximum": 90},
                             "weights": {"type": "array", "items": {"type": "number", "minimum": 0}, "maxItems": 8},
                             "k": _K, "exclude_artist": {"type": "boolean", "default": True}, "model": _MODEL},
                            ["seed", "direction", "theta_deg"])},
    {"name": "list_modes", "description": "The mode atlas (ICA/GMM modes): id, label, property, driver, size, coherence. Optional text filter and paging.",
     "inputSchema": _schema({"query": {"type": "string", "maxLength": MAX_STR}, "limit": {"type": "integer", "minimum": 1, "maximum": 500, "default": 100},
                             "offset": {"type": "integer", "minimum": 0, "default": 0}, "model": _MODEL})},
    {"name": "mode_members", "description": "Members of one mode, most pole-like first.",
     "inputSchema": _schema({"mode_id": {"type": "integer", "minimum": 0}, "limit": {"type": "integer", "minimum": 1, "maximum": 500, "default": 50},
                             "model": _MODEL}, ["mode_id"])},
    {"name": "describe_track", "description": "Metadata for a track: artist, title, album, year, category, region tags, mode labels. Never lyrics or audio.",
     "inputSchema": _schema({"name": _NAME, "model": _MODEL}, ["name"])},
    {"name": "list_poles", "description": "Addressable poles for slerp: supervised (category/region/era/dimension), modes and factors.",
     "inputSchema": _schema({"kind": {"type": "string", "enum": ["supervised", "mode", "factor"]}, "model": _MODEL})},
]


class MelomaneMCPServer:
    """Transport-independent JSON-RPC handler; :meth:`serve` wires it to text streams."""

    def __init__(self, bundles: dict[str, Melomane]) -> None:
        if not bundles:
            raise ValueError("at least one bundle is required")
        self.bundles = bundles
        self.default = next(iter(bundles))
        self.initialised = False
        self._tools: dict[str, Callable[[dict], Any]] = {
            "neighbors": self._neighbors, "slerp": self._slerp, "list_modes": self._list_modes,
            "mode_members": self._mode_members, "describe_track": self._describe, "list_poles": self._list_poles}

    # ---- helpers --------------------------------------------------------------------------------------------------
    def _model(self, args: dict) -> Melomane:
        name = args.get("model", self.default)
        if name not in self.bundles:
            raise ToolError(f"unknown model {name!r}; loaded: {sorted(self.bundles)}")
        return self.bundles[name]

    @staticmethod
    def _str(args: dict, key: str) -> str:
        v = args.get(key)
        if not isinstance(v, str) or not v.strip():
            raise ToolError(f"'{key}' must be a non-empty string")
        if len(v) > MAX_STR:
            raise ToolError(f"'{key}' is longer than {MAX_STR} characters")
        return v

    @staticmethod
    def _int(args: dict, key: str, default: int, lo: int, hi: int) -> int:
        v = args.get(key, default)
        if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
            raise ToolError(f"'{key}' must be an integer in [{lo}, {hi}]")
        return v

    @staticmethod
    def _bool(args: dict, key: str, default: bool) -> bool:
        v = args.get(key, default)
        if not isinstance(v, bool):
            raise ToolError(f"'{key}' must be a boolean")
        return v

    # ---- tools ----------------------------------------------------------------------------------------------------
    def _neighbors(self, a: dict) -> Any:
        m = self._model(a)
        return m.neighbors(self._str(a, "name"), k=self._int(a, "k", 10, 1, MAX_K), exclude_artist=self._bool(a, "exclude_artist", True))

    def _slerp(self, a: dict) -> Any:
        m = self._model(a)
        d = a.get("direction")
        if isinstance(d, str):
            dirs: Any = self._str(a, "direction")
        elif isinstance(d, list) and d and len(d) <= 8 and all(isinstance(x, str) and 0 < len(x) <= MAX_STR for x in d):
            dirs = d
        else:
            raise ToolError("'direction' must be a pole name or a non-empty list (max 8) of pole names")
        theta = a.get("theta_deg")
        if isinstance(theta, bool) or not isinstance(theta, (int, float)) or not 0 <= theta <= 90:
            raise ToolError("'theta_deg' must be a number in [0, 90]")
        w = a.get("weights")
        if w is not None and not (isinstance(w, list) and all(isinstance(x, (int, float)) and not isinstance(x, bool) and x >= 0 for x in w)):
            raise ToolError("'weights' must be a list of non-negative numbers")
        return m.slerp(self._str(a, "seed"), dirs, float(theta), k=self._int(a, "k", 10, 1, MAX_K), weights=w,
                       exclude_artist=self._bool(a, "exclude_artist", True))

    def _list_modes(self, a: dict) -> Any:
        m = self._model(a)
        modes = m.list_modes()
        q = a.get("query")
        if q is not None:
            if not isinstance(q, str) or len(q) > MAX_STR:
                raise ToolError("'query' must be a short string")
            ql = q.lower()
            modes = [x for x in modes if ql in " ".join(str(x.get(k) or "") for k in ("label", "property", "driver")).lower()]
        off, lim = self._int(a, "offset", 0, 0, 10**9), self._int(a, "limit", 100, 1, 500)
        return {"total": len(modes), "offset": off, "modes": modes[off:off + lim]}

    def _mode_members(self, a: dict) -> Any:
        m = self._model(a)
        mid = self._int(a, "mode_id", -1, 0, 10**9)
        limit = self._int(a, "limit", 50, 1, 500)
        rows = m.mode_members(mid)
        return {"mode_id": mid, "size": len(rows), "members": rows[:limit]}

    def _describe(self, a: dict) -> Any:
        return self._model(a).describe(self._str(a, "name"))

    def _list_poles(self, a: dict) -> Any:
        poles = self._model(a).list_poles()
        kind = a.get("kind")
        if kind is not None:
            if kind not in ("supervised", "mode", "factor"):
                raise ToolError("'kind' must be one of supervised, mode, factor")
            poles = [p for p in poles if p["group"] == kind]
        return {"total": len(poles), "poles": poles}

    # ---- JSON-RPC -------------------------------------------------------------------------------------------------
    @staticmethod
    def _err(id_: Any, code: int, message: str) -> dict:
        return {"jsonrpc": "2.0", "id": id_, "error": {"code": code, "message": message}}

    def handle(self, msg: Any) -> dict | list | None:
        """Handle one decoded JSON-RPC message (or batch).  Returns the response object, or None for notifications."""
        if isinstance(msg, list):
            if not msg:
                return self._err(None, INVALID_REQUEST, "empty batch")
            out = [r for r in (self.handle(m) for m in msg) if r is not None]
            return out or None
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0" or not isinstance(msg.get("method"), str):
            return self._err(msg.get("id") if isinstance(msg, dict) else None, INVALID_REQUEST, "not a JSON-RPC 2.0 request")
        has_id = "id" in msg
        id_ = msg.get("id")
        method, params = msg["method"], msg.get("params") or {}
        if not isinstance(params, dict):
            return self._err(id_, INVALID_PARAMS, "params must be an object") if has_id else None
        try:
            if method == "initialize":
                asked = params.get("protocolVersion")
                self.initialised = True
                result: Any = {"protocolVersion": asked if asked in SUPPORTED_VERSIONS else DEFAULT_VERSION,
                               "capabilities": {"tools": {"listChanged": False}}, "serverInfo": SERVER_INFO,
                               "instructions": "Read-only Melomane music-embedding tools (neighbors, slerp, modes, poles). "
                                               "Metadata only: no lyrics, no audio."}
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": TOOLS}
            elif method == "tools/call":
                result = self._call(params)
            elif method.startswith("notifications/"):
                return None
            else:
                return self._err(id_, METHOD_NOT_FOUND, f"method not found: {method}") if has_id else None
        except _ProtocolError as e:
            return self._err(id_, e.code, str(e)) if has_id else None
        except Exception as e:                                              # noqa: BLE001 - never crash the loop
            log.exception("internal error handling %s", method)
            return self._err(id_, INTERNAL_ERROR, f"internal error: {type(e).__name__}") if has_id else None
        return {"jsonrpc": "2.0", "id": id_, "result": result} if has_id else None

    def _call(self, params: dict) -> dict:
        name, args = params.get("name"), params.get("arguments") or {}
        if name not in self._tools:
            raise _ProtocolError(INVALID_PARAMS, f"unknown tool: {name!r}")
        if not isinstance(args, dict):
            raise _ProtocolError(INVALID_PARAMS, "arguments must be an object")
        try:
            payload = self._tools[name](args)
            return {"content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}], "isError": False}
        except (ToolError, TrackNotFound, KeyError, ValueError) as e:
            msg = e.args[0] if isinstance(e, KeyError) and e.args else str(e)
            return {"content": [{"type": "text", "text": json.dumps({"error": str(msg)}, ensure_ascii=False)}], "isError": True}

    def handle_line(self, line: str) -> str | None:
        """One newline-delimited message in, one serialised response (or None) out."""
        line = line.strip()
        if not line:
            return None
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            return json.dumps(self._err(None, PARSE_ERROR, "parse error"))
        resp = self.handle(msg)
        return None if resp is None else json.dumps(resp, ensure_ascii=False, separators=(",", ":"))

    def serve(self, stdin: TextIO, stdout: TextIO) -> None:
        """Read messages until EOF; write responses (one JSON document per line, flushed)."""
        for line in stdin:
            out = self.handle_line(line)
            if out is not None:
                stdout.write(out + "\n")
                stdout.flush()


class _ProtocolError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


def load_bundles(specs: list[str]) -> dict[str, Melomane]:
    """``['path']`` or ``['name=path', ...]`` -> loaded bundles (name defaults to the directory name)."""
    out: dict[str, Melomane] = {}
    for s in specs:
        name, _, path = s.partition("=") if "=" in s and not Path(s).exists() else ("", "", s)
        m = Melomane.from_pretrained(path)
        out[name or Path(path).name] = m
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Read-only Melomane MCP server (stdio).")
    ap.add_argument("--bundle", action="append", default=[], help="bundle directory, optionally name=dir (repeatable)")
    args = ap.parse_args(argv)
    specs = args.bundle or ([os.environ["MELOMANE_BUNDLE"]] if os.environ.get("MELOMANE_BUNDLE") else [])
    if not specs:
        ap.error("give --bundle DIR (or set MELOMANE_BUNDLE)")
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(name)s: %(message)s")
    server = MelomaneMCPServer(load_bundles(specs))
    log.info("ready: %s", ", ".join(f"{k} ({len(v)} tracks)" for k, v in server.bundles.items()))
    if hasattr(sys.stdin, "reconfigure"):                          # Windows: UTF-8 and bare "\n" line endings on the pipes
        sys.stdin.reconfigure(encoding="utf-8")
        sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    server.serve(sys.stdin, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
