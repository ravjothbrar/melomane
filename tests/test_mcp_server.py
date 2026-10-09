"""MCP server: JSON-RPC 2.0 by hand, exercised in-process and through real subprocess pipes."""
import importlib.util
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from fixtures_run import fake_run, fake_bundles  # noqa: F401
from melomane.api import Melomane

SERVER = Path(__file__).resolve().parents[1] / "apps" / "mcp_server" / "server.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("melomane_mcp_server", SERVER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


srv = _load_module()


@pytest.fixture(scope="module")
def server(fake_bundles):
    return srv.MelomaneMCPServer({"core": Melomane.from_pretrained(fake_bundles["core"]),
                                  "core-small": Melomane.from_pretrained(fake_bundles["core-small"])})


def rpc(server, method, params=None, id_=1):
    msg = {"jsonrpc": "2.0", "method": method}
    if id_ is not None:
        msg["id"] = id_
    if params is not None:
        msg["params"] = params
    return server.handle(msg)


def call(server, tool, **args):
    r = rpc(server, "tools/call", {"name": tool, "arguments": args})
    res = r["result"]
    assert res["content"][0]["type"] == "text" and len(res["content"]) == 1
    return json.loads(res["content"][0]["text"]), res["isError"]


def test_initialize_negotiates_version_and_declares_tools_only(server):
    r = rpc(server, "initialize", {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}})
    assert r["jsonrpc"] == "2.0" and r["id"] == 1
    res = r["result"]
    assert res["protocolVersion"] == "2024-11-05" and res["serverInfo"]["name"] == "melomane-mcp"
    assert set(res["capabilities"]) == {"tools"}
    assert rpc(server, "initialize", {"protocolVersion": "2025-03-26"})["result"]["protocolVersion"] == "2025-03-26"
    assert rpc(server, "initialize", {"protocolVersion": "1999-01-01"})["result"]["protocolVersion"] == "2024-11-05"
    assert rpc(server, "initialize")["result"]["protocolVersion"] == "2024-11-05"


def test_notifications_get_no_response_and_ping_works(server):
    assert rpc(server, "notifications/initialized", id_=None) is None
    assert rpc(server, "notifications/cancelled", {"requestId": 1}, id_=None) is None
    assert rpc(server, "ping")["result"] == {}
    assert rpc(server, "unknown/method", id_=None) is None                       # unknown notification: silent


def test_tools_list_schemas(server):
    tools = rpc(server, "tools/list")["result"]["tools"]
    assert {t["name"] for t in tools} == {"neighbors", "slerp", "list_modes", "mode_members", "describe_track", "list_poles"}
    for t in tools:
        assert t["description"] and t["inputSchema"]["type"] == "object" and t["inputSchema"]["additionalProperties"] is False
        assert set(t["inputSchema"]["required"]) <= set(t["inputSchema"]["properties"])
    by = {t["name"]: t for t in tools}
    assert by["slerp"]["inputSchema"]["required"] == ["seed", "direction", "theta_deg"]
    assert by["slerp"]["inputSchema"]["properties"]["theta_deg"]["maximum"] == 90


def test_neighbors_tool_matches_library(server, fake_bundles):
    lib = Melomane.from_pretrained(fake_bundles["core"])
    out, err = call(server, "neighbors", name="cid:12", k=7)
    assert not err and out == lib.neighbors(12, k=7)
    t = lib.vocab[12]
    out2, _ = call(server, "neighbors", name=f"{t['artist']} - {t['title']}", k=7)
    assert out2 == out
    leaky, _ = call(server, "neighbors", name="cid:12", k=7, exclude_artist=False)
    assert leaky == lib.neighbors(12, k=7, exclude_artist=False)
    small, _ = call(server, "neighbors", name="cid:12", k=7, model="core-small")
    assert len(small) == 7


def test_slerp_tool_single_multi_and_weights(server, fake_bundles):
    lib = Melomane.from_pretrained(fake_bundles["core"])
    out, err = call(server, "slerp", seed="cid:3", direction="category:jazz", theta_deg=60, k=5)
    assert not err and out == lib.slerp(3, "category:jazz", 60, k=5)
    multi, _ = call(server, "slerp", seed="cid:3", direction=["category:jazz", "era:1990s"], theta_deg=45, k=5, weights=[2, 1])
    assert multi == lib.slerp(3, ["category:jazz", "era:1990s"], 45, k=5, weights=[2, 1])
    zero, _ = call(server, "slerp", seed="cid:3", direction="mode:0", theta_deg=0, k=5)
    assert [r["cid"] for r in zero] == [r["cid"] for r in lib.neighbors(3, k=5)]
    assert all("cos_seed" in r for r in out)


def test_mode_and_pole_tools(server, fake_bundles):
    lib = Melomane.from_pretrained(fake_bundles["core"])
    modes, err = call(server, "list_modes")
    assert not err and modes["total"] == len(lib.list_modes()) and modes["modes"] == lib.list_modes()
    paged, _ = call(server, "list_modes", limit=3, offset=2)
    assert paged["modes"] == lib.list_modes()[2:5] and paged["total"] == modes["total"]
    label = lib.list_modes()[0]["label"].split()[0]
    filt, _ = call(server, "list_modes", query=label.upper())
    assert 1 <= filt["total"] <= modes["total"] and all(label in " ".join(str(v) for v in m.values()) for m in filt["modes"])
    mem, err = call(server, "mode_members", mode_id=0, limit=4)
    assert not err and mem["mode_id"] == 0 and len(mem["members"]) == 4 and mem["size"] == lib.list_modes()[0]["size"]
    assert mem["members"] == lib.mode_members(0)[:4]
    poles, _ = call(server, "list_poles")
    assert poles["total"] == len(lib.list_poles())
    sup, _ = call(server, "list_poles", kind="supervised")
    assert sup["total"] == len(lib.list_supervised_poles()) and {p["group"] for p in sup["poles"]} == {"supervised"}


def test_describe_track_is_metadata_only(server):
    out, err = call(server, "describe_track", name="cid:20")
    assert not err
    assert {"artist", "title", "category", "year", "mode", "album"} <= set(out)
    assert not {"lyrics", "audio", "preview_url", "user", "artist_id"} & set(out)


@pytest.mark.parametrize("tool,args,needle", [
    ("neighbors", {"name": "zzzz qqqq nonexistent"}, "could not resolve"),
    ("neighbors", {"name": "cid:99999"}, "out of range"),
    ("neighbors", {"name": "cid:1", "k": 0}, "'k'"),
    ("neighbors", {"name": "cid:1", "k": 10_000}, "'k'"),
    ("neighbors", {"name": "cid:1", "k": True}, "'k'"),
    ("neighbors", {"name": ""}, "'name'"),
    ("neighbors", {"name": "x" * 1000}, "longer"),
    ("neighbors", {"name": "cid:1", "exclude_artist": "yes"}, "boolean"),
    ("neighbors", {"name": "cid:1", "model": "nope"}, "unknown model"),
    ("slerp", {"seed": "cid:1", "direction": "category:jazz", "theta_deg": 120}, "theta_deg"),
    ("slerp", {"seed": "cid:1", "direction": "category:jazz", "theta_deg": "45"}, "theta_deg"),
    ("slerp", {"seed": "cid:1", "direction": "category:nonexistent-pole", "theta_deg": 45}, "unknown pole"),
    ("slerp", {"seed": "cid:1", "direction": [], "theta_deg": 45}, "direction"),
    ("slerp", {"seed": "cid:1", "direction": 5, "theta_deg": 45}, "direction"),
    ("slerp", {"seed": "cid:1", "direction": ["category:jazz"], "theta_deg": 45, "weights": [1, 2]}, "weights"),
    ("slerp", {"seed": "cid:1", "direction": ["category:jazz"], "theta_deg": 45, "weights": ["a"]}, "weights"),
    ("mode_members", {"mode_id": 99999}, "unknown mode"),
    ("mode_members", {"mode_id": -1}, "mode_id"),
    ("list_poles", {"kind": "bogus"}, "kind"),
    ("list_modes", {"query": 5}, "query"),
    ("describe_track", {}, "'name'"),
])
def test_tool_errors_are_reported_as_is_error_results(server, tool, args, needle):
    out, err = call(server, tool, **args)
    assert err is True and needle in out["error"]


def test_protocol_errors(server):
    assert rpc(server, "tools/call", {"name": "rm_rf", "arguments": {}})["error"]["code"] == -32602
    assert rpc(server, "tools/call", {"name": "neighbors", "arguments": [1]})["error"]["code"] == -32602
    assert rpc(server, "tools/call", [1, 2])["error"]["code"] == -32602
    assert rpc(server, "resources/list")["error"]["code"] == -32601
    assert server.handle({"jsonrpc": "1.0", "id": 5, "method": "ping"})["error"]["code"] == -32600
    assert server.handle({"jsonrpc": "2.0", "id": 5})["error"]["code"] == -32600
    assert server.handle("nonsense")["error"]["code"] == -32600
    assert server.handle([])["error"]["code"] == -32600


def test_batch_and_line_transport(server):
    batch = [{"jsonrpc": "2.0", "id": 1, "method": "ping"}, {"jsonrpc": "2.0", "method": "notifications/initialized"},
             {"jsonrpc": "2.0", "id": "b", "method": "tools/list"}]
    out = json.loads(server.handle_line(json.dumps(batch)))
    assert [r["id"] for r in out] == [1, "b"]
    assert server.handle_line("   ") is None
    assert json.loads(server.handle_line("{not json"))["error"]["code"] == -32700
    assert server.handle_line(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"})) is None
    line = server.handle_line(json.dumps({"jsonrpc": "2.0", "id": 9, "method": "ping"}))
    assert "\n" not in line and json.loads(line) == {"jsonrpc": "2.0", "id": 9, "result": {}}


def test_serve_in_process_streams(server):
    reqs = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "list_modes", "arguments": {"limit": 1}}}]
    stdin = io.StringIO("\n".join(json.dumps(r) for r in reqs) + "\n")
    stdout = io.StringIO()
    server.serve(stdin, stdout)
    lines = stdout.getvalue().splitlines()
    assert len(lines) == 2 and json.loads(lines[0])["id"] == 1 and json.loads(lines[1])["id"] == 2


def test_server_needs_a_bundle():
    with pytest.raises(ValueError):
        srv.MelomaneMCPServer({})


def test_internal_errors_do_not_kill_the_loop(fake_bundles):
    s = srv.MelomaneMCPServer({"core": Melomane.from_pretrained(fake_bundles["core"])})
    s._tools["neighbors"] = lambda a: 1 / 0
    r = rpc(s, "tools/call", {"name": "neighbors", "arguments": {"name": "cid:1"}})
    assert r["error"]["code"] == -32603 and "ZeroDivisionError" in r["error"]["message"]
    assert rpc(s, "ping")["result"] == {}


def test_subprocess_stdio_end_to_end(fake_bundles):
    """Real process, real pipes: stdout must contain protocol messages only."""
    proc = subprocess.Popen([sys.executable, str(SERVER), "--bundle", f"main={fake_bundles['core']}", "--bundle", str(fake_bundles["core-small"])],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
    reqs = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                                                                            "clientInfo": {"name": "pytest", "version": "1"}}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "neighbors", "arguments": {"name": "cid:4", "k": 3}}},
            {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "neighbors", "arguments": {"name": "cid:4", "k": 3, "model": "core-small"}}},
            {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "describe_track", "arguments": {"name": "cid:4"}}}]
    out, err = proc.communicate("\n".join(json.dumps(r) for r in reqs) + "\n", timeout=120)
    assert proc.returncode == 0, err
    msgs = [json.loads(l) for l in out.splitlines()]                 # every stdout line must be valid JSON-RPC
    assert [m["id"] for m in msgs] == [1, 2, 3, 4, 5] and all(m["jsonrpc"] == "2.0" for m in msgs)
    lib = Melomane.from_pretrained(fake_bundles["core"])
    assert json.loads(msgs[2]["result"]["content"][0]["text"]) == lib.neighbors(4, k=3)
    assert len(json.loads(msgs[3]["result"]["content"][0]["text"])) == 3
    assert json.loads(msgs[4]["result"]["content"][0]["text"])["cid"] == 4
    assert "ready:" in err and "main" in err


def test_subprocess_requires_bundle_and_reports_bad_path(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "MELOMANE_BUNDLE"}
    r = subprocess.run([sys.executable, str(SERVER)], capture_output=True, text=True, timeout=60, env=env, input="")
    assert r.returncode != 0 and "--bundle" in r.stderr
    r = subprocess.run([sys.executable, str(SERVER), "--bundle", str(tmp_path / "nope")], capture_output=True, text=True, timeout=60, input="")
    assert r.returncode != 0 and r.stdout == ""


def test_load_bundles_specs(fake_bundles):
    b = srv.load_bundles([f"a={fake_bundles['core']}", str(fake_bundles["core-small"])])
    assert list(b) == ["a", "core-small"]
