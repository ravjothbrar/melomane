import json, threading, http.server, pytest
from melomane.llm import LLMClient, BudgetExceeded, LLMError

SCHEMA = {"type": "object", "properties": {"label": {"type": "string"}}, "required": ["label"], "additionalProperties": False}


class Server:
    def __init__(self, script):
        self.script, self.calls, self.bodies = list(script), 0, []
        outer = self
        class H(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                n = int(self.headers["Content-Length"]); outer.bodies.append(json.loads(self.rfile.read(n)))
                step = outer.script[min(outer.calls, len(outer.script) - 1)]; outer.calls += 1
                code, payload, hdr = step
                self.send_response(code)
                for k, v in hdr.items(): self.send_header(k, v)
                self.end_headers(); self.wfile.write(json.dumps(payload).encode())
            def log_message(self, *a): pass
        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_port}/api/v1"
    def close(self): self.srv.shutdown()


def ok(label, tin=100, tout=50, cost=None, content=None):
    usage = {"prompt_tokens": tin, "completion_tokens": tout}
    if cost is not None: usage["cost"] = cost
    return (200, {"choices": [{"message": {"content": content or json.dumps({"label": label})}}], "usage": usage}, {})


def client(srv, tmp_path, **kw):
    return LLMClient(api_key="k", base_url=srv.url, cache_path=str(tmp_path / "c.sqlite"), sleep=lambda s: None, **kw)


def test_retry_on_429_then_success_and_cache(tmp_path):
    s = Server([(429, {"error": "slow down"}, {"Retry-After": "0"}), ok("A")])
    c = client(s, tmp_path)
    assert c.complete_json("sys", "u", SCHEMA) == {"label": "A"}
    assert s.calls == 2
    assert c.complete_json("sys", "u", SCHEMA) == {"label": "A"} and s.calls == 2 and c.cache_hits == 1   # cached: free rerun
    assert s.bodies[0]["response_format"]["json_schema"]["strict"] is True
    assert s.bodies[0]["provider"]["require_parameters"] is True
    s.close()


def test_bad_json_and_schema_violation_are_retried(tmp_path):
    s = Server([ok(None, content="not json at all"), ok(None, content=json.dumps({"wrong": 1})), ok(None, content="```json\n{\"label\": \"C\"}\n```")])
    c = client(s, tmp_path)
    assert c.complete_json("s", "u", SCHEMA) == {"label": "C"} and s.calls == 3
    assert "ONLY valid JSON" in s.bodies[-1]["messages"][-1]["content"]
    s.close()


def test_budget_guard_uses_reported_cost_and_persists(tmp_path):
    s = Server([ok("x", cost=0.6)]); c = client(s, tmp_path, budget_usd=1.0)
    c.complete_json("s", "1", SCHEMA); c.complete_json("s", "2", SCHEMA)          # 1.2 >= 1.0
    with pytest.raises(BudgetExceeded): c.complete_json("s", "3", SCHEMA)
    c2 = client(s, tmp_path, budget_usd=1.0)                                        # budget survives a restart
    with pytest.raises(BudgetExceeded): c2.complete_json("s", "4", SCHEMA)
    assert c2.complete_json("s", "1", SCHEMA) == {"label": "x"}                     # cached answers still free past the cap
    s.close()


def test_cost_falls_back_to_token_prices(tmp_path):
    s = Server([ok("x", tin=1_000_000, tout=1_000_000)]); c = client(s, tmp_path)
    c.complete_json("s", "u", SCHEMA)
    assert abs(c.spent() - (0.75 + 3.75)) < 1e-9
    s.close()


def test_gives_up_and_parallel_many(tmp_path):
    s = Server([(500, {"error": "boom"}, {})]); c = client(s, tmp_path, max_retries=2)
    with pytest.raises(LLMError): c.complete_json("s", "u", SCHEMA)
    assert s.calls == 3
    s.close()
    s2 = Server([ok("p")]); c2 = client(s2, tmp_path / "x" if False else tmp_path, max_retries=1)
    res = c2.complete_json_many([dict(system="s", user=f"job{i}", schema=SCHEMA) for i in range(12)], workers=4)
    assert all(r == {"label": "p"} for r in res)
    s2.close()
