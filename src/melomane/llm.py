"""OpenRouter chat-completions client for structured (JSON) LLM judgments: cached, budget-guarded, retrying.

Default model `google/gemini-3.7-flash` (OpenRouter slug; standard $0.75/$3.75 per M tokens in/out, `:batch` variant half price).
Design goals: (1) every LLM decision in the pipeline is a *cached function of its inputs* so reruns are free and reproducible,
(2) a hard USD budget can never be silently exceeded, (3) transient failures (429/5xx/timeouts/bad JSON) never kill a night run.
NOTE: OpenRouter feature flags (json_schema support, `reasoning` controls, `usage.include`) vary per model/provider - Stage 0
must smoke-test them and adjust `extra_body` in configs/llm.yaml; nothing here assumes them beyond sane defaults.
"""
from __future__ import annotations
import hashlib, json, os, random, re, sqlite3, threading, time
from concurrent.futures import ThreadPoolExecutor
import httpx
import jsonschema


class BudgetExceeded(RuntimeError): ...
class LLMError(RuntimeError): ...


def _strip_fences(s: str) -> str:
    s = s.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, re.S)
    return m.group(1) if m else s


class LLMClient:
    def __init__(self, api_key: str | None = None, model: str = "google/gemini-3.7-flash",
                 base_url: str = "https://openrouter.ai/api/v1", cache_path: str = "cache/llm.sqlite",
                 budget_usd: float = 150.0, price_in: float = 0.75, price_out: float = 3.75, max_retries: int = 6,
                 timeout: float = 180.0, extra_body: dict | None = None, referer: str = "https://ravjothbrar.com",
                 title: str = "melomane", transport: httpx.BaseTransport | None = None, sleep=time.sleep):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.model, self.base_url, self.max_retries = model, base_url.rstrip("/"), max_retries
        self.price_in, self.price_out, self.budget = price_in, price_out, budget_usd
        self.extra_body = extra_body or {}
        self.headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json",
                        "HTTP-Referer": referer, "X-Title": title}
        self.http = httpx.Client(timeout=timeout, transport=transport)
        self._sleep = sleep
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        self.db = sqlite3.connect(cache_path, check_same_thread=False); self.lock = threading.Lock()
        self.db.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, resp TEXT, cost REAL, tin INT, tout INT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v REAL)")
        self.db.commit(); self.calls = 0; self.cache_hits = 0

    # ---- accounting ----------------------------------------------------------------------------------------------
    def spent(self) -> float:
        with self.lock:
            r = self.db.execute("SELECT COALESCE(SUM(cost),0) FROM cache").fetchone()
        return float(r[0])

    def _key(self, system, user, schema, temperature, max_tokens) -> str:
        blob = json.dumps([self.model, system, user, schema, temperature, max_tokens, self.extra_body], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    # ---- main entry ----------------------------------------------------------------------------------------------
    def complete_json(self, system: str, user: str, schema: dict, name: str = "result", temperature: float = 0.0,
                      max_tokens: int = 8192) -> dict:
        k = self._key(system, user, schema, temperature, max_tokens)
        with self.lock:
            row = self.db.execute("SELECT resp FROM cache WHERE k=?", (k,)).fetchone()
        if row:
            self.cache_hits += 1
            return json.loads(row[0])
        if self.spent() >= self.budget:
            raise BudgetExceeded(f"LLM budget ${self.budget:.2f} reached (spent ${self.spent():.2f})")
        body = {"model": self.model, "temperature": temperature, "max_tokens": max_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "response_format": {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}},
                "provider": {"require_parameters": True}, "usage": {"include": True}}
        body.update(self.extra_body)
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                r = self.http.post(f"{self.base_url}/chat/completions", headers=self.headers, json=body)
                self.calls += 1
                if r.status_code in (429, 500, 502, 503, 504, 408):
                    ra = r.headers.get("Retry-After")
                    self._sleep(float(ra) if ra and ra.replace(".", "").isdigit() else min(60, 2 ** attempt) * (0.5 + random.random()))
                    last_err = LLMError(f"HTTP {r.status_code}"); continue
                if r.status_code >= 400:
                    raise LLMError(f"HTTP {r.status_code}: {r.text[:300]}")
                data = r.json()
                if "error" in data and "choices" not in data:
                    last_err = LLMError(str(data["error"])[:300]); self._sleep(min(30, 2 ** attempt)); continue
                content = data["choices"][0]["message"]["content"]
                out = json.loads(_strip_fences(content))
                jsonschema.validate(out, schema)
                u = data.get("usage", {}) or {}
                tin, tout = int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0))
                cost = float(u["cost"]) if isinstance(u.get("cost"), (int, float)) else (tin * self.price_in + tout * self.price_out) / 1e6
                with self.lock:
                    self.db.execute("INSERT OR REPLACE INTO cache VALUES (?,?,?,?,?)", (k, json.dumps(out), cost, tin, tout))
                    self.db.commit()
                return out
            except (json.JSONDecodeError, jsonschema.ValidationError, KeyError, IndexError) as e:
                last_err = e; body["messages"][-1]["content"] = user + "\n\nReturn ONLY valid JSON that matches the schema exactly."
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last_err = e; self._sleep(min(60, 2 ** attempt))
        raise LLMError(f"gave up after {self.max_retries + 1} attempts: {last_err!r}")

    def complete_json_many(self, jobs: list[dict], workers: int = 8):
        """jobs: dicts of complete_json kwargs. Returns results in order; failures come back as the exception object."""
        def run(j):
            try: return self.complete_json(**j)
            except BudgetExceeded: raise
            except Exception as e: return e
        with ThreadPoolExecutor(workers) as ex:
            return list(ex.map(run, jobs))
