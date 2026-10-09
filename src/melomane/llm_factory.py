"""Builds the shared, cached, budget-guarded LLM client from environment + configs/llm.yaml."""
from __future__ import annotations
import os

from .llm import LLMClient
from .workspace import Workspace


class LLMUnavailable(RuntimeError):
    """No usable OpenRouter key: LLM stages degrade (skip) instead of crashing the run."""


def have_key() -> bool:
    k = os.environ.get("OPENROUTER_API_KEY", "").strip()
    return bool(k) and not k.lower().startswith(("your", "sk-or-xxx", "changeme"))


_CLIENTS: dict[str, LLMClient] = {}


def make_llm(ws: Workspace, **overrides) -> LLMClient:
    if not have_key():
        raise LLMUnavailable("OPENROUTER_API_KEY is not set (put it in .env); LLM-dependent stages will be skipped")
    cfg = ws.config("llm")
    key = str(ws.cache / "llm.sqlite")
    if key in _CLIENTS and not overrides:
        return _CLIENTS[key]
    kw = dict(model=os.environ.get("LLM_MODEL") or cfg.get("model", "google/gemini-3.7-flash"),
              budget_usd=float(os.environ.get("LLM_BUDGET_USD") or cfg.get("budget_usd", 150)),
              price_in=cfg.get("price_in", 0.75), price_out=cfg.get("price_out", 3.75),
              extra_body=cfg.get("extra_body") or {}, cache_path=key)
    kw.update(overrides)
    c = LLMClient(**kw)
    if not overrides:
        _CLIENTS[key] = c
    return c
