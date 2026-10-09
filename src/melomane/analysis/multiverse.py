"""E8 sensitivity grid ("multiverse") runner (PLAN §6 S9): run the analysis in EVERY cell of a parameter grid and report
the whole grid. There is deliberately no 'best cell' accessor - `summarise` states min / median / max and the share of
cells meeting a pre-registered criterion."""
from __future__ import annotations

import itertools
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd


def cell_seed(seed: int, cell_id: int) -> int:
    """Deterministic per-cell seed derived from (seed, cell_id)."""
    return int(np.random.SeedSequence([int(seed), int(cell_id)]).generate_state(1)[0])


def run_grid(fn: Callable[..., Any], grid: Mapping[str, Sequence], seed: int = 0, pass_seed: bool = True,
             raise_errors: bool = False) -> pd.DataFrame:
    """Evaluate fn(**cell_params, seed=cell_seed) over the full Cartesian product of `grid` (dict name -> list of values).

    fn returns a float or a dict of floats (metrics). The result is one row per cell, in grid order, with columns
    cell_id, <one per grid key>, seed, status ('ok'|'error'), error, <metric columns>. A failing cell is recorded
    (status 'error', NaN metrics) rather than dropped, unless raise_errors=True. Cells never depend on each other."""
    keys = list(grid)
    rows = []
    for cid, combo in enumerate(itertools.product(*[list(grid[k]) for k in keys])):
        params = dict(zip(keys, combo))
        s = cell_seed(seed, cid)
        row: dict[str, Any] = {"cell_id": cid, **params, "seed": s, "status": "ok", "error": ""}
        try:
            res = fn(**params, seed=s) if pass_seed else fn(**params)
            if res is None:
                res = {}
            elif not isinstance(res, Mapping):
                res = {"value": res}
            for k, v in res.items():
                row[k] = float(v) if np.isscalar(v) and v is not None else v
        except Exception as exc:                                    # noqa: BLE001 - recorded, never silently dropped
            if raise_errors:
                raise
            row["status"] = "error"
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    df = pd.DataFrame(rows)
    df.attrs["grid_keys"] = keys
    return df


def summarise(df: pd.DataFrame, metric: str, criterion: float | Callable[[pd.Series], Any] | None = None,
              higher_is_better: bool = True, params: Sequence[str] | None = None) -> dict:
    """Whole-grid summary of `metric`: n_cells / n_valid / n_failed, min, q25, median, q75, max, mean, and - if a
    `criterion` is given (a threshold: value >= t when higher_is_better else <= t; or a callable returning a boolean mask) -
    share_meeting over ALL cells (failed cells count as not meeting). `by_param` gives the median of the metric for each value
    of each grid parameter (marginals). `statement` is a ready-to-print sentence."""
    if metric not in df.columns:
        raise KeyError(metric)
    keys = list(params) if params is not None else list(df.attrs.get("grid_keys", []))
    v = pd.to_numeric(df[metric], errors="coerce")
    ok = v.notna()
    n, nv = int(len(df)), int(ok.sum())
    out: dict[str, Any] = {"metric": metric, "n_cells": n, "n_valid": nv, "n_failed": n - nv}
    if nv:
        q = v[ok].quantile([0.25, 0.5, 0.75])
        out.update({"min": float(v[ok].min()), "q25": float(q[0.25]), "median": float(q[0.5]), "q75": float(q[0.75]),
                    "max": float(v[ok].max()), "mean": float(v[ok].mean())})
    else:
        out.update({k: float("nan") for k in ("min", "q25", "median", "q75", "max", "mean")})
    stmt = (f"{metric} over {n} cells ({n - nv} failed): min {out['min']:.4g} / median {out['median']:.4g} / "
            f"max {out['max']:.4g}")
    if criterion is not None:
        if callable(criterion):
            meets = pd.Series(criterion(v), index=v.index).fillna(False).astype(bool) & ok
            desc = "custom criterion"
        else:
            meets = ((v >= criterion) if higher_is_better else (v <= criterion)) & ok
            desc = f"{metric} {'>=' if higher_is_better else '<='} {criterion:g}"
        out["share_meeting"] = float(meets.mean()) if n else float("nan")
        out["n_meeting"] = int(meets.sum())
        out["criterion"] = desc
        stmt += f"; {out['n_meeting']}/{n} cells ({100 * out['share_meeting']:.1f}%) meet {desc}"
    out["by_param"] = {k: {(x.item() if hasattr(x, "item") else x): float(g[metric].pipe(pd.to_numeric, errors="coerce").median())
                           for x, g in df.groupby(k, sort=True)} for k in keys if k in df.columns}
    out["statement"] = stmt
    return out
