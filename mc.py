"""athene/mc.py - reading an ensemble: summaries per metric, fan charts across named cases, and which inputs move an output (rank correlation, robust to scale and to monotone non-linearity)."""
import numpy as np

PCTS = (5, 25, 50, 75, 95)

def summarize(results: list, metrics: list = None) -> dict:
    """{metric: {n, mean, std, min, p5, p25, p50, p75, p95, max}} over the runs that have that metric as a number."""
    ok = [r for r in results if not r.get("error")]
    metrics = metrics or sorted({k for r in ok for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
    out = {}
    for m in metrics:
        v = np.asarray([r[m] for r in ok if isinstance(r.get(m), (int, float))], dtype=float)
        bad = int((~np.isfinite(v)).sum())
        v = v[np.isfinite(v)]
        if v.size or bad: out[m] = {"n": int(v.size), "infinite": bad}
        if v.size: out[m] = {"n": int(v.size), "infinite": bad, "mean": float(v.mean()), "std": float(v.std()), "min": float(v.min()), "max": float(v.max()), **{f"p{p}": float(np.percentile(v, p)) for p in PCTS}}
    return {**out, "_failed": len(results) - len(ok)}

def ranks(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    order = x.argsort(kind="stable")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    for v in np.unique(x):   # ties share their mean rank
        m = x == v
        if m.sum() > 1: r[m] = r[m].mean()
    return r

def spearman(x, y) -> float:
    rx, ry = ranks(x), ranks(y)
    sx, sy = rx.std(), ry.std()
    return float(((rx - rx.mean()) * (ry - ry.mean())).mean() / (sx * sy)) if sx and sy else 0.0

def sensitivity(inputs: list, outputs: list, top: int = 20) -> list:
    """inputs: one {input name: sampled value} per run; outputs: the metric per run. Inputs ranked by |rank correlation|, sign kept: positive means the output rises with the input.
    clear: stronger than chance would give with this many runs; with few runs, unclear rows are noise, and more runs separate them."""
    names = sorted({k for d in inputs for k, v in d.items() if isinstance(v, (int, float))} & set.intersection(*[set(d) for d in inputs])) if inputs else []
    rows = [(n, spearman([d[n] for d in inputs], outputs)) for n in names if len({d[n] for d in inputs}) > 1]
    from statistics import NormalDist
    noise = NormalDist().inv_cdf(1 - 0.025 / max(len(rows), 1)) / np.sqrt(max(len(outputs) - 1, 1))   # chance alone stays below this for every input at once, 95% of the time (Bonferroni over the inputs tested)
    return [{"input": n, "rho": round(r, 4), "clear": abs(r) > noise} for n, r in sorted(rows, key=lambda t: -abs(t[1]))[:top]]
