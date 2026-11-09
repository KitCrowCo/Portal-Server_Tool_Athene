"""athene/goals.py - what "better" means. A Goal reads one metric from every run of an ensemble, reduces it to one number by a statistic (mean, a percentile, worst, best), and turns it into a penalty by its kind. A CostFunction adds the goals' weighted penalties.
Kinds: min, max, target (as close as possible), le / ge (only past the target counts). hard=True turns a violated le/ge/target into a penalty large enough that no soft goal can make up for it.
scale divides the raw value, so goals in seconds and dollars can share one cost; leave it None and the first value seen sets it."""
import numpy as np

HARD = 1e6
STATS = {"mean": np.mean, "median": np.median, "min": np.min, "max": np.max, "std": np.std}

def stat(values, name: str, better: str = "low") -> float:
    """mean, median, min, max, std, pNN (a percentile), worst or best (by the goal's direction)."""
    v = np.asarray(values, dtype=float)
    if name in STATS: return float(STATS[name](v))
    if name in ("worst", "best"): return float(v.max() if (name == "worst") == (better == "low") else v.min())
    if name.startswith("p"): return float(np.percentile(v, float(name[1:])))
    raise ValueError(f"unknown statistic {name}")

class Goal:
    def __init__(self, metric: str, kind: str = "min", target: float = None, weight: float = 1.0, stat: str = "mean", scale: float = None, hard: bool = False):
        if kind not in ("min", "max", "target", "le", "ge"): raise ValueError(f"unknown goal kind {kind}")
        if kind in ("target", "le", "ge") and target is None: raise ValueError(f"goal {metric} {kind} needs a target")
        self.metric, self.kind, self.target, self.weight, self.stat, self.scale, self.hard = metric, kind, target, weight, stat, scale, hard

    def value(self, results: list) -> float:
        try: return stat([r[self.metric] for r in results], self.stat, "high" if self.kind in ("max", "ge") else "low")
        except KeyError: raise KeyError(f"goal metric {self.metric} is not in the results: {sorted(results[0]) if results else []}")

    def penalty(self, v: float) -> float:
        s = self.scale or (abs(self.target) if self.target else 0) or abs(v) or 1.0
        if self.scale is None: self.scale = s
        k, t = self.kind, self.target
        raw = v / s if k == "min" else -v / s if k == "max" else abs(v - t) / s if k == "target" else max(0.0, v - t) / s if k == "le" else max(0.0, t - v) / s
        return raw * self.weight + (HARD * (1 + raw) if self.hard and raw > 0 and k != "min" and k != "max" else 0.0)

    def describe(self) -> dict: return {k: getattr(self, k) for k in ("metric", "kind", "target", "weight", "stat", "scale", "hard")}

class CostFunction:
    """Sum of goal penalties. A run that reports "error" makes the whole point infeasible (cost inf), so a broken plan never wins on luck."""
    def __init__(self, goals: list): self.goals = list(goals)

    def __call__(self, results: list) -> tuple:
        if not results or any(r.get("error") for r in results): return float("inf"), {"error": next((r["error"] for r in results if r.get("error")), "no results")}
        vals = {f"{g.metric}:{g.stat}": g.value(results) for g in self.goals}
        return float(sum(g.penalty(vals[f"{g.metric}:{g.stat}"]) for g in self.goals)), vals

    @classmethod
    def from_spec(cls, spec: list): return cls([Goal(**g) for g in spec])
    def describe(self) -> list: return [g.describe() for g in self.goals]
