"""athene/search.py - goal-seeking searches over a GenePool, scored by a CostFunction on an Ensemble. Each keeps a history (one row per iteration: best cost, evaluations so far) and can be started from another's population, so they chain: GA to explore, then annealing or tabu to refine.
Individual and Population are plain objects to subclass: an individual is a point (params), its cost, the statistics its goals read, and where it came from."""
import math, time
import numpy as np
from .ensemble import pkey

# --- Individuals ---

class Individual:
    def __init__(self, params: dict, origin: str = "random"): self.params, self.origin, self.cost, self.values, self.results = params, origin, None, {}, None
    def __repr__(self): return f"Individual(cost={self.cost}, params={self.params})"
    def to_dict(self) -> dict: return {"params": self.params, "cost": self.cost, "values": self.values, "origin": self.origin}

class Population(list):
    def best(self) -> Individual: return min(self, key=lambda i: i.cost)
    def ranked(self) -> list: return sorted(self, key=lambda i: i.cost)
    def diversity(self, pool) -> float:
        """Mean share of genes that differ between pairs: near 0 means the population has collapsed onto one point."""
        n = len(self)
        return sum(pool.distance(self[i].params, self[j].params) for i in range(n) for j in range(i + 1, n)) / max(n * (n - 1) / 2, 1)

# --- The common loop ---

class Search:
    """budget: the most points to evaluate. on_iteration(search, row) is called after every iteration (a log, a progress bar, a stop: return True to stop)."""
    individual = Individual

    def __init__(self, pool, cost, ensemble, seed: int = 0, budget: int = 2000, on_iteration=None):
        self.pool, self.cost, self.ens, self.rng, self.budget, self.on_iteration = pool, cost, ensemble, np.random.default_rng(seed), budget, on_iteration
        self.evals, self.history, self.best, self.seen = 0, [], None, {}

    def score(self, ind: Individual) -> Individual:
        k = pkey(ind.params)
        if k in self.seen:
            ind.cost, ind.values = self.seen[k].cost, self.seen[k].values
            return ind
        ind.results = self.ens.run(ind.params)
        ind.cost, ind.values = self.cost(ind.results)
        self.evals += 1
        self.seen[k] = ind
        if self.best is None or ind.cost < self.best.cost: self.best = ind
        return ind

    def _log(self, i: int, **extra) -> bool:
        row = {"iteration": i, "best": self.best.cost if self.best else None, "evaluations": self.evals, "t": time.time(), **extra}
        self.history.append(row)
        return bool(self.on_iteration and self.on_iteration(self, row)) or self.evals >= self.budget

    def start(self, n: int, init: list = None) -> Population:
        pop = Population(self.score(i if isinstance(i, Individual) else self.individual(i, "given")) for i in (init or [])[:n])
        while len(pop) < n: pop.append(self.score(self.individual(self.pool.random(self.rng))))
        return pop

    def result(self) -> dict:
        """best, history, evaluations, ensemble statistics, and points: every point scored (params, cost, the statistics its goals read), for trade-off views such as pareto()."""
        return {"best": self.best.to_dict() if self.best else None, "history": self.history, "evaluations": self.evals, "ensemble": self.ens.stats, "points": [i.to_dict() for i in self.seen.values()]}

def pareto(points: list, objectives: dict) -> list:
    """The points no other point beats on every objective: objectives {value name: "min" or "max"}, read from each point's "values". Points missing a value, or with a non-finite one, are left out."""
    sign = {k: 1.0 if d == "min" else -1.0 for k, d in objectives.items()}
    ok = [p for p in points if all(isinstance(p["values"].get(k), (int, float)) and math.isfinite(p["values"][k]) for k in sign)]
    vec = [[sign[k] * p["values"][k] for k in sign] for p in ok]
    return [p for p, v in zip(ok, vec) if not any(all(a <= b for a, b in zip(w, v)) and any(a < b for a, b in zip(w, v)) for w in vec)]

# --- Genetic algorithm ---

class GA(Search):
    """Tournament selection, crossover, mutation, elitism. Mutation reach starts wide and narrows as generations pass; it widens again when the population collapses."""
    def run(self, generations: int = 50, size: int = 30, elite: int = 2, tournament: int = 3, cx: float = 0.9, rate: float = None, init: list = None) -> dict:
        pop, scale = self.start(size, init), 1.0
        for g in range(generations):
            ranked = pop.ranked()
            nxt = Population(ranked[:elite])
            while len(nxt) < size:
                a, b = self._pick(pop, tournament), self._pick(pop, tournament)
                child = self.pool.cross(a.params, b.params, self.rng) if self.rng.random() < cx else dict(a.params)
                nxt.append(self.score(self.individual(self.pool.mutate(child, self.rng, scale, rate), "bred")))
            pop, div = nxt, nxt.diversity(self.pool)
            scale = min(1.0, scale * 2) if div < 0.05 else max(0.05, scale * 0.93)
            self.population = pop
            if self._log(g, diversity=round(div, 3), scale=round(scale, 3)): break
        self.population = pop
        return self.result()

    def _pick(self, pop, k: int) -> Individual: return min((pop[i] for i in self.rng.integers(len(pop), size=k)), key=lambda i: i.cost)

# --- Tabu search ---

class Tabu(Search):
    """Moves to the best neighbor not visited recently (tenure iterations), even when it is worse, so it walks out of local minima; a tabu point is allowed when it beats the best found (aspiration)."""
    def run(self, iterations: int = 200, neighbors: int = 16, tenure: int = 20, init: list = None) -> dict:
        cur = self.start(1, init)[0]
        recent = [self.pool.key(cur.params)]
        for i in range(iterations):
            cands = [self.score(self.individual(p, "neighbor")) for p in self.pool.neighbors(cur.params, self.rng, neighbors)]
            ok = [c for c in cands if self.pool.key(c.params) not in recent or c.cost < self.best.cost]
            if not ok: ok = [self.individual(self.pool.random(self.rng), "restart")]
            cur = min((self.score(c) for c in ok), key=lambda c: c.cost)
            recent = (recent + [self.pool.key(cur.params)])[-tenure:]
            if self._log(i, current=cur.cost): break
        return self.result()

# --- Simulated annealing ---

class Anneal(Search):
    """Accepts a worse point with probability exp(-delta / T); T falls geometrically. T0 None sets it from the spread of a few random points, so it suits any cost scale."""
    def run(self, iterations: int = 500, t0: float = None, alpha: float = 0.99, t_min: float = 1e-6, init: list = None) -> dict:
        cur = self.start(1, init)[0]
        if t0 is None:
            sample = [c for c in (self.score(self.individual(self.pool.random(self.rng))).cost for _ in range(8)) if math.isfinite(c)]
            t0 = float(np.std(sample)) or 1.0 if sample else 1.0
        t = t0
        for i in range(iterations):
            nxt = self.score(self.individual(self.pool.mutate(cur.params, self.rng, max(0.05, t / t0)), "step"))
            d = nxt.cost - cur.cost
            if d <= 0 or (math.isfinite(d) and self.rng.random() < math.exp(-d / max(t, 1e-300))): cur = nxt
            t = max(t * alpha, t_min)
            if self._log(i, current=cur.cost, temperature=t): break
        return self.result()

def chain(stages: list, pool, cost, ensemble, seed: int = 0, budget: int = 2000) -> dict:
    """[(SearchClass, run kwargs), ...]: each stage starts from the best points of the one before; one budget for all."""
    init, out, used = None, None, 0
    for i, (cls, kw) in enumerate(stages):
        s = cls(pool, cost, ensemble, seed + i, budget - used)
        out = s.run(**kw, init=init)
        used += s.evals
        init = [Individual(x.params, "carried") for x in (s.population.ranked() if getattr(s, "population", None) else [s.best])]
        if used >= budget: break
    return {**out, "evaluations": used}
