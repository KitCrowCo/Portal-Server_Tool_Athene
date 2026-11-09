"""athene/ensemble.py - one point evaluated over many runs of a stochastic model. evaluate(params, seed) -> {metric: value, ...} is any function; the ensemble decides the seeds.
Every point gets the same seeds (common random numbers), so two points are compared on the same luck and far fewer runs separate them. Results are cached by (point, seed): a point met again, or an ensemble grown later, reruns nothing.
workers > 1 runs seeds in separate processes (evaluate and params must then be picklable); 0 runs them in this one."""
import os, json, hashlib
from concurrent.futures import ProcessPoolExecutor

def pkey(params: dict) -> str: return hashlib.sha256(json.dumps(params, sort_keys=True, default=repr).encode()).hexdigest()[:20]

class Ensemble:
    def __init__(self, evaluate, n: int = 16, seed: int = 0, workers: int = 0, cache: bool = True):
        self.evaluate, self.n, self.seed, self.workers = evaluate, n, seed, workers if workers >= 0 else (os.cpu_count() or 1)
        self.cache, self.stats = {} if cache else None, {"runs": 0, "hits": 0}
        self._pool = ProcessPoolExecutor(self.workers) if self.workers > 1 else None

    def seeds(self, n: int = None) -> list: return [self.seed + i for i in range(n or self.n)]

    def run(self, params: dict, n: int = None) -> list:
        k, out, todo = pkey(params), {}, []
        for s in self.seeds(n):
            if self.cache is not None and (k, s) in self.cache:
                out[s] = self.cache[(k, s)]
                self.stats["hits"] += 1
            else: todo.append(s)
        done = list(self._pool.map(self.evaluate, [params] * len(todo), todo)) if self._pool and len(todo) > 1 else [self.evaluate(params, s) for s in todo]
        for s, r in zip(todo, done):
            out[s] = r
            if self.cache is not None: self.cache[(k, s)] = r
        self.stats["runs"] += len(todo)
        return [out[s] for s in self.seeds(n)]

    def close(self):
        if self._pool: self._pool.shutdown()
