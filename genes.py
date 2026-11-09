"""athene/genes.py - what a point in parameter space is made of. A Gene knows the values one parameter may take and how to draw, mutate, cross and step through them; a GenePool is the ordered set of genes that makes an individual.
Every operation takes a numpy Generator, so a run is reproducible from its seed. scale (0..1] sets how far a mutation reaches: searches shrink it as they settle."""
import math

# --- Genes ---

class Gene:
    """One parameter. Subclasses define random, mutate and cross; neighbors defaults to a few small mutations."""
    def __init__(self, name: str): self.name = name
    def random(self, rng): raise NotImplementedError
    def mutate(self, v, rng, scale: float = 1.0): raise NotImplementedError
    def cross(self, a, b, rng): return a if rng.random() < 0.5 else b
    def neighbors(self, v, rng, n: int = 4) -> list: return [self.mutate(v, rng, 0.3) for _ in range(n)]
    def key(self, v): return v
    def describe(self) -> dict: return {"name": self.name, "type": type(self).__name__}

class FloatGene(Gene):
    """A real value in [lo, hi]; log spaces it by orders of magnitude; step rounds it to a grid."""
    def __init__(self, name, lo: float, hi: float, step: float = None, log: bool = False):
        super().__init__(name)
        if log and lo <= 0: raise ValueError(f"{name}: a log gene needs lo > 0")
        self.lo, self.hi, self.step, self.log = float(lo), float(hi), step, log

    def _to(self, v): return math.log(v) if self.log else v
    def _from(self, u): return math.exp(u) if self.log else u
    def _fix(self, v):
        v = min(max(v, self.lo), self.hi)
        return min(max(round((v - self.lo) / self.step) * self.step + self.lo, self.lo), self.hi) if self.step else v

    def random(self, rng): return self._fix(self._from(rng.uniform(self._to(self.lo), self._to(self.hi))))
    def mutate(self, v, rng, scale=1.0):
        span = self._to(self.hi) - self._to(self.lo)
        out = self._fix(self._from(self._to(v) + rng.normal(0, 0.2 * scale * span)))
        return out if out != v or not self.step else self._fix(v + self.step * rng.choice([-1, 1]))
    def cross(self, a, b, rng):
        """Blend crossover: anywhere between the parents and a little beyond, so the range can widen."""
        lo, hi = sorted((self._to(a), self._to(b)))
        d = (hi - lo) * 0.3
        return self._fix(self._from(rng.uniform(lo - d, hi + d)))
    def key(self, v): return round(v, 12)
    def describe(self): return {**super().describe(), "lo": self.lo, "hi": self.hi, "step": self.step, "log": self.log}

class IntGene(FloatGene):
    """An integer in [lo, hi]."""
    def __init__(self, name, lo: int, hi: int, log: bool = False): super().__init__(name, lo, hi, 1, log)
    def _fix(self, v): return int(min(max(round(v), self.lo), self.hi))
    def mutate(self, v, rng, scale=1.0):
        out = super().mutate(v, rng, scale)
        return out if out != v else self._fix(v + rng.choice([-1, 1]))
    def neighbors(self, v, rng, n=4): return sorted({self._fix(v + d) for d in (-2, -1, 1, 2)} - {v}) or [v]

class ChoiceGene(Gene):
    """One of a list of options (any JSON-able values: names, quant types, node ids)."""
    def __init__(self, name, options: list):
        super().__init__(name)
        if not options: raise ValueError(f"{name}: no options")
        self.options = list(options)
    def random(self, rng): return self.options[rng.integers(len(self.options))]
    def mutate(self, v, rng, scale=1.0):
        rest = [o for o in self.options if o != v]
        return rest[rng.integers(len(rest))] if rest else v
    def neighbors(self, v, rng, n=4): return [o for o in self.options if o != v][:max(n, 1)] or [v]
    def key(self, v): return repr(v)
    def describe(self): return {**super().describe(), "options": self.options}

class BoolGene(ChoiceGene):
    def __init__(self, name): super().__init__(name, [False, True])

class PermGene(Gene):
    """An order of items: batch orders, priorities, routes."""
    def __init__(self, name, items: list):
        super().__init__(name)
        self.items = list(items)
    def random(self, rng): return tuple(self.items[i] for i in rng.permutation(len(self.items)))
    def mutate(self, v, rng, scale=1.0):
        v = list(v)
        for _ in range(max(1, int(round(scale * len(v) / 4)))):
            i, j = rng.integers(len(v), size=2)
            if rng.random() < 0.5: v[i], v[j] = v[j], v[i]
            else: v.insert(j, v.pop(i))
        return tuple(v)
    def cross(self, a, b, rng):
        """Order crossover: a slice of one parent kept in place, the rest in the other parent's order."""
        n = len(a)
        if n < 2: return tuple(a)
        i, j = sorted(rng.choice(n, 2, replace=False))
        mid = a[i:j]
        rest = [x for x in b if x not in mid]
        return tuple(rest[:i] + list(mid) + rest[i:])
    def neighbors(self, v, rng, n=4):
        out = []
        for _ in range(n):
            w, (i, j) = list(v), rng.choice(len(v), 2, replace=False) if len(v) > 1 else (0, 0)
            w[i], w[j] = w[j], w[i]
            out.append(tuple(w))
        return out
    def describe(self): return {**super().describe(), "items": self.items}

class SubsetGene(Gene):
    """A set of items, between k_min and k_max of them: which nodes to buy, which models to keep loaded."""
    def __init__(self, name, items: list, k_min: int = 0, k_max: int = None):
        super().__init__(name)
        self.items, self.k_min, self.k_max = list(items), k_min, len(items) if k_max is None else k_max
    def _fix(self, s):
        s = [x for x in self.items if x in set(s)]
        return tuple(s)
    def random(self, rng):
        k = int(rng.integers(self.k_min, self.k_max + 1))
        return self._fix([self.items[i] for i in rng.choice(len(self.items), k, replace=False)])
    def mutate(self, v, rng, scale=1.0):
        s, out = set(v), [x for x in self.items if x not in v]
        moves = [m for m in ("add", "drop", "swap") if (m != "add" or (out and len(s) < self.k_max)) and (m != "drop" or len(s) > self.k_min) and (m != "swap" or (out and s))]
        if not moves: return v
        m = moves[rng.integers(len(moves))]
        if m in ("drop", "swap"): s.discard(sorted(s, key=self.items.index)[rng.integers(len(s))])
        if m in ("add", "swap"): s.add(out[rng.integers(len(out))])
        return self._fix(s)
    def cross(self, a, b, rng):
        pick = [x for x in set(a) | set(b) if (x in a and x in b) or rng.random() < 0.5]
        while len(pick) < self.k_min: pick.append([x for x in self.items if x not in pick][0])
        return self._fix(pick[:self.k_max])
    def describe(self): return {**super().describe(), "items": self.items, "k_min": self.k_min, "k_max": self.k_max}

# --- Pools ---

class GenePool:
    """The genes of one kind of individual, in order. repair(params) -> params may enforce rules no single gene knows (raise ValueError for a point that cannot be made valid)."""
    def __init__(self, genes: list, repair=None):
        names = [g.name for g in genes]
        if len(set(names)) != len(names): raise ValueError(f"gene names repeat: {names}")
        self.genes, self.by_name, self.repair = list(genes), {g.name: g for g in genes}, repair

    def _ok(self, p: dict) -> dict: return self.repair(p) if self.repair else p
    def random(self, rng) -> dict: return self._ok({g.name: g.random(rng) for g in self.genes})

    def mutate(self, p: dict, rng, scale: float = 1.0, rate: float = None) -> dict:
        """Each gene mutates with probability rate (default 1/number of genes), and at least one always does."""
        rate = rate if rate is not None else 1 / len(self.genes)
        hit = [g for g in self.genes if rng.random() < rate] or [self.genes[rng.integers(len(self.genes))]]
        return self._ok({**p, **{g.name: g.mutate(p[g.name], rng, scale) for g in hit}})

    def cross(self, a: dict, b: dict, rng) -> dict: return self._ok({g.name: g.cross(a[g.name], b[g.name], rng) for g in self.genes})

    def neighbors(self, p: dict, rng, n: int = 12) -> list:
        """Points one gene away, spread over the genes."""
        out = []
        for i in range(n):
            g = self.genes[(i + rng.integers(len(self.genes))) % len(self.genes)]
            for v in g.neighbors(p[g.name], rng, 1):
                try: out.append(self._ok({**p, g.name: v}))
                except ValueError: pass
        return out

    def key(self, p: dict) -> tuple: return tuple(g.key(p[g.name]) for g in self.genes)
    def distance(self, a: dict, b: dict) -> float: return sum(g.key(a[g.name]) != g.key(b[g.name]) for g in self.genes) / len(self.genes)
    def describe(self) -> list: return [g.describe() for g in self.genes]
