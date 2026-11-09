"""python3 tests/test_athene.py - every search on problems with known answers, with noise, ensembles and caching."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from load import athene
globals().update({k: getattr(athene(), k) for k in dir(athene()) if not k.startswith("_")})

FAILS = []
def check(name, ok, info=""):
    print(("ok   " if ok else "FAIL ") + name + ("" if ok else f"  {info}"))
    if not ok: FAILS.append(name)

def noisy_sphere(p, seed):
    """Minimum at x=1.5, y=-2, mode 'b'; noise from the seed only."""
    r = np.random.default_rng(seed)
    return {"f": (p["x"] - 1.5) ** 2 + (p["y"] + 2) ** 2 + (0 if p["mode"] == "b" else 3) + r.normal(0, 0.3), "y": p["y"]}

pool = GenePool([FloatGene("x", -5, 5), IntGene("y", -10, 10), ChoiceGene("mode", ["a", "b", "c"])])
cost = CostFunction([Goal("f", "min", scale=1.0)])
rng = np.random.default_rng(1)

# --- Genes ---
p = pool.random(rng)
check("random point in range", -5 <= p["x"] <= 5 and -10 <= p["y"] <= 10 and p["mode"] in "abc", p)
check("mutation stays in range", all(-5 <= pool.mutate(p, rng)["x"] <= 5 for _ in range(200)))
pg = PermGene("o", list(range(8)))
a, b = pg.random(rng), pg.random(rng)
check("order crossover keeps every item once", all(sorted(pg.cross(a, b, rng)) == list(range(8)) for _ in range(100)))
sg = SubsetGene("s", list("abcdef"), 1, 3)
check("subset stays within its sizes", all(1 <= len(sg.mutate(sg.random(rng), rng)) <= 3 for _ in range(200)))
fg = FloatGene("lg", 1e-3, 1e3, log=True)
check("log gene spreads over magnitudes", np.median([fg.random(rng) for _ in range(400)]) < 10)
try:
    GenePool([IntGene("a", 0, 1), IntGene("a", 0, 2)]); check("repeated gene names refused", False)
except ValueError: check("repeated gene names refused", True)

# --- Goals ---
g = Goal("t", "le", target=10, hard=True)
check("hard limit dominates", g.penalty(11) > 1e5 and g.penalty(9) == 0)
check("statistics", stat([1, 2, 3, 4], "p50") == 2.5 and stat([1, 5], "worst", "low") == 5 and stat([1, 5], "worst", "high") == 1)
check("a failed run makes a point infeasible", cost([{"f": 1}, {"error": "x"}])[0] == float("inf"))

# --- Ensembles ---
ens = Ensemble(noisy_sphere, n=8)
r1 = ens.run({"x": 0.0, "y": 0, "mode": "a"})
r2 = ens.run({"x": 0.0, "y": 0, "mode": "a"})
check("cache answers a point met again", r1 == r2 and ens.stats["hits"] == 8)
d = [x["f"] - y["f"] for x, y in zip(ens.run({"x": 1.0, "y": 0, "mode": "a"}), ens.run({"x": 0.0, "y": 0, "mode": "a"}))]
check("common random numbers: the noise cancels in a comparison", np.std(d) < 1e-9, d)
check("a grown ensemble reruns only the new seeds", len(ens.run({"x": 0.0, "y": 0, "mode": "a"}, n=12)) == 12 and ens.stats["runs"] == 8 + 8 + 4)

# --- Searches ---
def near(res): return abs(res["best"]["params"]["x"] - 1.5) < 0.5 and res["best"]["params"]["y"] == -2 and res["best"]["params"]["mode"] == "b"
for cls, kw in ((GA, {"generations": 40, "size": 24}), (Tabu, {"iterations": 120}), (Anneal, {"iterations": 900})):
    res = cls(pool, cost, Ensemble(noisy_sphere, n=6), seed=3, budget=3000).run(**kw)
    check(f"{cls.__name__} finds the minimum through noise", near(res), res["best"])
    check(f"{cls.__name__} keeps a history", len(res["history"]) > 5 and res["history"][-1]["best"] <= res["history"][0]["best"])
res = chain([(GA, {"generations": 10, "size": 16}), (Anneal, {"iterations": 300})], pool, cost, Ensemble(noisy_sphere, n=6), budget=1500)
check("a chain carries points between searches", near(res) and res["evaluations"] <= 1500, res["best"])

# --- A permutation problem: an order that matters ---
pts = np.random.default_rng(7).uniform(0, 1, (9, 2))
def tour(p, seed): return {"len": sum(np.linalg.norm(pts[a] - pts[b]) for a, b in zip(p["o"], p["o"][1:] + p["o"][:1]))}
tp = GenePool([PermGene("o", list(range(9)))])
import itertools
best = min(sum(np.linalg.norm(pts[a] - pts[b]) for a, b in zip((0,) + q, q + (0,))) for q in itertools.permutations(range(1, 9)))
res = GA(tp, CostFunction([Goal("len")]), Ensemble(tour, n=1), seed=2, budget=6000).run(generations=150, size=40)
check("GA orders a tour within 5% of the true best", res["best"]["values"]["len:mean"] <= best * 1.05, (res["best"]["values"], best))

# --- Reading an ensemble ---
ins = [{"a": float(i), "b": float(-i), "c": float(np.random.default_rng(i).random())} for i in range(50)]
outs = [i * 2.0 for i in range(50)]
s = sensitivity(ins, outs)
check("sensitivity ranks the inputs that move the output", {s[0]["input"], s[1]["input"]} == {"a", "b"} and s[0]["rho"] * s[1]["rho"] < 0 and abs(s[2]["rho"]) < 0.5, s)
sm = summarize([{"m": float(i)} for i in range(101)] + [{"error": "x"}])
check("summary percentiles and failures", sm["m"]["p50"] == 50 and sm["_failed"] == 1)

# --- Trade-offs ---
pts = [{"values": {"t": t, "c": c}} for t, c in [(1, 9), (2, 5), (3, 6), (4, 2), (5, 2), (2, float("inf"))]]
front = pareto(pts, {"t": "min", "c": "min"})
check("pareto keeps the points nothing beats on every objective", sorted((p["values"]["t"], p["values"]["c"]) for p in front) == [(1, 9), (2, 5), (4, 2)], front)
r = Anneal(pool, cost, Ensemble(noisy_sphere, n=2), seed=3, budget=30).run(iterations=30)
check("a search returns every point it scored", len(r["points"]) == r["evaluations"] and all("params" in p and "values" in p for p in r["points"]))

# --- Parallel runs give the same answers ---
pe = Ensemble(noisy_sphere, n=6, workers=2)
check("process workers give identical results", pe.run({"x": 0.0, "y": 0, "mode": "a"}) == Ensemble(noisy_sphere, n=6).run({"x": 0.0, "y": 0, "mode": "a"}))
pe.close()

print("ALL PASSED" if not FAILS else f"FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
