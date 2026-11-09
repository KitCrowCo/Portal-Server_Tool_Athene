# Athene

## Where it goes

This repository is a Portal Server tool: put it at `tools/athene/` (so `tools/athene/athene.py` exists) and restart the portal. It has no page and keeps no data; any tool or module reaches it as `ENV["tools"]["athene"]`, and keeps its own results under its own data folder.

`__init__.py` is empty, as in every portal tool; `athene.py` holds the exports. Outside the portal: `pip install -e .` and `from athene import athene`, or load the folder as `tests/load.py` does (it works whatever the folder is called).


Goal-seeking search over parameter spaces whose evaluation is stochastic. General: nothing in it knows about AI. AGPL-3.0-or-later.

- **genes** - what a point is made of: `FloatGene` (linear or log, optional grid), `IntGene`, `ChoiceGene`, `BoolGene`, `PermGene` (orders), `SubsetGene` (which items). A `GenePool` is the ordered set, with an optional `repair` for rules no single gene knows.
- **goals** - what better means: `Goal(metric, kind, target, weight, stat, scale, hard)`, kinds `min`, `max`, `target`, `le`, `ge`; `stat` reduces an ensemble (`mean`, `median`, `pNN`, `worst`, `best`). `CostFunction` adds them. A run that reports `error` makes the point infeasible.
- **ensemble** - one point over many runs. Every point gets the same seeds (common random numbers), so two points are compared on the same luck and far fewer runs separate them. Results are cached by (point, seed). `workers` runs seeds in processes.
- **search** - `GA` (tournament, crossover, mutation whose reach narrows as it settles and widens when the population collapses), `Tabu` (best non-recent neighbour, with aspiration), `Anneal` (temperature set from the cost spread when not given), `chain` (one search starts from another's points). Each keeps a history and returns every point it scored; `pareto(points, {value: "min" | "max"})` picks the trade-off front from them. `Individual` and `Population` are plain classes to subclass.
- **mc** - `summarize` (percentiles per metric, infinite results counted apart), `sensitivity` (rank correlation of each sampled input with an output, with a noise threshold corrected for how many inputs were tested).

```python
athene = ENV["tools"]["athene"]   # in the portal; elsewhere: from athene import athene
pool = athene.GenePool([athene.FloatGene("x", -5, 5), athene.ChoiceGene("mode", ["a", "b"])])
cost = athene.CostFunction([athene.Goal("f", "min"), athene.Goal("t", "le", target=10, hard=True)])
ens = athene.Ensemble(lambda p, seed: my_model(p, seed), n=16)
best = athene.GA(pool, cost, ens, budget=2000).run(generations=50, size=30)["best"]
```

`python3 tests/test_athene.py` - each search finds known minima through noise; an order problem within 5% of the exact best; the cache and common random numbers; process workers give identical results.

Needs numpy. Lineage: the structure (individuals as points, genes as parameter specifics, gene pools, breeding methods and cost functions as objects passed between searches) follows the optimizer Joshua Tanner wrote for his dissertation research, rebuilt from his description.
