"""Athene - goal-seeking search over parameter spaces with stochastic evaluation. AGPL-3.0-or-later.
As a Portal Server tool (tools/athene/) this is the file the portal loads; every tool and module reaches it as ENV["tools"]["athene"]. It has no page and keeps no data: whoever uses it keeps results under its own data folder.
genes: what a point is made of. goals: what better means. ensemble: one point over many runs, with common random numbers and a cache. search: GA, tabu, annealing, chains. mc: summaries and sensitivity."""
from .genes import Gene, FloatGene, IntGene, ChoiceGene, BoolGene, PermGene, SubsetGene, GenePool
from .goals import Goal, CostFunction, stat
from .ensemble import Ensemble, pkey
from .search import Individual, Population, Search, GA, Tabu, Anneal, chain, pareto
from .mc import summarize, sensitivity, spearman

TOOL_META = {"label": "Athene", "icon": "&#x1F989;", "description": "Goal-seeking search (GA, tabu, annealing) with ensemble evaluation, for any tool or module"}
__version__ = "0.1.0"
