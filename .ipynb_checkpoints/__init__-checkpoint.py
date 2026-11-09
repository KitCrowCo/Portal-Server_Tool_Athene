"""Athene - goal-seeking search over parameter spaces with stochastic evaluation. AGPL-3.0-or-later.
genes: what a point is made of. goals: what better means. ensemble: one point over many runs, with common random numbers and a cache. search: GA, tabu, annealing, chains. mc: summaries and sensitivity."""
from .genes import Gene, FloatGene, IntGene, ChoiceGene, BoolGene, PermGene, SubsetGene, GenePool
from .goals import Goal, CostFunction, stat
from .ensemble import Ensemble, pkey
from .search import Individual, Population, Search, GA, Tabu, Anneal, chain
from .mc import summarize, sensitivity, spearman

__version__ = "0.1.0"
