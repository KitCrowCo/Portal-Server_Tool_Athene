"""Athene as a Portal Server tool: tools/athene/ is this repository, and this file is what the portal loads. It has no page and no data of its own; it hands its search library to every tool and module that asks for ENV["tools"]["athene"].
Whoever uses it keeps its own results under its own data folder, so two users of Athene never share or overwrite each other's files."""
from . import *   # everything the package exports (__init__.py)

TOOL_META = {"label": "Athene", "icon": "&#x1F989;", "description": "Goal-seeking search (GA, tabu, annealing) with ensemble evaluation, for any tool or module"}
