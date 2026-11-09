"""Loads this repository as the package athene whatever its folder is called (Athene/ in a checkout, tools/athene/ in a portal)."""
import sys, pathlib, importlib.util

def athene():
    root = pathlib.Path(__file__).resolve().parents[1]
    if "athene" in sys.modules: return sys.modules["athene"]
    spec = importlib.util.spec_from_file_location("athene", root / "__init__.py", submodule_search_locations=[str(root)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["athene"] = mod
    spec.loader.exec_module(mod)
    return mod
