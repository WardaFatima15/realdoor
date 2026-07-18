"""Load the organizer's shipped, unmodified starter code by file path.

We deliberately avoid adding ``realdoor-hackathon-starter-pack/starter`` to
``sys.path`` because that package is *also* named ``src`` and would collide
with this project's own ``src`` package. Loading the three shipped modules
directly from their file paths lets us reuse ``calculate.py``, ``rules.py``
and ``load_documents.py`` byte-for-byte -- untouched, as required -- without
any import collision or copy/paste fork.
"""
import importlib.util

from ._paths import STARTER_SRC


def _load(mod_name: str, filename: str):
    path = STARTER_SRC / filename
    spec = importlib.util.spec_from_file_location(f"realdoor_shipped_{mod_name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_calculate = _load("calculate", "calculate.py")
_rules = _load("rules", "rules.py")
_load_documents = _load("load_documents", "load_documents.py")

# Re-exported, unmodified shipped functions.
annualize = _calculate.annualize
compare_to_threshold = _calculate.compare_to_threshold
FREQUENCY = _calculate.FREQUENCY

load_rules = _rules.load_rules

load_gold = _load_documents.load_gold
validate_boxes = _load_documents.validate_boxes
