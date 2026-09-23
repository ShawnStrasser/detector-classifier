"""Import path helper for the research scripts.

The research code is a flat namespace of loose scripts, exactly as it was while the
study ran.  Importing this module puts every folder under `research/code/` **and** the
production folder `model/` on `sys.path`, so a research script can say
`from common import DC_WORK` (research) and `from features import build_window`
(production) without knowing where either file sits.

Every research script starts with the same three lines::

    import sys; from pathlib import Path
    sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                                if p.name == "code")))
    import rpath  # noqa: F401
"""
from __future__ import annotations

import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent
REPO = CODE.parents[1]
MODEL = REPO / "model"

_dirs = [CODE, MODEL] + sorted(p for p in CODE.rglob("*")
                               if p.is_dir() and p.name != "__pycache__")
for _d in reversed(_dirs):          # so CODE and MODEL end up first
    _s = str(_d)
    # re-insert rather than skip: the caller has already put CODE on the path itself
    # (that is how this module was found), and merely skipping it would leave CODE
    # *behind* MODEL, so `import common` would find the production module instead of
    # the research one.
    if _s in sys.path:
        sys.path.remove(_s)
    sys.path.insert(0, _s)
