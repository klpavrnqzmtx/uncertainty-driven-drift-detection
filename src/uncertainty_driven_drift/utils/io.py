"""Small I/O helpers used by the runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as fh:
        json.dump(data, fh, indent=2, default=_json_default)


def _json_default(obj: Any) -> Any:
    # numpy scalars / arrays round-trip through list/float
    try:
        import numpy as np
    except ImportError:
        raise TypeError(f"Cannot serialize {type(obj).__name__}")
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Cannot serialize {type(obj).__name__}")
