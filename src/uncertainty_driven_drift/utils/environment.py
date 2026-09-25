"""Capture the software environment a run was produced in.

Every run already records its *configuration* — components, hyperparameters,
seeds — which is enough to re-issue the same command. It is not enough to
reproduce the same numbers, because the detector implementations live in
``river``: an ADWIN or KSWIN change between minor versions moves the alarm
counts without touching a single line of this repo or its configs.

So stamp the versions that actually decide the output into ``metrics.json``.
This costs microseconds and turns "these numbers don't match" from a mystery
into a diff.

Deliberately best-effort: a missing optional package or a repo that is not a git
checkout records ``None`` rather than failing a run that is otherwise fine.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Optional

#: Packages whose version can change results. ``river`` is the important one —
#: it owns the drift detectors.
_PACKAGES = ("torch", "torchvision", "river", "numpy", "scipy", "sklearn", "matplotlib")


def _version(name: str) -> Optional[str]:
    try:
        return getattr(import_module(name), "__version__", None)
    except Exception:
        return None


def _git_commit() -> Optional[str]:
    """Short commit of the repo this code lives in, with a dirty marker."""
    repo = Path(__file__).resolve().parents[3]
    try:
        rev = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if rev.returncode != 0:
            return None
        commit = rev.stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        )
        if dirty.returncode == 0 and dirty.stdout.strip():
            commit += "-dirty"
        return commit
    except Exception:
        return None


def _gpu() -> Optional[Dict[str, Any]]:
    """Device name + compute capability, which gate whether CUDA kernels exist."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        major, minor = torch.cuda.get_device_capability()
        return {
            "name": torch.cuda.get_device_name(),
            "capability": f"sm_{major}{minor}",
            "torch_arch_list": list(torch.cuda.get_arch_list()),
        }
    except Exception:
        return None


def capture() -> Dict[str, Any]:
    """Environment snapshot for ``metrics.json``."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "packages": {name: _version(name) for name in _PACKAGES},
        "gpu": _gpu(),
    }
