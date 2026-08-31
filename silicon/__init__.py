"""Shared path constants for the supplement.

ROOT     repository root (the directory holding run.py and analyze.py)
DATA     shipped result files and item lists
RESULTS  outputs of fresh runs (created on demand)
PROJECT  root of the original experiment tree, for the few inputs that are
         not redistributed (override with the PROJECT_ROOT env var)
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
PROJECT = Path(os.environ.get("PROJECT_ROOT", "~/project")).expanduser()


def require_env(name):
    """Exit with a readable message when a required key is missing."""
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"{name} is not set; export it first (see README, Environment).")
    return value


def results_dir(name):
    """Return RESULTS/name, creating it if needed."""
    d = RESULTS / name
    d.mkdir(parents=True, exist_ok=True)
    return d
