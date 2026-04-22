from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    return data or {}


def load_runs(path: str | Path) -> Dict[str, Dict[str, Any]]:
    data = load_yaml(path)
    runs = data.get("runs")
    if not isinstance(runs, dict):
        raise ValueError(f"{path} must contain a top-level 'runs' mapping.")
    return runs


def iter_runs(runs_config: str | Path, run: Optional[str] = None, enabled_only: bool = True) -> Iterable[tuple[str, Dict[str, Any]]]:
    runs = load_runs(runs_config)
    if run is not None:
        run = str(run).zfill(6)
        if run not in runs:
            raise KeyError(f"Run {run} not found in {runs_config}")
        yield run, runs[run]
        return
    for run_number, meta in runs.items():
        if enabled_only and not bool(meta.get("enabled", True)):
            continue
        yield str(run_number).zfill(6), meta


def repo_path(path: str | Path, repo_root: str | Path | None = None) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    repo_root = Path.cwd() if repo_root is None else Path(repo_root)
    return repo_root / path
