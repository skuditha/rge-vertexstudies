#!/usr/bin/env python3
from __future__ import annotations
import argparse
import subprocess
from pathlib import Path
from rge_vertex.io.configs import iter_runs, repo_path


def make_file_list(run: str, meta: dict, output_dir: Path, recursive: bool = False) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    list_path = output_dir / f"{run}.list"
    if meta.get("input_list"):
        return Path(meta["input_list"])
    if meta.get("input_files"):
        files = [Path(p) for p in meta["input_files"]]
    else:
        input_dir = Path(meta["input_dir"])
        files = sorted(input_dir.glob("**/*.hipo" if recursive else "*.hipo"))
    if not files:
        raise RuntimeError(f"No HIPO files found for run {run}. Check input_dir/input_files in runs.yaml.")
    with list_path.open("w", encoding="utf-8") as fout:
        for path in files:
            fout.write(str(path) + "\n")
    return list_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the C++ HIPO-to-ROOT extractor.")
    parser.add_argument("--runs-config", default="configs/runs.yaml")
    parser.add_argument("--run", default=None)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--extractor", default="cpp/build/rge_hipo_to_root")
    parser.add_argument("--file-list-dir", default="outputs/file_lists")
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    repo_root = Path(args.repo_root)
    extractor = repo_path(args.extractor, repo_root)
    file_list_dir = repo_path(args.file_list_dir, repo_root)
    if not extractor.exists() and not args.dry_run:
        raise FileNotFoundError(f"Extractor not found: {extractor}. Build it with cpp/scripts/build.sh")
    for run, meta in iter_runs(args.runs_config, run=args.run, enabled_only=True):
        output_root = repo_path(meta.get("output_root", f"outputs/ntuples/{run}.root"), repo_root)
        output_root.parent.mkdir(parents=True, exist_ok=True)
        input_list = make_file_list(run, meta, file_list_dir, recursive=args.recursive)
        cmd = [str(extractor), "--run", run, "--input-list", str(input_list), "--output", str(output_root), "--max-events", str(args.max_events), "--max-files", str(args.max_files)]
        print(" ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
