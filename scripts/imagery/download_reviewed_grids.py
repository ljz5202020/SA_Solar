"""
Download full-resolution tiles for grids selected in a preview-review batch.

Examples:
  python scripts/imagery/download_reviewed_grids.py \
    --batch-dir results/grid_previews/batch_001

  python scripts/imagery/download_reviewed_grids.py \
    --batch-dir results/grid_previews/batch_001 \
    --decision keep review \
    --dry
"""

from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.imagery.download_tiles import download_grid

DEFAULT_DECISIONS = ("keep",)


def load_grid_ids(batch_dir: Path, decisions: tuple[str, ...]) -> list[str]:
    decisions_path = batch_dir / "grid_review_decisions.csv"
    if not decisions_path.exists():
        raise FileNotFoundError(f"review decisions not found: {decisions_path}")

    wanted = {decision.strip().lower() for decision in decisions if decision.strip()}
    if not wanted:
        raise ValueError("at least one --decision value is required")

    grid_ids: list[str] = []
    seen: set[str] = set()
    with decisions_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            decision = str(row.get("decision", "")).strip().lower()
            grid_id = str(row.get("grid_id", "")).strip().upper()
            if decision not in wanted or not grid_id or grid_id in seen:
                continue
            seen.add(grid_id)
            grid_ids.append(grid_id)
    return grid_ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download full-resolution tiles for grids chosen in browser review output"
    )
    parser.add_argument(
        "--batch-dir",
        required=True,
        type=Path,
        help="Preview batch directory, e.g. results/grid_previews/batch_001",
    )
    parser.add_argument(
        "--decision",
        nargs="+",
        default=list(DEFAULT_DECISIONS),
        help="Decision labels to include, e.g. keep review",
    )
    parser.add_argument(
        "--dry",
        action="store_true",
        help="Only print the selected grids without downloading tiles",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of grids to download in parallel",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    grid_ids = load_grid_ids(args.batch_dir, tuple(args.decision))
    print(f"[INFO] batch_dir={args.batch_dir}")
    print(f"[INFO] selected_grids={len(grid_ids)}")
    if not grid_ids:
        print("[INFO] no matching grids found")
        return

    print("[INFO] grid_ids=" + ", ".join(grid_ids))
    if args.dry:
        return

    workers = max(1, min(int(args.workers), len(grid_ids)))
    print(f"[INFO] workers={workers}")

    def run_one(job: tuple[int, str]) -> str:
        idx, grid_id = job
        print(f"\n=== [{idx}/{len(grid_ids)}] {grid_id} ===", flush=True)
        download_grid(grid_id, dry_run=False)
        return grid_id

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_one, job) for job in enumerate(grid_ids, start=1)]
        for future in as_completed(futures):
            grid_id = future.result()
            print(f"[DONE] grid={grid_id}", flush=True)


if __name__ == "__main__":
    main()
