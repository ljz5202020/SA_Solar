"""
Record project progress and refresh managed sections in ROADMAP.md / STATUS.md.

Typical usage:
  python scripts/progress_tracker.py --summary "Added feature X"
  python scripts/progress_tracker.py --from-last-commit --skip-duplicates
  python scripts/progress_tracker.py --next-focus "Clean up root-level file structure"
"""

from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PROGRESS_LOG_PATH = ROOT / "docs" / "progress_log.md"
ROADMAP_PATH = ROOT / "ROADMAP.md"
STATUS_PATH = ROOT / "STATUS.md"
REGIONS_YAML_PATH = ROOT / "configs" / "datasets" / "regions.yaml"

STATUS_START = "<!-- progress:status:start -->"
STATUS_END = "<!-- progress:status:end -->"
ROADMAP_START = "<!-- progress:roadmap:start -->"
ROADMAP_END = "<!-- progress:roadmap:end -->"
STATE_START = "<!-- progress:state:start -->"
STATE_END = "<!-- progress:state:end -->"
RESULTS_START = "<!-- progress:results:start -->"
RESULTS_END = "<!-- progress:results:end -->"

DEFAULT_NEXT_FOCUS = [
    "Repository structure cleanup: reduce root-level script clutter and group workflows by purpose.",
    "Export reviewed keep/exclude decisions into a reusable grid manifest for later tile downloads.",
]


@dataclass(frozen=True)
class ProgressEntry:
    timestamp: str
    source: str
    summary: str

    @property
    def day(self) -> str:
        return self.timestamp[:10]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_progress_entries() -> list[ProgressEntry]:
    if not PROGRESS_LOG_PATH.exists():
        return []

    pattern = re.compile(r"^- ([^|]+) \| ([^|]+) \| (.+)$")
    entries: list[ProgressEntry] = []
    for line in PROGRESS_LOG_PATH.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        entries.append(
            ProgressEntry(
                timestamp=match.group(1).strip(),
                source=match.group(2).strip(),
                summary=match.group(3).strip(),
            )
        )
    return entries


def save_progress_log(entries: list[ProgressEntry], next_focus: list[str]) -> None:
    PROGRESS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Progress Log",
        "",
        "This file is updated by `scripts/progress_tracker.py` and the optional git `post-commit` hook.",
        "",
        "## Current Focus",
    ]
    for item in next_focus:
        lines.append(f"- {item}")
    lines.extend(["", "## Entries"])
    for entry in sorted(entries, key=lambda item: item.timestamp, reverse=True):
        lines.append(f"- {entry.timestamp} | {entry.source} | {entry.summary}")
    PROGRESS_LOG_PATH.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def get_last_commit_entry() -> ProgressEntry:
    output = subprocess.check_output(
        ["git", "log", "-1", "--pretty=format:%H%n%s"],
        cwd=ROOT,
        text=True,
    )
    commit_hash, summary = output.splitlines()[:2]
    return ProgressEntry(
        timestamp=utc_now_iso(),
        source=f"commit:{commit_hash[:8]}",
        summary=summary.strip(),
    )


def last_recorded_next_focus() -> list[str]:
    if not PROGRESS_LOG_PATH.exists():
        return list(DEFAULT_NEXT_FOCUS)

    lines = PROGRESS_LOG_PATH.read_text(encoding="utf-8").splitlines()
    next_focus: list[str] = []
    in_focus = False
    for line in lines:
        stripped = line.strip()
        if stripped == "## Current Focus":
            in_focus = True
            continue
        if in_focus and stripped.startswith("## "):
            break
        if in_focus and stripped.startswith("- "):
            next_focus.append(stripped[2:].strip())
    return next_focus or list(DEFAULT_NEXT_FOCUS)


def render_recent_updates(entries: list[ProgressEntry], limit: int = 8) -> list[str]:
    return [f"- {entry.day}: {entry.summary}" for entry in sorted(entries, key=lambda item: item.timestamp, reverse=True)[:limit]]


def replace_or_insert_block(
    text: str,
    start_marker: str,
    end_marker: str,
    block_title: str,
    body_lines: list[str],
) -> str:
    block = "\n".join(
        [block_title, start_marker, *body_lines, end_marker]
    ).rstrip()

    if start_marker in text and end_marker in text:
        pattern = re.compile(
            rf"{re.escape(block_title)}\n{re.escape(start_marker)}.*?{re.escape(end_marker)}",
            re.DOTALL,
        )
        return pattern.sub(block, text, count=1)

    anchor = re.search(r"^## V0:.*$", text, flags=re.MULTILINE)
    if anchor:
        return text[: anchor.start()] + block + "\n\n" + text[anchor.start() :]
    return text.rstrip() + "\n\n" + block + "\n"


def update_status(entries: list[ProgressEntry], next_focus: list[str]) -> None:
    text = STATUS_PATH.read_text(encoding="utf-8")
    text = re.sub(r"\*\*Last Updated\*\*: .*", f"**Last Updated**: {datetime.now(timezone.utc).date().isoformat()}", text, count=1)

    body_lines = ["### Recent Updates", *render_recent_updates(entries), "", "### Current Ops Focus"]
    body_lines.extend(f"- {item}" for item in next_focus)

    text = replace_or_insert_block(
        text=text,
        start_marker=STATUS_START,
        end_marker=STATUS_END,
        block_title="## Progress Tracker",
        body_lines=body_lines,
    )
    STATUS_PATH.write_text(text, encoding="utf-8")


def update_roadmap(entries: list[ProgressEntry], next_focus: list[str]) -> None:
    text = ROADMAP_PATH.read_text(encoding="utf-8")
    body_lines = ["### Recently Completed", *render_recent_updates(entries), "", "### Next Up"]
    body_lines.extend(f"- {item}" for item in next_focus)

    text = replace_or_insert_block(
        text=text,
        start_marker=ROADMAP_START,
        end_marker=ROADMAP_END,
        block_title="## Execution Track",
        body_lines=body_lines,
    )
    ROADMAP_PATH.write_text(text, encoding="utf-8")


def _load_grid_ids() -> list[str]:
    """Read grid IDs from regions.yaml (all regions)."""
    import yaml

    if not REGIONS_YAML_PATH.exists():
        return []
    data = yaml.safe_load(REGIONS_YAML_PATH.read_text(encoding="utf-8"))
    grid_ids: list[str] = []
    for region in (data.get("regions") or {}).values():
        for gid in (region.get("grids") or {}):
            grid_ids.append(gid)
    return grid_ids


def _read_grid_metrics(grid_id: str) -> dict[str, str] | None:
    """Read presence_metrics.csv for a grid. Returns None if not available."""
    csv_path = ROOT / "results" / grid_id / "presence_metrics.csv"
    if not csv_path.exists():
        return None
    import csv

    with csv_path.open(encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        row = next(reader, None)
    if row is None:
        return None
    return {
        "precision": f"{float(row['precision']):.4f}",
        "recall": f"{float(row['recall']):.4f}",
        "f1": f"{float(row['f1']):.4f}",
    }


def refresh_results() -> None:
    """Read metrics from results/ and update the results table in STATUS.md."""
    grid_ids = _load_grid_ids()
    if not grid_ids:
        print("[SKIP] No grids found in regions.yaml")
        return

    table_lines = [
        "| Grid | Precision | Recall | F1 | Status |",
        "|------|-----------|--------|----|--------|",
    ]
    found = 0
    for gid in grid_ids:
        metrics = _read_grid_metrics(gid)
        if metrics is None:
            table_lines.append(f"| {gid} | — | — | — | not evaluated |")
        else:
            found += 1
            table_lines.append(
                f"| {gid} | {metrics['precision']} | {metrics['recall']} | {metrics['f1']} | evaluated |"
            )

    text = STATUS_PATH.read_text(encoding="utf-8")
    text = replace_or_insert_block(
        text=text,
        start_marker=RESULTS_START,
        end_marker=RESULTS_END,
        block_title="## Evaluation Results Summary",
        body_lines=table_lines,
    )
    STATUS_PATH.write_text(text, encoding="utf-8")
    print(f"[RESULTS] {found}/{len(grid_ids)} grids with metrics → {STATUS_PATH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record progress and refresh roadmap/status")
    parser.add_argument("--summary", help="Human-written progress summary")
    parser.add_argument("--source", default="manual", help="Entry source label for manual records")
    parser.add_argument("--from-last-commit", action="store_true", help="Record the latest git commit subject")
    parser.add_argument("--skip-duplicates", action="store_true", help="Skip if the same source+summary is already recorded")
    parser.add_argument(
        "--next-focus",
        action="append",
        default=None,
        help="Replace the current next-focus list. Pass multiple times for multiple bullets.",
    )
    parser.add_argument(
        "--refresh-results",
        action="store_true",
        help="Read results/<grid>/presence_metrics.csv and refresh the results table in STATUS.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entries = load_progress_entries()
    next_focus = args.next_focus if args.next_focus is not None else last_recorded_next_focus()

    new_entry: ProgressEntry | None = None
    if args.from_last_commit:
        new_entry = get_last_commit_entry()
    elif args.summary:
        new_entry = ProgressEntry(timestamp=utc_now_iso(), source=args.source.strip(), summary=args.summary.strip())

    if new_entry is not None:
        duplicate = any(
            entry.source == new_entry.source and entry.summary == new_entry.summary
            for entry in entries
        )
        if not (args.skip_duplicates and duplicate):
            entries.append(new_entry)

    save_progress_log(entries, next_focus)
    update_status(entries, next_focus)
    update_roadmap(entries, next_focus)

    if args.refresh_results:
        refresh_results()

    if new_entry is not None:
        print(f"[RECORDED] {new_entry.summary}")
    print(f"[LOG] {PROGRESS_LOG_PATH}")
    print(f"[STATUS] {STATUS_PATH}")
    print(f"[ROADMAP] {ROADMAP_PATH}")


if __name__ == "__main__":
    main()
