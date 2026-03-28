#!/usr/bin/env python3
import argparse
import re
from pathlib import Path


SECTION_BY_DATASET = {
    "livejournal_16p": "## LiveJournal 16p",
    "twitter_16p": "## Twitter 16p",
    "freebase86m_16p": "## Freebase86M 16p",
}


def parse_summary(path: Path):
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def trim_cell(value: str) -> str:
    return value.strip()


def parse_table_row(line: str):
    parts = line.strip().split("|")
    return [trim_cell(part) for part in parts[1:-1]]


def render_table_row(cells):
    return "| " + " | ".join(cells) + " |"


def fmt_float(value, digits):
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def fmt_ms(value, digits=2):
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f} ms"


def none_if_missing(summary, key):
    raw = summary.get(key)
    if raw is None or raw == "" or raw.lower() == "none":
        return None
    return float(raw)


def update_row(cells, summary, args):
    row_label = f"`{args.row}`"
    metrics = {
        "Row": row_label,
        "Branch": f"`{args.branch}`",
        "Config": f"`{args.config}`",
        "Flags Enabled": f"`{args.flags}`" if args.flags else "`none`",
        "Epochs": f"`{args.epochs}`",
        "Avg Epoch Runtime": fmt_ms(none_if_missing(summary, "avg_epoch_runtime_ms"), 2),
        "Avg Edges per Second": fmt_float(none_if_missing(summary, "avg_edges_per_second"), 2),
        "Avg Inter-Epoch Gap": fmt_ms(none_if_missing(summary, "avg_inter_epoch_gap_ms"), 2),
        "Avg swap_count": fmt_float(none_if_missing(summary, "avg_swap_count"), 1),
        "Avg swap_barrier_wait_ms": fmt_float(none_if_missing(summary, "avg_swap_barrier_wait_ms"), 4),
        "Avg swap_update_ms": fmt_float(none_if_missing(summary, "avg_swap_update_ms"), 4),
        "Avg swap_rebuild_ms": fmt_float(none_if_missing(summary, "avg_swap_rebuild_ms"), 4),
        "Avg swap_sync_wait_ms": fmt_float(none_if_missing(summary, "avg_swap_sync_wait_ms"), 4),
        "MRR": "n/a",
        "Hits@1": "n/a",
        "Hits@3": "n/a",
        "Hits@10": "n/a",
        "Eval Notes": args.eval_notes,
        "Train Log": f"`{args.train_log}`",
        "Eval Log": f"`{args.eval_log}`" if args.eval_log != "n/a" else "`n/a`",
        "Notes": args.notes,
    }
    new_cells = [metrics.get(header, value) for header, value in zip(args.headers, cells)]

    if "Branch" in args.headers:
        branch_idx = args.headers.index("Branch")
        marker_match = re.match(r"(<!--\s*row:\s*.*?-->\s*)", cells[branch_idx])
        marker = marker_match.group(1) if marker_match else f"<!-- row: {args.row} --> "
        new_cells[branch_idx] = marker + new_cells[branch_idx].lstrip()

    return new_cells


def main():
    parser = argparse.ArgumentParser(description="Update a row in single_gpu_ablation_results_template.md")
    parser.add_argument("--md", required=True)
    parser.add_argument("--dataset", required=True, choices=sorted(SECTION_BY_DATASET))
    parser.add_argument("--row", required=True)
    parser.add_argument("--branch", default="main")
    parser.add_argument("--config", required=True)
    parser.add_argument("--flags", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--train-log", required=True)
    parser.add_argument("--eval-log", default="n/a")
    parser.add_argument("--eval-notes", default="train only; eval skipped")
    parser.add_argument("--notes", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    md_path = Path(args.md)
    lines = md_path.read_text(encoding="utf-8").splitlines()
    summary = parse_summary(Path(args.summary))

    section_header = SECTION_BY_DATASET[args.dataset]
    try:
        section_start = lines.index(section_header)
    except ValueError as exc:
        raise SystemExit(f"missing section: {section_header}") from exc

    section_end = len(lines)
    for idx in range(section_start + 1, len(lines)):
        if lines[idx].startswith("## "):
            section_end = idx
            break

    header_idx = None
    for idx in range(section_start + 1, section_end):
        if lines[idx].startswith("| ") and "Branch" in lines[idx] and "Avg Epoch Runtime" in lines[idx]:
            header_idx = idx
            break
    if header_idx is None:
        raise SystemExit(f"missing table header in section {section_header}")

    args.headers = parse_table_row(lines[header_idx])

    row_idx = None
    marker = f"<!-- row: {args.row} -->"
    needle = f"| `{args.row}` |"
    for idx in range(header_idx + 2, section_end):
        if marker in lines[idx] or lines[idx].startswith(needle):
            row_idx = idx
            break
    if row_idx is None:
        raise SystemExit(f"missing row {args.row} in section {section_header}")

    current_cells = parse_table_row(lines[row_idx])
    new_cells = update_row(current_cells, summary, args)
    lines[row_idx] = render_table_row(new_cells)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
