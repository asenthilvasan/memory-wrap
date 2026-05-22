"""Aggregate per-cell ablation results from train.py log files.

Reads every ``NN_*.txt`` log in a directory (the convention used by
``paper/config/run_*_ablation*.sh`` scripts), pulls the FINAL "Run:..."
summary line printed by ``paper/train.py`` for each, and prints a results
table in plain text (default), markdown, or CSV.

The summary line format we parse is fixed by ``paper/train.py:284``:

    Run:15 | Best Loss:0.4567 | Accuracy 84.90 | Last Loss: Accuracy:|
        Mean Accuracy:84.90 | Std Dev Accuracy:0.48  T:2.34min  E:0.12

We take the LAST such line in each log because train.py prints one per run
and the final one carries the running mean/std over all completed runs.

Usage:
    # Default: pretty aligned plain-text table to stdout
    python -u scripts/extract_ablation_results.py /root/svhn_run_500/logs

    # Markdown table (good for pasting into analysis_*.md files)
    python -u scripts/extract_ablation_results.py /root/svhn_run_500/logs \\
        --format=markdown

    # CSV to a file
    python -u scripts/extract_ablation_results.py /root/svhn_run_500/logs \\
        --format=csv --out /root/svhn_run_500/logs/results.csv
"""
import argparse
import glob
import os
import re
import sys


# Match the final Run summary line. Allow optional leading whitespace and
# tolerate variations in spacing between fields. We capture run index, mean
# accuracy and std dev accuracy as floats.
_RUN_LINE_RE = re.compile(
    r"Run:\s*(\d+)\s*\|.*?Mean\s+Accuracy:\s*([0-9]+\.[0-9]+)\s*\|"
    r"\s*Std\s+Dev\s+Accuracy:\s*([0-9]+\.[0-9]+)"
)


# Pretty cell labels keyed by the leading two-digit prefix of the log
# filename. The prefixes match the cell numbers in the ablation shell
# scripts (run_svhn_ablation*.sh, run_cinic_ablation.sh).
_CELL_LABELS = {
    "01": "Scratch + Linear",
    "02": "Scratch + MW",
    "03": "SupCon + Linear (frozen)",
    "04": "SupCon + Linear (fine-tune)",
    "05": "SupCon + MW (frozen)",
    "06": "SupCon + MW (fine-tune)",
}


def parse_log(path):
    """Return (last_run_idx, mean_acc, std_acc) from a single log file.

    Returns ``None`` if no Run summary line was found (e.g. the cell crashed
    before completing run 1, or the log is for a pretrain step rather than a
    train.py run).
    """
    last = None
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            m = _RUN_LINE_RE.search(line)
            if m:
                last = (int(m.group(1)), float(m.group(2)), float(m.group(3)))
    return last


def cell_label(filename):
    """Map ``03_supcon_linear_frozen.txt`` -> ``SupCon + Linear (frozen)``.

    Falls back to the raw stem if the prefix isn't recognised so unexpected
    files (e.g. an experimenter's extra cell) still appear in the table.
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    prefix = stem[:2]
    if prefix in _CELL_LABELS:
        return _CELL_LABELS[prefix]
    return stem


def collect_rows(log_dir, pattern):
    """Walk a logs directory and return one row per matched log file.

    Each row is a dict so format-specific renderers can pick the columns
    they care about without re-parsing.
    """
    paths = sorted(glob.glob(os.path.join(log_dir, pattern)))
    rows = []
    for p in paths:
        parsed = parse_log(p)
        row = {
            "file": os.path.basename(p),
            "label": cell_label(p),
            "runs": None,
            "mean": None,
            "std": None,
            "ok": parsed is not None,
        }
        if parsed is not None:
            row["runs"], row["mean"], row["std"] = parsed
        rows.append(row)
    return rows


def render_plain(rows, log_dir):
    """Aligned plain-text table; the default human-readable format.

    Columns: Cell | Runs | Test Accuracy. We omit the log filename here
    because the cell label already encodes the same information and the
    extra column makes the table painfully wide on a 100-col terminal.
    Use --format=markdown if you want filenames preserved for citation.
    """
    successful = [r for r in rows if r["ok"]]
    best_mean = max((r["mean"] for r in successful), default=None)

    # Compute column widths from the data so the table auto-fits.
    label_w = max((len(r["label"]) for r in rows), default=4)
    label_w = max(label_w, len("Cell"))
    runs_w = max(len("Runs"), 4)
    acc_w = max(len("Test Accuracy"), len("99.99 \u00b1 9.99"))
    # Trailing column for the "<- best" marker; empty for non-best rows.
    mark_w = len("  <- best")

    sep = "-" * (label_w + 2 + runs_w + 2 + acc_w + mark_w)
    out = []
    out.append(f"\nResults from: {log_dir}")
    out.append(f"Cells found:  {len(rows)} ({len(successful)} completed, "
               f"{len(rows) - len(successful)} incomplete)\n")
    out.append(sep)
    out.append(f"{'Cell':<{label_w}}  {'Runs':>{runs_w}}  {'Test Accuracy':>{acc_w}}")
    out.append(sep)
    for r in rows:
        if not r["ok"]:
            line = (f"{r['label']:<{label_w}}  "
                    f"{'-':>{runs_w}}  "
                    f"{'(no Run summary)':>{acc_w}}")
        else:
            acc = f"{r['mean']:.2f} \u00b1 {r['std']:.2f}"
            mark = "  <- best" if r["mean"] == best_mean else ""
            line = (f"{r['label']:<{label_w}}  "
                    f"{r['runs']:>{runs_w}}  "
                    f"{acc:>{acc_w}}{mark}")
        out.append(line)
    out.append(sep)
    return "\n".join(out) + "\n"


def render_markdown(rows):
    """GitHub-flavored markdown table; use when pasting into a .md file."""
    lines = []
    lines.append("| Cell | Log file | Runs | Test Acc (mean \u00b1 std) |")
    lines.append("|------|----------|------|---------------------------|")
    for r in rows:
        if not r["ok"]:
            lines.append(f"| {r['label']} | `{r['file']}` | \u2014 | "
                         f"(no Run summary line found) |")
        else:
            lines.append(f"| {r['label']} | `{r['file']}` | {r['runs']} | "
                         f"{r['mean']:.2f} \u00b1 {r['std']:.2f} |")
    return "\n".join(lines) + "\n"


def render_csv(rows):
    """CSV: ``cell,log_file,runs,mean_acc,std_acc`` (one row per log)."""
    out = ["cell,log_file,runs,mean_acc,std_acc"]
    for r in rows:
        if not r["ok"]:
            out.append(f"\"{r['label']}\",{r['file']},,,")
        else:
            out.append(f"\"{r['label']}\",{r['file']},{r['runs']},"
                       f"{r['mean']:.4f},{r['std']:.4f}")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log_dir", help="Directory containing NN_*.txt train.py logs")
    ap.add_argument("--format", choices=["plain", "markdown", "csv"],
                    default="plain",
                    help="Output format (default: plain aligned text table)")
    ap.add_argument("--out", default=None,
                    help="Write to this file instead of stdout")
    ap.add_argument("--pattern", default="[0-9][0-9]_*.txt",
                    help="Glob pattern for log files (default: NN_*.txt)")
    args = ap.parse_args()

    rows = collect_rows(args.log_dir, args.pattern)
    if not rows:
        print(f"No logs matching {args.pattern} in {args.log_dir}",
              file=sys.stderr)
        sys.exit(1)

    if args.format == "plain":
        text = render_plain(rows, args.log_dir)
    elif args.format == "markdown":
        text = render_markdown(rows)
    else:  # csv
        text = render_csv(rows)

    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
        print(f"Wrote {args.format} table to: {args.out}")
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
