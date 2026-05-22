"""Aggregate per-cell ablation results from train.py log files.

Reads every ``NN_*.txt`` log in a directory (the convention used by
``paper/config/run_*_ablation*.sh`` scripts), pulls the FINAL "Run:..."
summary line printed by ``paper/train.py`` for each, and prints a markdown
table plus an optional CSV of mean/std test accuracy across runs.

The summary line format we parse is fixed by ``paper/train.py:284``:

    Run:15 | Best Loss:0.4567 | Accuracy 84.90 | Last Loss: Accuracy:|
        Mean Accuracy:84.90 | Std Dev Accuracy:0.48  T:2.34min  E:0.12

We take the LAST such line in each log because train.py prints one per run
and the final one carries the running mean/std over all completed runs.

Usage:
    python -u scripts/extract_ablation_results.py /root/svhn_run_500/logs
    python -u scripts/extract_ablation_results.py /root/svhn_run_500/logs \\
        --csv /root/svhn_run_500/logs/results.csv
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log_dir", help="Directory containing NN_*.txt train.py logs")
    ap.add_argument("--csv", default=None,
                    help="Optional path to also write a CSV of the same data")
    ap.add_argument("--pattern", default="[0-9][0-9]_*.txt",
                    help="Glob pattern for log files (default: NN_*.txt)")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.log_dir, args.pattern)))
    if not paths:
        print(f"No logs matching {args.pattern} in {args.log_dir}", file=sys.stderr)
        sys.exit(1)

    rows = []  # (filename, label, runs, mean, std) or (filename, label, None, None, None)
    for p in paths:
        parsed = parse_log(p)
        label = cell_label(p)
        fname = os.path.basename(p)
        if parsed is None:
            rows.append((fname, label, None, None, None))
        else:
            runs, mean, std = parsed
            rows.append((fname, label, runs, mean, std))

    # Markdown table to stdout.
    print()
    print("| Cell | Log file | Runs | Test Acc (mean ± std) |")
    print("|------|----------|------|------------------------|")
    for fname, label, runs, mean, std in rows:
        if runs is None:
            print(f"| {label} | `{fname}` | — | (no Run summary line found) |")
        else:
            print(f"| {label} | `{fname}` | {runs} | {mean:.2f} ± {std:.2f} |")
    print()

    if args.csv:
        with open(args.csv, "w") as fh:
            fh.write("cell,log_file,runs,mean_acc,std_acc\n")
            for fname, label, runs, mean, std in rows:
                if runs is None:
                    fh.write(f"\"{label}\",{fname},,,\n")
                else:
                    fh.write(f"\"{label}\",{fname},{runs},{mean:.4f},{std:.4f}\n")
        print(f"Wrote CSV: {args.csv}")


if __name__ == "__main__":
    main()
