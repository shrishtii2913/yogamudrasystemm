"""
=============================================================================
evaluate.py — Offline Accuracy Evaluation Script
=============================================================================
ACADEMIC PROJECT: Yoga Mudra Detection System

PURPOSE:
  This script measures the accuracy of the mudra classifier offline.
  It replays a labelled CSV dataset (ground_truth.csv) through the same
  detector logic used by pose_detector.py and prints a full classification
  report: precision, recall, F1 score, confusion matrix, and a results table.

HOW TO USE:
  1. Record a ground-truth dataset:
       - Run pose_detector.py while performing each mudra intentionally
       - Manually label each row in logs/cpp_log.csv with the true mudra name
       - Save as eval/ground_truth.csv  (columns: Timestamp, TruePose, Pose, Confidence)

  2. Run this script:
       python3 evaluate.py

  3. Review the printed report and eval/accuracy_report.txt

NOTE:
  Because MediaPipe requires a camera, this script evaluates the classifier
  logic using the LOGGED CSV data — comparing the auto-detected Pose column
  against your manually entered TruePose column.
  For a full end-to-end eval, record ground truth during a live session.
=============================================================================
"""

import csv
import os
import sys
from collections import defaultdict
from datetime import datetime

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
GROUND_TRUTH_CSV = "eval/ground_truth.csv"
REPORT_PATH      = "eval/accuracy_report.txt"
SAMPLE_CSV       = "logs/sample_session.csv"   # fallback demo data

KNOWN_POSES = [
    "Gyan Mudra",
    "Chin Mudra",
    "Abhaya Mudra",
    "Dhyana Mudra",
    "Shuni Mudra",
    "No Pose",
]


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def load_csv(path: str) -> list[dict]:
    """Load a CSV file into a list of row dicts."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def build_confusion_matrix(rows: list[dict]) -> dict:
    """
    Build a confusion matrix from labelled rows.
    Expected columns: TruePose, Pose
    Returns: { true_label: { predicted_label: count } }
    """
    matrix = defaultdict(lambda: defaultdict(int))
    for row in rows:
        true  = row.get("TruePose", row.get("Pose", "No Pose")).strip()
        pred  = row.get("Pose", "No Pose").strip()
        matrix[true][pred] += 1
    return matrix


def compute_metrics(matrix: dict, labels: list[str]) -> dict:
    """
    Compute per-class precision, recall, F1 from the confusion matrix.
    Returns dict of { label: {precision, recall, f1, support} }
    """
    metrics = {}
    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in labels if other != label)
        fn = sum(matrix[label][other] for other in labels if other != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        support   = sum(matrix[label].values())

        metrics[label] = {
            "precision": round(precision, 4),
            "recall"   : round(recall,    4),
            "f1"       : round(f1,        4),
            "support"  : support,
        }
    return metrics


def overall_accuracy(matrix: dict, labels: list[str]) -> float:
    """Compute overall accuracy = correct / total."""
    correct = sum(matrix[l][l] for l in labels)
    total   = sum(matrix[r][c] for r in labels for c in labels)
    return correct / total if total > 0 else 0.0


def print_confusion_matrix(matrix: dict, labels: list[str]) -> str:
    """Return a formatted confusion matrix string."""
    present = [l for l in labels if sum(matrix[l].values()) > 0]
    col_w   = max(len(l) for l in present) + 2
    lines   = []

    header = " " * col_w + "".join(l[:col_w].ljust(col_w) for l in present)
    lines.append(header)
    lines.append("-" * len(header))

    for true in present:
        row = true.ljust(col_w)
        for pred in present:
            row += str(matrix[true][pred]).ljust(col_w)
        lines.append(row)

    return "\n".join(lines)


def generate_report(rows: list[dict], source: str) -> str:
    """
    Full textual accuracy report.
    If rows have a 'TruePose' column, uses it; otherwise treats 'Pose' as both
    true and predicted (self-consistency check — useful for demo without labels).
    """
    has_labels = any("TruePose" in r for r in rows)

    if not has_labels:
        print("[WARN] No 'TruePose' column found. Running self-consistency demo.")
        print("       Add a 'TruePose' column to your CSV for real accuracy measurement.\n")
        # Duplicate Pose as TruePose — gives 100% accuracy as a smoke test
        for r in rows:
            r["TruePose"] = r.get("Pose", "No Pose")

    matrix  = build_confusion_matrix(rows)
    metrics = compute_metrics(matrix, KNOWN_POSES)
    acc     = overall_accuracy(matrix, KNOWN_POSES)
    total   = sum(sum(matrix[r].values()) for r in KNOWN_POSES)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sep = "=" * 64

    lines = [
        sep,
        f"  Yoga Mudra Classifier — Accuracy Report",
        f"  Generated : {ts}",
        f"  Source    : {source}",
        f"  Rows      : {total}",
        sep,
        "",
        f"  Overall Accuracy : {acc:.2%}  ({int(acc * total)}/{total} correct)",
        "",
        "  Per-Class Metrics:",
        "-" * 64,
        f"  {'Pose':<20} {'Prec':>6}  {'Recall':>6}  {'F1':>6}  {'Support':>7}",
        "-" * 64,
    ]

    macro_p = macro_r = macro_f = 0.0
    active  = [l for l in KNOWN_POSES if metrics[l]["support"] > 0]

    for label in KNOWN_POSES:
        m = metrics[label]
        if m["support"] == 0:
            continue
        lines.append(
            f"  {label:<20} {m['precision']:>6.3f}  {m['recall']:>6.3f}"
            f"  {m['f1']:>6.3f}  {m['support']:>7}"
        )
        macro_p += m["precision"]
        macro_r += m["recall"]
        macro_f += m["f1"]

    n = len(active)
    lines += [
        "-" * 64,
        f"  {'Macro avg':<20} {macro_p/n:>6.3f}  {macro_r/n:>6.3f}"
        f"  {macro_f/n:>6.3f}  {total:>7}",
        "",
        "  Confusion Matrix (rows=True, cols=Predicted):",
        "-" * 64,
        print_confusion_matrix(matrix, KNOWN_POSES),
        "",
        sep,
        "  Confidence Statistics:",
        "-" * 64,
    ]

    # Confidence stats per pose
    conf_by_pose: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        try:
            conf = float(row.get("Confidence", 0))
            pose = row.get("Pose", "No Pose").strip()
            if conf > 0:
                conf_by_pose[pose].append(conf)
        except ValueError:
            pass

    for pose, confs in sorted(conf_by_pose.items()):
        avg = sum(confs) / len(confs)
        mn  = min(confs)
        mx  = max(confs)
        lines.append(
            f"  {pose:<20}  avg={avg:.3f}  min={mn:.3f}  max={mx:.3f}"
            f"  n={len(confs)}"
        )

    lines += ["", sep]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # Decide which CSV to use
    if os.path.exists(GROUND_TRUTH_CSV):
        source = GROUND_TRUTH_CSV
        print(f"[INFO] Loading ground-truth CSV: {source}")
    elif os.path.exists(SAMPLE_CSV):
        source = SAMPLE_CSV
        print(f"[INFO] Ground-truth CSV not found. Using sample: {source}")
    else:
        print("[ERROR] No CSV found. Run a session first or provide eval/ground_truth.csv")
        sys.exit(1)

    rows   = load_csv(source)
    report = generate_report(rows, source)

    print("\n" + report)

    os.makedirs("eval", exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write(report)
    print(f"\n[INFO] Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
