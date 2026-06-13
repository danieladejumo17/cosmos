"""Utilities for Cosmos3 anomaly-reasoning inference: metrics, output parsing,
dataset discovery, and action-sequence loading."""

import re
from pathlib import Path


# ============================================================
# Metrics — anomaly is the positive class (label 1)
# ============================================================
class Metrics:
    """Accumulate predictions/ground-truth/timings and compute classification metrics."""

    def __init__(self):
        self.preds = []
        self.trues = []
        self.times = []

    @property
    def count(self):
        return len(self.preds)

    def update(self, preds, trues, times):
        self.preds.extend(preds)
        self.trues.extend(trues)
        self.times.extend(times)

    def compute(self):
        tp = fp = tn = fn = 0
        for pred, true in zip(self.preds, self.trues):
            if pred == 1 and true == 1:
                tp += 1
            elif pred == 1 and true == 0:
                fp += 1
            elif pred == 0 and true == 0:
                tn += 1
            elif pred == 0 and true == 1:
                fn += 1

        total = tp + fp + tn + fn
        accuracy = (tp + tn) / total if total else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        avg_time = sum(self.times) / len(self.times) if self.times else 0.0

        return {
            "Accuracy": accuracy,
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "Avg Inference Time": avg_time,
            "count": total,
        }


# ============================================================
# Output parsing
# ============================================================
def parse_classification(raw: str) -> str:
    """Classify the model output as 'Anomaly', 'Normal', or 'Unknown'.

    The model may emit a <think>...</think> preamble before the verdict, so we
    prefer an explicit 'Classification:' line but fall back to scanning the text.
    """
    if not raw:
        return "Unknown"
    text = raw.lower()

    # Prefer the explicit verdict line.
    m = re.search(r"classification:\s*(anomaly|normal)", text)
    if m:
        return "Anomaly" if m.group(1) == "anomaly" else "Normal"

    # Fall back to whichever keyword appears.
    has_anomaly = "anomaly" in text
    has_normal = "normal" in text
    if has_anomaly and not has_normal:
        return "Anomaly"
    if has_normal and not has_anomaly:
        return "Normal"
    if has_anomaly and has_normal:
        # Both present without a clear verdict line — take the last mention.
        return "Anomaly" if text.rfind("anomaly") > text.rfind("normal") else "Normal"
    return "Unknown"


# ============================================================
# Dataset discovery
# ============================================================
def discover_videos(dataset_root) -> list[tuple[Path, int]]:
    """Recursively find .mp4 videos under dataset_root and label them by folder.

    Ground truth: a path containing 'negative_scenario' is an anomaly (label 1);
    'positive_scenario' is normal (label 0). Unlabeled videos are skipped.
    """
    root = Path(dataset_root)
    labeled = []
    for video in sorted(root.rglob("*.mp4")):
        path_str = str(video).lower()
        if "negative_scenario" in path_str:
            labeled.append((video, 1))
        elif "positive_scenario" in path_str:
            labeled.append((video, 0))
    return labeled


def read_action_sequence(video_path) -> str:
    """Read the same-stem .txt action sequence next to a video."""
    txt_path = Path(video_path).with_suffix(".txt")
    if not txt_path.exists():
        raise FileNotFoundError(f"Action sequence file not found: {txt_path}")
    return txt_path.read_text().strip()
