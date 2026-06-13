#!/usr/bin/env python3
"""Semantic anomaly reasoning over ego-vehicle driving videos with Cosmos-Reason1.

Mirrors vllm_cosmos_reasoning.py but runs the model in-process via the Hugging
Face transformers library (Qwen2.5-VL architecture) instead of a vLLM server.
Classifies each video as Anomaly/Normal and writes a detailed JSON report.
"""

import argparse
import json
import os
import socket
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import transformers

from utils import Metrics, discover_videos, parse_classification, read_action_sequence

warnings.filterwarnings("ignore", category=UserWarning)

DEFAULT_MODEL = "nvidia/Cosmos-Reason1-7B"

# Prompts (verbatim from cosmos1_anomaly_reasoning.md).
BASE_PROMPT_ACTION = (
    "You are an autonomous driving safety expert analyzing this ego vehicle's video "
    "for semantic or contextual anomalies, that may impact safe AV operation. The ego "
    "vehicle's action (state sequence) given afterwards is in the format "
    "[[velocity_in_mph, heading_in_degrees], ...].\n\n"
    "<think>\n"
    "Think about the video scenario and the ego vehicle's action.\n"
    "</think>\n\n"
    "<answer>\n"
    "Is there any semantic misunderstanding of the autopilot that requires "
    "intervention? Reply with exactly one word of the following:\n"
    "Classification: Anomaly — if there is a semantic anomaly\n"
    "Classification: Normal — if there is no semantic anomaly.\n"
    "</answer>"
)

BASE_PROMPT_NO_ACTION = (
    "You are an autonomous driving safety expert analyzing this ego vehicle's video "
    "for semantic or contextual anomalies, that may impact safe AV operation.\n\n"
    "<think>\n"
    "Think about the video scenario.\n"
    "</think>\n\n"
    "<answer>\n"
    "Is there any semantic misunderstanding of the autopilot that requires "
    "intervention? Reply with exactly one word of the following:\n"
    "Classification: Anomaly — if there is a semantic anomaly\n"
    "Classification: Normal — if there is no semantic anomaly.\n"
    "</answer>"
)


# ============================================================
# Model loading
# ============================================================
def load_model(model_name: str):
    print(f"🔧 Loading {model_name}...")
    start = time.time()
    model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    processor = transformers.AutoProcessor.from_pretrained(model_name)
    print(f"✅ Model ready in {time.time() - start:.1f}s\n")
    return model, processor


# ============================================================
# Prompt / inference
# ============================================================
def build_prompt(mode: str, action_text: str | None) -> str:
    if mode == "action_grounding":
        return f"{BASE_PROMPT_ACTION}\nEgo Vehicle State Sequence (5Hz): {action_text}"
    return BASE_PROMPT_NO_ACTION


def sample_frames(video_path: Path, fps: int) -> np.ndarray:
    """Decode a video to RGB frames sampled at ~`fps` using OpenCV.

    Done directly (instead of via qwen_vl_utils) to avoid its torchvision/decord
    backends, which are incompatible with this stack. Returns (T, H, W, 3) uint8
    with an even frame count (required by Qwen2.5-VL's temporal patch size of 2).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    native_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    step = max(1, round(native_fps / fps))
    frames, idx = [], 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()
    if not frames:
        raise ValueError(f"No frames decoded from {video_path}")
    if len(frames) % 2 == 1 and len(frames) > 1:
        frames = frames[:-1]
    return np.asarray(frames)


def analyze_video(model, processor, video_path: Path, prompt: str, fps: int, max_tokens: int) -> str:
    video = sample_frames(video_path, fps)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": str(video_path.resolve())},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        videos=[video],
        fps=float(fps),
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)

    trimmed = generated[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


# ============================================================
# Environment info
# ============================================================
def gpu_info():
    if not torch.cuda.is_available():
        return {"count": 0, "names": []}
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return {"count": len(names), "names": names}


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Cosmos-Reason1 anomaly reasoning over driving videos")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset root (videos found recursively; labeled by folder)")
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--mode", choices=["action_grounding", "no_action_grounding"],
                        default="no_action_grounding")
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--max_tokens", type=int, default=1024)
    args = parser.parse_args()

    videos = discover_videos(args.dataset)
    if not videos:
        print(f"No labeled videos found under {args.dataset}")
        sys.exit(1)

    # Experiment log directory: <exp_name>[_action]_reason1_<timestamp>.
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = "_action" if args.mode == "action_grounding" else ""
    log_dir = Path("logs") / f"{args.exp_name}{suffix}_reason1_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    report_path = log_dir / "report.json"

    print(f"📂 Found {len(videos)} videos  |  model={args.model}  mode={args.mode}")
    print(f"📝 Logs: {log_dir}\n" + "=" * 60)

    start_time = datetime.now()
    metrics = Metrics()
    results = []
    counts = {"Anomaly": 0, "Normal": 0, "Unknown": 0, "Error": 0}

    model, processor = load_model(args.model)

    for i, (video_path, true_label) in enumerate(videos, 1):
        action_text = None
        try:
            if args.mode == "action_grounding":
                action_text = read_action_sequence(video_path)
            prompt = build_prompt(args.mode, action_text)

            t0 = time.time()
            raw = analyze_video(model, processor, video_path, prompt, args.fps, args.max_tokens)
            inference_time = time.time() - t0

            prediction = parse_classification(raw)
            counts[prediction] += 1
            pred_label = 1 if prediction == "Anomaly" else 0
            metrics.update([pred_label], [true_label], [inference_time])

            results.append({
                "file": str(video_path),
                "true_label": "Anomaly" if true_label == 1 else "Normal",
                "prediction": prediction,
                "correct": pred_label == true_label,
                "raw_output": raw,
                "inference_time_s": round(inference_time, 3),
            })
            print(f"[{i}/{len(videos)}] {video_path.name}: {prediction} "
                  f"(truth={'Anomaly' if true_label == 1 else 'Normal'}, {inference_time:.2f}s)")

        except Exception as e:
            counts["Error"] += 1
            results.append({
                "file": str(video_path),
                "true_label": "Anomaly" if true_label == 1 else "Normal",
                "prediction": "Error",
                "raw_output": str(e),
                "inference_time_s": 0.0,
            })
            print(f"[{i}/{len(videos)}] {video_path.name}: ERROR - {e}")

    end_time = datetime.now()
    metric_results = metrics.compute() if metrics.count > 0 else None

    report = {
        "experiment": {
            "exp_name": args.exp_name,
            "mode": args.mode,
            "backend": "transformers",
            "timestamp": timestamp,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_s": round((end_time - start_time).total_seconds(), 1),
        },
        "environment": {
            "hostname": socket.gethostname(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "gpu": gpu_info(),
        },
        "cli_args": vars(args),
        "inference_params": {
            "model": args.model,
            "fps": args.fps,
            "max_tokens": args.max_tokens,
        },
        "summary": {
            "total_videos": len(videos),
            **counts,
        },
        "metrics": metric_results,
        "results": results,
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Console summary.
    print("=" * 60)
    print(f"\nSUMMARY — {args.exp_name} ({args.mode}) [Cosmos-Reason1 / transformers]")
    print("=" * 60)
    print(f"Total videos: {len(videos)}")
    for k in ("Anomaly", "Normal", "Unknown", "Error"):
        print(f"  - {k}: {counts[k]}")

    if metric_results:
        m = metric_results
        print("\nCLASSIFICATION METRICS")
        print("=" * 60)
        print(f"  TP: {m['TP']}  TN: {m['TN']}  FP: {m['FP']}  FN: {m['FN']}")
        print(f"  Accuracy:  {m['Accuracy']:.4f}")
        print(f"  Precision: {m['Precision']:.4f}")
        print(f"  Recall:    {m['Recall']:.4f}")
        print(f"  F1-Score:  {m['F1-Score']:.4f}")
        print(f"  Avg Inference Time: {m['Avg Inference Time']:.3f}s")

    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
