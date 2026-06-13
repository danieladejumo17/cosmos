#!/usr/bin/env python3
"""Semantic anomaly reasoning over ego-vehicle driving videos with Cosmos3 Reasoner.

Auto-launches a vLLM OpenAI-compatible server (Nano on 1 GPU, Super on 4 GPUs with
tensor-parallelism 4), runs inference over a dataset of videos, classifies each as
Anomaly/Normal, and writes a detailed JSON experiment report.
"""

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import openai

from utils import Metrics, discover_videos, parse_classification, read_action_sequence

MODELS = {
    "nano": "nvidia/Cosmos3-Nano",
    "super": "nvidia/Cosmos3-Super",
}

# Base prompts (verbatim from anomaly_reasoning.md).
BASE_PROMPT_ACTION = (
    "You are an autonomous driving safety expert analyzing this ego vehicle's video "
    "for semantic or contextual anomalies, that may impact safe AV operation. The ego "
    "vehicle's action (state sequence) given afterwards is in the format "
    "[[velocity_in_mph, heading_in_degrees], ...]. Think about the video and the ego "
    "vehicle's action, is there any semantic misunderstanding of the autopilot that "
    "requires intervention? Reply with exactly one word of the following:\n"
    "Classification: Anomaly — if there is a semantic anomaly\n"
    "Classification: Normal — if there is no semantic anomaly."
)

BASE_PROMPT_NO_ACTION = (
    "You are an autonomous driving safety expert analyzing this ego vehicle's video "
    "for semantic or contextual anomalies, that may impact safe AV operation. Think "
    "about the video, is there any semantic misunderstanding of the autopilot that "
    "requires intervention? Reply with exactly one word of the following:\n"
    "Classification: Anomaly — if there is a semantic anomaly\n"
    "Classification: Normal — if there is no semantic anomaly."
)


# ============================================================
# vLLM server lifecycle
# ============================================================
def vllm_executable() -> str:
    """Locate the vllm CLI, preferring the one next to the running interpreter
    (so it works whether or not the venv is activated)."""
    candidate = Path(sys.executable).parent / "vllm"
    if candidate.exists():
        return str(candidate)
    found = shutil.which("vllm")
    if found:
        return found
    raise FileNotFoundError(
        "Could not find the 'vllm' CLI. Install it (see setup_reasoner.sh) and "
        "run with the project's .venv."
    )


def launch_server(model_name: str, model_key: str, port: int, log_path: Path):
    """Launch `vllm serve` as a subprocess sized for the model. Returns the Popen."""
    cmd = [
        vllm_executable(), "serve", model_name,
        "--hf-overrides", '{"architectures": ["Cosmos3ReasonerForConditionalGeneration"]}',
        "--async-scheduling",
        "--allowed-local-media-path", "/",
        # Load all video frames and let the processor sample at request `fps`;
        # without this the default loader pre-truncates to 32 frames while the
        # metadata still references the full timeline, breaking do_sample_frames.
        "--media-io-kwargs", '{"video": {"num_frames": -1}}',
        "--port", str(port),
    ]
    env = os.environ.copy()
    if model_key == "super":
        # 64B model: split across 4 GPUs with tensor parallelism.
        env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        cmd += ["--tensor-parallel-size", "4", "--mm-encoder-tp-mode", "data"]
    else:
        # 16B Nano: single GPU.
        env["CUDA_VISIBLE_DEVICES"] = "0"

    print(f"🚀 Launching vLLM server: {' '.join(cmd)}")
    print(f"   CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}  (log: {log_path})")
    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env,
        start_new_session=True,  # own process group so we can tear down children
    )
    return proc


def wait_for_health(proc, port: int, timeout: int = 1800):
    """Poll the server /health endpoint until ready or timeout/crash."""
    url = f"http://127.0.0.1:{port}/health"
    print(f"⏳ Waiting for vLLM server on {url} (timeout {timeout}s)...")
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            raise RuntimeError(
                f"vLLM server exited early with code {proc.returncode}; check the server log."
            )
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    print(f"✅ Server ready in {time.time() - start:.1f}s\n")
                    return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"vLLM server not ready after {timeout}s")


def shutdown_server(proc):
    """Terminate the server process group."""
    if proc is None or proc.poll() is not None:
        return
    print("🛑 Shutting down vLLM server...")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=30)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


# ============================================================
# Inference
# ============================================================
def build_prompt(mode: str, action_text: str | None) -> str:
    if mode == "action_grounding":
        return f"{BASE_PROMPT_ACTION}\nEgo Vehicle State Sequence (5Hz): {action_text}"
    return BASE_PROMPT_NO_ACTION


def analyze_video(client, model_id, video_path: Path, prompt: str, fps: int, max_tokens: int) -> str:
    video_url = video_path.resolve().as_uri()
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": video_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        max_tokens=max_tokens,
        extra_body={"mm_processor_kwargs": {"fps": fps, "do_sample_frames": True}},
    )
    return response.choices[0].message.content


# ============================================================
# Environment info
# ============================================================
def gpu_info():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
        ).strip().splitlines()
        return {"count": len(out), "names": out}
    except Exception:
        return {"count": 0, "names": []}


def vllm_version():
    try:
        import vllm
        return vllm.__version__
    except Exception:
        return "unknown"


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Cosmos3 anomaly reasoning over driving videos")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset root (videos found recursively; labeled by folder)")
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name")
    parser.add_argument("--model", choices=["nano", "super"], default="nano")
    parser.add_argument("--mode", choices=["action_grounding", "no_action_grounding"],
                        default="no_action_grounding")
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--max_tokens", type=int, default=100)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    model_name = MODELS[args.model]
    tp_size = 4 if args.model == "super" else 1

    videos = discover_videos(args.dataset)
    if not videos:
        print(f"No labeled videos found under {args.dataset}")
        sys.exit(1)

    # Experiment log directory.
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = "_action" if args.mode == "action_grounding" else ""
    log_dir = Path("logs") / f"{args.exp_name}{suffix}_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    server_log = log_dir / "vllm_server.log"
    report_path = log_dir / "report.json"

    print(f"📂 Found {len(videos)} videos  |  model={model_name}  mode={args.mode}")
    print(f"📝 Logs: {log_dir}\n" + "=" * 60)

    start_time = datetime.now()
    proc = None
    metrics = Metrics()
    results = []
    counts = {"Anomaly": 0, "Normal": 0, "Unknown": 0, "Error": 0}

    try:
        proc = launch_server(model_name, args.model, args.port, server_log)
        wait_for_health(proc, args.port)

        client = openai.OpenAI(api_key="EMPTY", base_url=f"http://localhost:{args.port}/v1")
        model_id = client.models.list().data[0].id

        for i, (video_path, true_label) in enumerate(videos, 1):
            action_text = None
            try:
                if args.mode == "action_grounding":
                    action_text = read_action_sequence(video_path)
                prompt = build_prompt(args.mode, action_text)

                t0 = time.time()
                raw = analyze_video(client, model_id, video_path, prompt, args.fps, args.max_tokens)
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
                      f"(truth={'Anomaly' if true_label == 1 else 'Normal'}, "
                      f"{inference_time:.2f}s)")

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

    finally:
        shutdown_server(proc)

    end_time = datetime.now()
    metric_results = metrics.compute() if metrics.count > 0 else None

    report = {
        "experiment": {
            "exp_name": args.exp_name,
            "mode": args.mode,
            "timestamp": timestamp,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_s": round((end_time - start_time).total_seconds(), 1),
        },
        "environment": {
            "hostname": socket.gethostname(),
            "python": sys.version.split()[0],
            "vllm": vllm_version(),
            "gpu": gpu_info(),
        },
        "cli_args": vars(args),
        "inference_params": {
            "model": model_name,
            "tensor_parallel_size": tp_size,
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
    print(f"\nSUMMARY — {args.exp_name} ({args.mode})")
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
