#!/usr/bin/env python3
"""Semantic anomaly reasoning over ego-vehicle driving videos with Cosmos-Reason1.

Mirrors transformers_cosmos1_reasoning.py, but runs the model behind a vLLM
OpenAI-compatible server (instead of in-process transformers). The number of GPUs
and the tensor-parallel size are configurable via CLI. Classifies each video as
Anomaly/Normal and writes a detailed JSON report.
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


def launch_server(model_name: str, num_gpus: int, tensor_parallel_size: int, port: int, log_path: Path):
    """Launch `vllm serve` as a subprocess. Returns the Popen.

    Cosmos-Reason1 is a standard Qwen2.5-VL model, so vLLM serves it natively
    (no architecture override needed). The model is exposed `num_gpus` GPUs and
    sharded with the requested tensor-parallel size.
    """
    cmd = [
        vllm_executable(), "serve", model_name,
        "--async-scheduling",
        "--allowed-local-media-path", "/",
        # Load all video frames and let the processor sample at request `fps`;
        # without this the default loader pre-truncates to 32 frames while the
        # metadata still references the full timeline, breaking do_sample_frames.
        "--media-io-kwargs", '{"video": {"num_frames": -1}}',
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--port", str(port),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_gpus))

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
    parser = argparse.ArgumentParser(description="Cosmos-Reason1 anomaly reasoning over driving videos (vLLM)")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset root (videos found recursively; labeled by folder)")
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--mode", choices=["action_grounding", "no_action_grounding"],
                        default="no_action_grounding")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs to expose to the server")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM tensor-parallel size")
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.tensor_parallel_size > args.num_gpus:
        parser.error(f"--tensor-parallel-size ({args.tensor_parallel_size}) cannot exceed "
                     f"--num-gpus ({args.num_gpus})")
    if args.num_gpus % args.tensor_parallel_size != 0:
        parser.error(f"--num-gpus ({args.num_gpus}) must be divisible by "
                     f"--tensor-parallel-size ({args.tensor_parallel_size})")

    videos = discover_videos(args.dataset)
    if not videos:
        print(f"No labeled videos found under {args.dataset}")
        sys.exit(1)

    # Experiment log directory: <exp_name>[_action]_reason1_<timestamp>.
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = "_action" if args.mode == "action_grounding" else ""
    log_dir = Path("logs") / f"{args.exp_name}{suffix}_reason1_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    server_log = log_dir / "vllm_server.log"
    report_path = log_dir / "report.json"

    print(f"📂 Found {len(videos)} videos  |  model={args.model}  mode={args.mode}")
    print(f"   num_gpus={args.num_gpus}  tensor_parallel_size={args.tensor_parallel_size}")
    print(f"📝 Logs: {log_dir}\n" + "=" * 60)

    start_time = datetime.now()
    proc = None
    metrics = Metrics()
    results = []
    counts = {"Anomaly": 0, "Normal": 0, "Unknown": 0, "Error": 0}

    try:
        proc = launch_server(args.model, args.num_gpus, args.tensor_parallel_size, args.port, server_log)
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

    finally:
        shutdown_server(proc)

    end_time = datetime.now()
    metric_results = metrics.compute() if metrics.count > 0 else None

    report = {
        "experiment": {
            "exp_name": args.exp_name,
            "mode": args.mode,
            "backend": "vllm",
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
            "model": args.model,
            "num_gpus": args.num_gpus,
            "tensor_parallel_size": args.tensor_parallel_size,
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
    print(f"\nSUMMARY — {args.exp_name} ({args.mode}) [Cosmos-Reason1 / vLLM]")
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
