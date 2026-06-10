"""generate_videos.py - Cosmos3 text-to-video generation.

Generates 5s / 720p videos (no audio) for every structured JSON prompt under
./prompts using the Cosmos3 Diffusers pipeline - the same inference path as
cookbooks/cosmos3/generator/audiovisual/run_with_diffusers.ipynb - writes the
MP4s to ./generated_vids mirroring the prompt folder structure and names, and
uploads each finished video to a Hugging Face dataset repo in the background.

Setup (run once; handled by the existing helper script):

    ./setup_diffusers.sh
    source .venv-cosmos3-diffusers/bin/activate
    export HF_TOKEN=<token with access to the gated model and the dataset repo>

Run:

    python generate_videos.py

Switch models by editing MODEL_ID below (Cosmos3-Nano by default).
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video
from huggingface_hub import HfApi

# --- Configuration (the only things you should need to change) -----------
MODEL_ID = "nvidia/Cosmos3-Nano"  # change to "nvidia/Cosmos3-Super" for Super
HF_DATASET_REPO = "danieladejumo/av_semantic_anomalies"

SCRIPT_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = SCRIPT_DIR / "prompts"
OUTPUT_DIR = SCRIPT_DIR / "generated_vids"
NEGATIVE_PROMPT_FILE = SCRIPT_DIR / "text2video_neg_prompt.json"

# 5s of video. Video diffusion expects a 4n+1 frame count, so 121 -> ~5.04s @ 24fps.
NUM_FRAMES = 121
FPS = 24
HEIGHT = 720
WIDTH = 1280

NUM_STEPS = 35
GUIDANCE = 6.0
SHIFT = 10.0
SEED = 1234


def compact_json(path: Path) -> str:
    """Load a structured JSON file and serialize it compactly (as the notebook does)."""
    return json.dumps(json.loads(path.read_text()), ensure_ascii=True, separators=(",", ":"))


# Structured negative prompt, matching the notebook's text2video negative prompt.
NEGATIVE_PROMPT = compact_json(NEGATIVE_PROMPT_FILE)


def load_pipeline() -> Cosmos3OmniPipeline:
    print(f"loading {MODEL_ID} ...")
    t0 = time.time()
    pipe = Cosmos3OmniPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        safety_checker=None,
        enable_safety_checker=False,  # disable the Cosmos guardrail (text + video checks)
        token=os.environ.get("HF_TOKEN") or None,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=SHIFT)
    pipe.to("cuda")
    print(f"loaded pipeline in {time.time() - t0:.1f}s")
    return pipe


def generate_video(pipe: Cosmos3OmniPipeline, prompt_text: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cuda").manual_seed(SEED)
    result = pipe(
        prompt=prompt_text,
        negative_prompt=NEGATIVE_PROMPT,
        image=None,
        num_frames=NUM_FRAMES,
        height=HEIGHT,
        width=WIDTH,
        fps=FPS,
        num_inference_steps=NUM_STEPS,
        guidance_scale=GUIDANCE,
        enable_sound=False,
        add_resolution_template=False,
        add_duration_template=False,
        generator=generator,
    )
    export_to_video(result.video, str(out_path), fps=FPS, macro_block_size=1)


def upload_video(api: HfApi, local_path: Path, path_in_repo: str) -> None:
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
    )
    print(f"  uploaded -> {HF_DATASET_REPO}:{path_in_repo}")


def main() -> None:
    prompt_paths = sorted(PROMPTS_DIR.rglob("*.json"))
    if not prompt_paths:
        raise SystemExit(f"No JSON prompts found under {PROMPTS_DIR}")
    print(f"found {len(prompt_paths)} prompt(s) under {PROMPTS_DIR}")

    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    api.create_repo(HF_DATASET_REPO, repo_type="dataset", exist_ok=True)

    uploads: list[Future] = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        pipe = load_pipeline()
        for prompt_path in prompt_paths:
            rel_mp4 = prompt_path.relative_to(PROMPTS_DIR).with_suffix(".mp4")
            out_path = OUTPUT_DIR / rel_mp4
            if out_path.exists():
                # Already generated on a previous run; skip generation but still
                # upload so the dataset stays in sync with local files.
                print(f"skip generation (exists): {rel_mp4}")
            else:
                prompt_text = compact_json(prompt_path)
                print(f"generating {rel_mp4} ...")
                t0 = time.time()
                generate_video(pipe, prompt_text, out_path)
                print(f"  generated in {time.time() - t0:.1f}s -> {out_path}")

            # Upload in the background so the next video starts generating immediately.
            uploads.append(pool.submit(upload_video, api, out_path, rel_mp4.as_posix()))

        # Wait for all uploads to finish (and surface any errors) before exiting.
        print("waiting for pending uploads ...")
        for fut in uploads:
            fut.result()

    print("done.")


if __name__ == "__main__":
    main()
