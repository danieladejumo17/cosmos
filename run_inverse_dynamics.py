"""run_inverse_dynamics.py - Cosmos3 action inverse-dynamics over generated_vids.

Runs Cosmos3-Nano inverse-dynamics inference (native Cosmos Framework PyTorch
entrypoint, the same path as
cookbooks/cosmos3/generator/action/run_id_with_cosmos_framework.ipynb) on every
video under ./generated_vids. For each video it:

  1. predicts the ego-motion action trajectory ([T-1, 9] = translation(3) + rot6d(6)),
  2. converts it with pose_rel_to_abs into camera-to-world poses,
  3. derives a [[velocity, heading_angle_from_center], ...] sequence downsampled
     to 5 Hz - velocity anchored to an initial 30 mph, heading in degrees relative
     to the first frame's heading,
  4. validates the tail of that sequence against the last sentence of the matching
     prompts/<scenario>/prompt.txt line and flags mismatches,
  5. saves the sequence as <video>.txt next to the video, and
  6. uploads the .txt to the Hugging Face dataset repo at the mirrored path.

Setup (run once):

    ./setup_inverse_dynamics.sh
    export HF_TOKEN=<write token with gated-model + dataset access>

Run (any interpreter; it re-execs into the framework venv automatically):

    python run_inverse_dynamics.py
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
from pathlib import Path


# --- Configuration (the only things you should need to change) -----------
HF_DATASET_REPO = "danieladejumo/av_semantic_anomalies"
INITIAL_SPEED_MPH = 30.0  # assumed ego speed at the first frame (anchors velocity)
FPS = 10                  # model action rate (AV inverse-dynamics is 10 Hz)
TARGET_HZ = 5             # downsample the final action sequence to this rate
ACTION_CHUNK_SIZE = 60    # AV reference setting (60 frames @ 10 FPS)
IMAGE_SIZE = 480
CHECKPOINT = "Cosmos3-Nano"

# Validation thresholds (heuristic, language-based).
TAIL_FRAC = 0.2     # fraction of the sequence treated as "tail"/"start"
DECEL_RATIO = 0.6   # end/start velocity below this => decelerating
ACCEL_RATIO = 1.4   # end/start velocity above this => accelerating
MAINTAIN_LO = 0.6   # maintain band: MAINTAIN_LO*start <= end <= MAINTAIN_HI*start
MAINTAIN_HI = 1.6
STEER_DEG = 5.0     # max |heading change| above this => steering


def find_repo_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "README.md").exists() and (path / "cookbooks").exists():
            return path
    return start


SCRIPT_DIR = Path(__file__).resolve().parent
COSMOS_ROOT = find_repo_root(SCRIPT_DIR)
GENERATED_VIDS_DIR = COSMOS_ROOT / "generated_vids"
PROMPTS_DIR = COSMOS_ROOT / "prompts"
COSMOS3_REPO = Path(os.environ.get("COSMOS3_REPO", COSMOS_ROOT / "packages" / "cosmos3")).resolve()
WORK_DIR = Path(os.environ.get("COSMOS3_ID_WORK_DIR", COSMOS_ROOT / "outputs" / "inverse_dynamics")).resolve()
SPEC_PATH = WORK_DIR / "inverse_dynamics_av.jsonl"
RUNS_DIR = WORK_DIR / "runs"


# ---------------------------------------------------------------------------
# Runtime environment (ported from run_id_with_cosmos_framework.ipynb, step 5):
# the framework venv ships CUDA + PyAV/ffmpeg shared libraries that must be on
# LD_LIBRARY_PATH before torch/cosmos_framework import. We configure the env and
# re-exec once under the framework venv python so the dynamic linker picks it up.
# ---------------------------------------------------------------------------
def framework_site_packages(python_bin: Path) -> Path | None:
    venv_root = python_bin.parent.parent
    for site_packages in sorted((venv_root / "lib").glob("python*/site-packages")):
        if (site_packages / "nvidia").is_dir():
            return site_packages
    return None


def nvidia_cuda_library_dirs(python_bin: Path) -> list[Path]:
    site_packages = framework_site_packages(python_bin)
    if site_packages is None:
        return []
    nvidia_root = site_packages / "nvidia"
    lib_dirs = []
    for lib_dir in sorted(nvidia_root.glob("**/lib")):
        if any(lib_dir.glob("lib*.so*")):
            lib_dirs.append(lib_dir)
    return lib_dirs


def torchcodec_ffmpeg_library_dirs(python_bin: Path, link_dir: Path) -> list[Path]:
    site_packages = framework_site_packages(python_bin)
    if site_packages is None:
        return []
    av_libs = site_packages / "av.libs"
    if not av_libs.is_dir():
        return []
    soname_patterns = {
        "libavcodec.so.62": "libavcodec-*.so.62*",
        "libavdevice.so.62": "libavdevice-*.so.62*",
        "libavfilter.so.11": "libavfilter-*.so.11*",
        "libavformat.so.62": "libavformat-*.so.62*",
        "libavutil.so.60": "libavutil-*.so.60*",
        "libswresample.so.6": "libswresample-*.so.6*",
        "libswscale.so.9": "libswscale-*.so.9*",
    }
    linked_any = False
    for soname, pattern in soname_patterns.items():
        matches = sorted(av_libs.glob(pattern))
        if not matches:
            continue
        link_dir.mkdir(parents=True, exist_ok=True)
        link = link_dir / soname
        target = matches[-1].resolve()
        if link.is_symlink() and link.resolve() == target:
            linked_any = True
            continue
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(target)
        linked_any = True
    return [link_dir, av_libs] if linked_any else []


def set_nvidia_package_home(env, site_packages: Path | None, env_name: str, package_name: str) -> None:
    if site_packages is None:
        return
    package_dir = site_packages / "nvidia" / package_name
    if package_dir.is_dir():
        env.setdefault(env_name, str(package_dir))


def ensure_nvidia_package_alias(site_packages: Path | None, alias_name: str, package_name: str) -> None:
    if site_packages is None:
        return
    package_dir = site_packages / "nvidia" / package_name
    alias_dir = site_packages / "nvidia" / alias_name
    if not package_dir.is_dir():
        return
    if alias_dir.is_symlink() and not alias_dir.exists():
        alias_dir.unlink()
    if not alias_dir.exists():
        alias_dir.symlink_to(package_dir, target_is_directory=True)


def prepend_env_paths(env, name: str, paths: list[Path]) -> None:
    new_paths = [str(path) for path in paths if path.exists()]
    old_paths = [path for path in env.get(name, "").split(":") if path]
    merged = []
    for path in [*new_paths, *old_paths]:
        if path not in merged:
            merged.append(path)
    if merged:
        env[name] = ":".join(merged)


def bootstrap_runtime_env() -> None:
    """Configure CUDA/ffmpeg env for the framework venv and re-exec once under it."""
    if os.environ.get("COSMOS3_ID_ENV_READY") == "1":
        return

    python_bin = COSMOS3_REPO / ".venv" / "bin" / "python"
    if not python_bin.exists():
        raise SystemExit(
            f"missing framework venv python: {python_bin}\n"
            f"Run ./setup_inverse_dynamics.sh first."
        )

    cuda_lib_dirs = nvidia_cuda_library_dirs(python_bin)
    ffmpeg_lib_dirs = torchcodec_ffmpeg_library_dirs(python_bin, WORK_DIR / "torchcodec_ffmpeg_links")
    prepend_env_paths(os.environ, "PYTHONPATH", [COSMOS3_REPO])
    prepend_env_paths(os.environ, "LD_LIBRARY_PATH", [*ffmpeg_lib_dirs, *cuda_lib_dirs])

    site_packages = framework_site_packages(python_bin)
    ensure_nvidia_package_alias(site_packages, "cudart", "cuda_runtime")
    set_nvidia_package_home(os.environ, site_packages, "CUDNN_HOME", "cudnn")
    set_nvidia_package_home(os.environ, site_packages, "CUDART_HOME", "cuda_runtime")
    set_nvidia_package_home(os.environ, site_packages, "NVRTC_HOME", "cuda_nvrtc")
    set_nvidia_package_home(os.environ, site_packages, "CURAND_HOME", "curand")
    cuda_include_dir = site_packages / "nvidia" / "cuda_runtime" / "include" if site_packages else None
    if cuda_include_dir and cuda_include_dir.exists():
        os.environ.setdefault("NVTE_CUDA_INCLUDE_DIR", str(cuda_include_dir))

    os.environ["COSMOS3_ID_ENV_READY"] = "1"
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    # Re-exec under the framework venv python so the new LD_LIBRARY_PATH applies
    # and cosmos_framework / torch import from the right interpreter.
    os.execv(str(python_bin), [str(python_bin), os.path.abspath(__file__), *sys.argv[1:]])


# ---------------------------------------------------------------------------
# Discovery + spec
# ---------------------------------------------------------------------------
def video_name(rel_path: Path) -> str:
    """Stable run name from a generated_vids-relative path (slashes -> '__')."""
    return rel_path.with_suffix("").as_posix().replace("/", "__")


def discover_videos() -> list[dict]:
    videos = sorted(GENERATED_VIDS_DIR.rglob("*.mp4"))
    if not videos:
        raise SystemExit(f"No .mp4 files found under {GENERATED_VIDS_DIR}")
    records = []
    for path in videos:
        rel = path.relative_to(GENERATED_VIDS_DIR)
        records.append({"name": video_name(rel), "video_path": path, "rel_path": rel})
    return records


def build_spec(records: list[dict]) -> None:
    SPEC_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for rec in records:
        lines.append(json.dumps({
            "action_chunk_size": ACTION_CHUNK_SIZE,
            "domain_name": "av",
            "fps": FPS,
            "image_size": IMAGE_SIZE,
            "view_point": "ego_view",
            "model_mode": "inverse_dynamics",
            "name": rec["name"],
            "prompt": "You are an autonomous vehicle planning system.",
            "seed": 0,
            "vision_path": str(rec["video_path"].resolve()),
        }))
    SPEC_PATH.write_text("\n".join(lines) + "\n")
    print(f"wrote spec ({len(records)} run(s)): {SPEC_PATH}")


def free_local_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


def run_inference() -> None:
    python_bin = COSMOS3_REPO / ".venv" / "bin" / "python"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env["MASTER_ADDR"] = env.get("MASTER_ADDR", "127.0.0.1")
    env["MASTER_PORT"] = env.get("MASTER_PORT", free_local_port())
    env["RANK"] = "0"
    env["WORLD_SIZE"] = "1"
    env["LOCAL_RANK"] = "0"
    cmd = [
        str(python_bin), "-m", "cosmos_framework.scripts.inference",
        "--parallelism-preset=latency",
        "-i", str(SPEC_PATH),
        "-o", str(RUNS_DIR),
        "--checkpoint-path", CHECKPOINT,
        "--seed", "0",
        "--no-guardrails",  # disable the Cosmos guardrail (avoids the gated Guardrail1 download + safety checks)
    ]
    print("running inference:\n  " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(COSMOS3_REPO), env=env, check=True)


def output_path_for(name: str) -> Path:
    return RUNS_DIR / name / "sample_outputs.json"


# ---------------------------------------------------------------------------
# Conversion: predicted action -> [[velocity, heading_angle_from_center], ...]
# ---------------------------------------------------------------------------
def action_to_sequence(action) -> tuple[list[list[float]], bool]:
    """Convert a predicted action [T-1, 9] into a 5 Hz [[velocity, heading], ...].

    Returns (sequence, anchor_fallback). velocity is anchored so the first 5 Hz
    step is INITIAL_SPEED_MPH; heading is degrees relative to the first frame.
    anchor_fallback is True when the first step was ~stationary and velocity was
    anchored on the max step instead.
    """
    import numpy as np
    from cosmos_framework.data.vfm.action.pose_utils import pose_rel_to_abs

    action = np.asarray(action, dtype=np.float64)
    poses_abs = np.asarray(pose_rel_to_abs(
        action,
        rotation_format="rot6d",
        pose_convention="backward_framewise",
        translation_scale=1.35,
    ), dtype=np.float64)  # [T, 4, 4] camera-to-world at FPS Hz

    # Downsample poses to TARGET_HZ first so all derived values share a 5 Hz step.
    stride = max(1, round(FPS / TARGET_HZ))
    poses = poses_abs[::stride]
    if len(poses) < 2:
        return [], False

    pos = poses[:, :3, 3]   # camera centers (world; X right, Y up, Z heading)
    fwd = poses[:, :3, 2]   # heading direction (+Z)

    # Heading (deg) relative to the first frame, on the ground plane (X-Z).
    yaw = np.arctan2(fwd[:, 0], fwd[:, 2])
    heading = np.degrees(np.unwrap(yaw - yaw[0]))

    # Ground-plane per-step displacement.
    disp = np.diff(pos[:, [0, 2]], axis=0)
    d = np.linalg.norm(disp, axis=1)  # [T_5hz - 1]

    eps = 1e-6
    anchor_fallback = False
    anchor = d[0]
    if anchor <= eps:
        # Start-from-stop scenario: anchoring on a ~0 first step would explode.
        anchor = d.max() if d.max() > eps else eps
        anchor_fallback = True
    velocity = INITIAL_SPEED_MPH * d / anchor

    seq = [[round(float(velocity[i]), 4), round(float(heading[i]), 4)] for i in range(len(d))]
    return seq, anchor_fallback


# ---------------------------------------------------------------------------
# Validation against the prompt's last sentence
# ---------------------------------------------------------------------------
def last_sentence(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    parts = [p.strip() for p in parts if p.strip()]
    return parts[-1] if parts else text.strip()


def prompt_sentence_for(rel_path: Path) -> str | None:
    """Last sentence of prompts/<scenario>/prompt.txt at line N for prompt_N_*."""
    scenario = rel_path.parts[0]
    match = re.match(r"prompt_(\d+)_", rel_path.stem)
    if match is None:
        return None
    idx = int(match.group(1))
    prompt_file = PROMPTS_DIR / scenario / "prompt.txt"
    if not prompt_file.exists():
        return None
    lines = [ln for ln in prompt_file.read_text().splitlines()]
    if idx >= len(lines):
        return None
    return last_sentence(lines[idx])


def classify_expected(sentence: str) -> dict:
    """Heuristic end-behavior expectation from a natural-language sentence."""
    s = sentence.lower()
    expected = {"speed": None, "steer": False}

    steer_kw = ["steer", "maneuver", "manoeuvr", "navigat", "oncoming lane",
                "into an oncoming", "jagged", "swerv", "changes lane"]
    accel_kw = ["starts moving", "starts to move", "begins to move",
                "begins to drive", "accelerat", "starts driving"]
    # Negations / "keeps going" phrases -> NOT decelerating, treat as maintain.
    no_decel_kw = ["no deceleration", "without stopping", "without slowing",
                   "runs into", "fails to detect", "driving through",
                   "continues to drive", "drive forward"]
    decel_kw = ["decelerat", "comes to a stop", "to a stop", "slows down",
                "slow down", "controlled stop", "complete stop", "compliant stop"]
    maintain_kw = ["maintains", "normal speed", "cruising", "continues driving",
                   "drives past", "smoothly past", "remains safely stopped",
                   "remains stopped", "lane position"]

    if any(k in s for k in steer_kw):
        expected["steer"] = True

    if any(k in s for k in accel_kw):
        expected["speed"] = "accelerate"
    elif any(k in s for k in no_decel_kw):
        expected["speed"] = "maintain"
    elif "resuming" in s or "nominal speed" in s:
        # e.g. "slows down ... before resuming nominal speed" -> net maintain
        expected["speed"] = "maintain"
    elif any(k in s for k in decel_kw):
        expected["speed"] = "decelerate"
    elif any(k in s for k in maintain_kw):
        expected["speed"] = "maintain"
    return expected


def evaluate_sequence(seq: list[list[float]]) -> tuple[float, float, float]:
    import numpy as np
    v = np.array([row[0] for row in seq], dtype=np.float64)
    h = np.array([row[1] for row in seq], dtype=np.float64)
    n = len(v)
    k = max(1, int(round(n * TAIL_FRAC)))
    start_v = float(v[:k].mean())
    end_v = float(v[-k:].mean())
    max_abs_heading = float(np.max(np.abs(h))) if n else 0.0
    return start_v, end_v, max_abs_heading


def check_match(expected: dict, start_v: float, end_v: float, max_heading: float) -> tuple[bool, list[str]]:
    eps = 1e-6
    ratio = end_v / max(start_v, eps)
    reasons: list[str] = []
    ok = True
    sp = expected["speed"]
    if sp == "decelerate":
        if not (ratio < DECEL_RATIO):
            ok = False
            reasons.append(f"expected deceleration, end/start velocity={ratio:.2f}")
    elif sp == "accelerate":
        if not (ratio > ACCEL_RATIO):
            ok = False
            reasons.append(f"expected acceleration, end/start velocity={ratio:.2f}")
    elif sp == "maintain":
        if not (MAINTAIN_LO <= ratio <= MAINTAIN_HI):
            ok = False
            reasons.append(f"expected ~constant speed, end/start velocity={ratio:.2f}")
    if expected["steer"] and max_heading < STEER_DEG:
        ok = False
        reasons.append(f"expected steering, max|heading|={max_heading:.1f}deg")
    return ok, reasons


# ---------------------------------------------------------------------------
def main() -> None:
    bootstrap_runtime_env()  # configures env + re-execs once; returns only when ready

    from huggingface_hub import HfApi

    print(f"cosmos root:        {COSMOS_ROOT}")
    print(f"framework:          {COSMOS3_REPO}")
    print(f"generated_vids:     {GENERATED_VIDS_DIR}")
    print(f"work dir:           {WORK_DIR}")

    records = discover_videos()
    print(f"found {len(records)} video(s)")

    # Resumable: only run inference for videos lacking a prior prediction.
    pending = [r for r in records if not output_path_for(r["name"]).exists()]
    if pending:
        build_spec(pending)
        run_inference()
    else:
        print("all predictions already present; skipping inference")

    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    api.create_repo(HF_DATASET_REPO, repo_type="dataset", exist_ok=True)

    flags: list[str] = []
    for rec in records:
        name, rel, video_path = rec["name"], rec["rel_path"], rec["video_path"]
        out_json = output_path_for(name)
        if not out_json.exists():
            msg = f"FLAG  {rel}: no prediction output ({out_json})"
            print(msg)
            flags.append(msg)
            continue

        outputs = json.loads(out_json.read_text())
        action = outputs["outputs"][0]["content"]["action"]  # [T-1, 9]
        seq, anchor_fallback = action_to_sequence(action)
        if not seq:
            msg = f"FLAG  {rel}: action too short to derive a sequence"
            print(msg)
            flags.append(msg)
            continue

        # Save the clean action sequence next to the video (same name, .txt).
        txt_path = video_path.with_suffix(".txt")
        txt_path.write_text(json.dumps(seq) + "\n")

        # Validate against the prompt's last sentence.
        sentence = prompt_sentence_for(rel)
        start_v, end_v, max_heading = evaluate_sequence(seq)
        note = " [velocity anchored on max step: start was ~stationary]" if anchor_fallback else ""
        if sentence is None:
            line = (f"NOTE  {rel}: no matching prompt sentence; "
                    f"start_v={start_v:.1f} end_v={end_v:.1f} max|heading|={max_heading:.1f}deg{note}")
            print(line)
        else:
            expected = classify_expected(sentence)
            if expected["speed"] is None and not expected["steer"]:
                line = (f"NOTE  {rel}: unclassified prompt; "
                        f"start_v={start_v:.1f} end_v={end_v:.1f} "
                        f"max|heading|={max_heading:.1f}deg{note} :: \"{sentence}\"")
                print(line)
            else:
                ok, reasons = check_match(expected, start_v, end_v, max_heading)
                tag = "MATCH" if ok else "FLAG "
                exp_str = expected["speed"] or "-"
                if expected["steer"]:
                    exp_str += "+steer"
                line = (f"{tag} {rel}: expected={exp_str} "
                        f"start_v={start_v:.1f} end_v={end_v:.1f} "
                        f"max|heading|={max_heading:.1f}deg{note}")
                if not ok:
                    line += " | " + "; ".join(reasons) + f" :: \"{sentence}\""
                    flags.append(line)
                print(line)

        # Upload the .txt to the dataset at the mirrored path.
        path_in_repo = rel.with_suffix(".txt").as_posix()
        try:
            api.upload_file(
                path_or_fileobj=str(txt_path),
                path_in_repo=path_in_repo,
                repo_id=HF_DATASET_REPO,
                repo_type="dataset",
            )
            print(f"  uploaded -> {HF_DATASET_REPO}:{path_in_repo}")
        except Exception as exc:  # noqa: BLE001 - surface upload errors but continue
            msg = f"FLAG  {rel}: upload failed: {exc}"
            print(msg)
            flags.append(msg)

    summary_path = GENERATED_VIDS_DIR / "inverse_dynamics_flags.txt"
    header = (f"Inverse-dynamics validation flags ({len(flags)} issue(s))\n"
              f"Heuristic, language-based comparison of the action-sequence tail "
              f"vs the last sentence of each prompt.\n\n")
    summary_path.write_text(header + ("\n".join(flags) + "\n" if flags else "(no flags)\n"))
    print(f"\nwrote summary: {summary_path}")
    print(f"done. {len(flags)} flag(s).")


if __name__ == "__main__":
    main()
