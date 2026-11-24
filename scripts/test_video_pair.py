#!/usr/bin/env python3
"""Test a single LunarLander clip pair using true video_url input on vLLM (:8002)."""

import argparse
import base64
import json
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import imageio.v2 as imageio
import numpy as np

from prefrl.vlm import _side_by_side_video

DIVIDER_H = 4


def split_pair_strip(strip: np.ndarray, num_frames: int) -> tuple[np.ndarray, np.ndarray]:
    total_h, total_w, _ = strip.shape
    frame_h = (total_h - DIVIDER_H) // 2
    frame_w = total_w // num_frames
    row1 = strip[:frame_h]
    row2 = strip[frame_h + DIVIDER_H :]
    clip1 = np.stack([row1[:, i * frame_w : (i + 1) * frame_w] for i in range(num_frames)], axis=0)
    clip2 = np.stack([row2[:, i * frame_w : (i + 1) * frame_w] for i in range(num_frames)], axis=0)
    return clip1, clip2


def encode_mp4_data_url(frames: np.ndarray, fps: float) -> str:
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        with imageio.get_writer(tmp.name, fps=fps, codec="libx264", macro_block_size=1) as w:
            for f in frames:
                w.append_data(f)
        raw = Path(tmp.name).read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:video/mp4;base64,{b64}"


def post_chat(api_url: str, payload: dict) -> tuple[dict, float]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        api_url.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data, time.perf_counter() - t0


def run_case(api_url: str, model: str, fps: int, messages: list[dict], max_tokens: int, label: str):
    payload = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "logprobs": True,
        "top_logprobs": 5,
        "media_io_kwargs": {"video": {"fps": fps}},
        "mm_processor_kwargs": {"cap_pixels_per_frame": False},
        "messages": messages,
    }
    data, dt = post_chat(api_url, payload)
    choice = data["choices"][0]
    text = choice["message"]["content"]
    usage = data.get("usage", {})
    print(f"=== {label} ({dt:.2f}s | prompt_tokens={usage.get('prompt_tokens')} | completion_tokens={usage.get('completion_tokens')}) ===")
    print(text.strip())
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strips-dir", type=Path, default=Path("random_rollout_strips"))
    parser.add_argument("--pair", type=int, default=0)
    parser.add_argument("--prompt-file", type=Path, default=REPO_ROOT / "prompt.txt")
    parser.add_argument("--api-url", default="http://10.24.6.85:8002/v1")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    meta = json.loads((args.strips_dir / "metadata.json").read_text())
    rec = meta["pairs"][args.pair]
    prompt_text = args.prompt_file.read_text().strip()
    strip = imageio.imread(args.strips_dir / rec["strip"])
    clip_a, clip_b = split_pair_strip(strip, rec["num_frames"])
    native_fps = float(meta["poll_fps"])

    print(
        f"Pair {rec['pair']} ({rec['strip']}) | Ret1={rec['true_returns'][0]:.1f}, Ret2={rec['true_returns'][1]:.1f} "
        f"| Oracle={'1' if rec['oracle_mu'] == 1.0 else '2' if rec['oracle_mu'] == 0.0 else 'TIE'}"
    )
    print(f"Prompt: {prompt_text!r}")
    print(f"Native clip: {len(clip_a)} frames @ {native_fps:g} FPS ({len(clip_a)/native_fps:.2f}s), requesting vLLM fps={args.fps}\n")

    vid_a = encode_mp4_data_url(clip_a, native_fps)
    vid_b = encode_mp4_data_url(clip_b, native_fps)

    # 1. Two sequential video_url clips: (A, B) and swapped (B, A)
    for tag, (v1, v2) in [("2x sequential video (A, B)", (vid_a, vid_b)), ("2x sequential video (B, A) [swapped]", (vid_b, vid_a))]:
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Clip 1:"},
                    {"type": "video_url", "video_url": {"url": v1}},
                    {"type": "text", "text": "Clip 2:"},
                    {"type": "video_url", "video_url": {"url": v2}},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        run_case(args.api_url, args.model, args.fps, msgs, args.max_tokens, tag)

    # 2. Single side-by-side video_url: (A, B) and swapped (B, A)
    sbs_ab = encode_mp4_data_url(_side_by_side_video(clip_a, clip_b, scale=1), native_fps)
    sbs_ba = encode_mp4_data_url(_side_by_side_video(clip_b, clip_a, scale=1), native_fps)
    for tag, vid in [("1x side-by-side video (A, B)", sbs_ab), ("1x side-by-side video (B, A) [swapped]", sbs_ba)]:
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": vid}},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        run_case(args.api_url, args.model, args.fps, msgs, args.max_tokens, tag)


if __name__ == "__main__":
    main()
