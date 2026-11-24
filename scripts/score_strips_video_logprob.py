#!/usr/bin/env python3
"""Score all saved LunarLander strips using 1x side-by-side video_url (fps=16) + logprob swap-debiasing.

Example:
    python scripts/score_strips_video_logprob.py \
        --strips-dir random_rollout_strips \
        --api-url http://10.24.6.85:8002/v1
"""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
import math
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
import yaml

from prefrl.vlm import _side_by_side_video

DIVIDER_H = 4

BINARY_QUESTION = (
    "Compare CLIP 1 (left) and CLIP 2 (right) across the video using the judging criteria.\n"
    "You MUST choose one clip. Reply with ONLY a single digit and nothing else: `1` if CLIP 1 (left) is better, "
    "or `2` if CLIP 2 (right) is better."
)


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
    return "data:video/mp4;base64," + base64.b64encode(raw).decode("ascii")


def extract_binary_logprob_delta(choice: dict) -> tuple[str, float, dict]:
    """Return (hard_pick, logP('1') - logP('2'), debug_info) from choice logprobs."""
    content = (choice.get("logprobs") or {}).get("content") or []
    text = (choice["message"]["content"] or "").strip()

    # Find the first token position where '1' or '2' appears in top_logprobs.
    for pos in content:
        tops = pos.get("top_logprobs") or []
        lp1_vals = [t["logprob"] for t in tops if t["token"].strip() == "1"]
        lp2_vals = [t["logprob"] for t in tops if t["token"].strip() == "2"]
        if lp1_vals or lp2_vals:
            min_lp = min(t["logprob"] for t in tops) if tops else -20.0
            lp1 = max(lp1_vals) if lp1_vals else (min_lp - 1.0)
            lp2 = max(lp2_vals) if lp2_vals else (min_lp - 1.0)
            hard = "1" if lp1 >= lp2 else "2"
            return hard, float(lp1 - lp2), {
                "token": pos.get("token"),
                "lp1": float(lp1),
                "lp2": float(lp2),
                "found_both": bool(lp1_vals and lp2_vals),
                "raw_text": text,
            }
    return "?", 0.0, {"raw_text": text, "found_both": False}


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strips-dir", type=Path, default=Path("random_rollout_strips"))
    parser.add_argument("--prompts", type=Path, default=REPO_ROOT / "configs/prompts.yaml")
    parser.add_argument("--api-url", default="http://10.24.6.85:8002/v1")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--top-logprobs", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--use-simple-prompt", action="store_true", help="Use prompt.txt instead of prompts.yaml system")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    meta = json.loads((args.strips_dir / "metadata.json").read_text())
    native_fps = float(meta["poll_fps"])

    if args.use_simple_prompt:
        system_prompt = None
        user_question = (
            (REPO_ROOT / "prompt.txt").read_text().strip()
            + "\nReply with ONLY a single digit and nothing else: 1 if CLIP 1 (left) lands better, or 2 if CLIP 2 (right) lands better."
        )
    else:
        prompts = yaml.safe_load(args.prompts.read_text())
        system_prompt = prompts["system"] + prompts["out_of_view_rule"]
        user_question = BINARY_QUESTION

    # Encode side-by-side MP4s for both (A, B) and (B, A)
    t_enc0 = time.perf_counter()
    jobs = []  # (pair_idx, order_tag, mp4_url)
    for rec in meta["pairs"]:
        strip = imageio.imread(args.strips_dir / rec["strip"])
        c1, c2 = split_pair_strip(strip, rec["num_frames"])
        sbs_ab = encode_mp4_data_url(_side_by_side_video(c1, c2, scale=1), native_fps)
        sbs_ba = encode_mp4_data_url(_side_by_side_video(c2, c1, scale=1), native_fps)
        jobs.append((rec["pair"], "ab", sbs_ab))
        jobs.append((rec["pair"], "ba", sbs_ba))
    t_enc = time.perf_counter() - t_enc0

    def run_job(job):
        pair_idx, order_tag, mp4_url = job
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": mp4_url}},
                    {"type": "text", "text": user_question},
                ],
            }
        )
        payload = {
            "model": args.model,
            "temperature": 0.0,
            "max_tokens": 4,
            "logprobs": True,
            "top_logprobs": args.top_logprobs,
            "media_io_kwargs": {"video": {"fps": args.fps}},
            "mm_processor_kwargs": {"cap_pixels_per_frame": False},
            "messages": messages,
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            args.api_url.rstrip("/") + "/chat/completions",
            data=body,
            headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        hard, delta, info = extract_binary_logprob_delta(data["choices"][0])
        return pair_idx, order_tag, hard, delta, info, data.get("usage", {})

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=min(len(jobs), args.concurrency)) as pool:
        raw_results = list(pool.map(run_job, jobs))
    t_inf = time.perf_counter() - t0

    by_pair = {}
    for pair_idx, order_tag, hard, delta, info, usage in raw_results:
        by_pair.setdefault(pair_idx, {})[order_tag] = {
            "hard": hard,
            "delta": delta,
            "info": info,
            "usage": usage,
        }

    print(
        f"Model: {args.model} | mode=1x_sbs_video (fps={args.fps}) | logprob_swap_debias=True | "
        f"encode={t_enc:.2f}s | infer={t_inf:.2f}s ({t_inf/len(meta['pairs']):.2f}s/pair)\n"
    )
    header = (
        f"{'Pair':>4}  {'Ret1':>7}  {'Ret2':>7}  {'Ora':>3}  "
        f"{'Fwd':>3} {'dAB':>6}  {'Swp':>3} {'dBA':>6}  {'Bias':>6}  {'Delta':>6}  {'muSoft':>6}  {'VLM':>3}  {'Match':>5}"
    )
    print(header)
    print("-" * len(header))

    pair_records = []
    naive_clear, naive_right = 0, 0
    debias_right = 0
    conf_total, conf_right = 0, 0
    biases = []
    raw_picks_1 = 0
    raw_picks_total = 0

    for rec in meta["pairs"]:
        p = rec["pair"]
        ab = by_pair[p]["ab"]
        ba = by_pair[p]["ba"]
        # dAB = logP(1|A,B) - logP(2|A,B) = s(A) - s(B) - b
        # ba["delta"] = logP(1|B,A) - logP(2|B,A) = s(B) - s(A) - b
        # So dBA_for_A = logP(2|B,A) - logP(1|B,A) = -ba["delta"] = s(A) - s(B) + b
        d_ab = ab["delta"]
        d_ba_for_a = -ba["delta"]
        delta = 0.5 * (d_ab + d_ba_for_a)
        bias = 0.5 * (d_ba_for_a - d_ab)  # positive => favors slot 2
        mu_soft = sigmoid(delta)
        vlm_hard = "1" if delta >= 0 else "2"
        ora = "1" if rec["oracle_mu"] == 1.0 else "2"
        match = "YES" if vlm_hard == ora else "NO"

        biases.append(bias)
        for h in (ab["hard"], ba["hard"]):
            if h in ("1", "2"):
                raw_picks_total += 1
                raw_picks_1 += int(h == "1")

        # Naive hard swap check
        if ab["hard"] in ("1", "2") and ab["hard"] != ba["hard"]:
            naive_clear += 1
            if ab["hard"] == ora:
                naive_right += 1

        if match == "YES":
            debias_right += 1
        if abs(delta) >= 0.25:
            conf_total += 1
            if match == "YES":
                conf_right += 1

        r1, r2 = rec["true_returns"]
        print(
            f"{p:4d}  {r1:7.1f}  {r2:7.1f}  {ora:>3}  "
            f"{ab['hard']:>3} {d_ab:+6.2f}  {ba['hard']:>3} {d_ba_for_a:+6.2f}  {bias:+6.2f}  "
            f"{delta:+6.2f}  {mu_soft:6.3f}  {vlm_hard:>3}  {match:>5}"
        )
        pair_records.append(
            {
                "pair": p,
                "strip": rec["strip"],
                "true_returns": rec["true_returns"],
                "oracle": ora,
                "ab": ab,
                "ba": ba,
                "delta_ab": d_ab,
                "delta_ba_for_a": d_ba_for_a,
                "bias_toward_2": bias,
                "debiased_delta": delta,
                "mu_soft": mu_soft,
                "vlm_hard": vlm_hard,
                "match": match,
            }
        )

    n = len(meta["pairs"])
    prompt_toks = raw_results[0][5].get("prompt_tokens")
    print("\nSummary:")
    print(f"  prompt_tokens: {prompt_toks}")
    print(f"  raw says_1 (before debiasing): {raw_picks_1}/{raw_picks_total} ({raw_picks_1/max(raw_picks_total,1):.3f})")
    print(f"  mean position bias toward slot 2: {np.mean(biases):+.3f} logits (std {np.std(biases):.3f})")
    print(f"  naive hard swap-check: {naive_clear}/{n} clear, {naive_right}/{max(naive_clear,1)} agree ({naive_right/max(naive_clear,1):.1%})")
    print(f"  logprob debiased (all {n}): {debias_right}/{n} agree ({debias_right/n:.1%})")
    if conf_total:
        print(f"  logprob debiased (|delta| >= 0.25): {conf_right}/{conf_total} agree ({conf_right/conf_total:.1%})")

    out_path = args.out or (args.strips_dir / "vlm_scores_video_logprob.json")
    out_path.write_text(
        json.dumps(
            {
                "api_url": args.api_url,
                "model": args.model,
                "fps": args.fps,
                "encode_s": t_enc,
                "infer_s": t_inf,
                "prompt_tokens": prompt_toks,
                "mean_bias_toward_2": float(np.mean(biases)),
                "naive_clear": naive_clear,
                "naive_right": naive_right,
                "debiased_right": debias_right,
                "total": n,
                "pairs": pair_records,
            },
            indent=2,
        )
    )
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
