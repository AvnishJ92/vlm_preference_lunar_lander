#!/usr/bin/env python3
"""Score saved LunarLander frame strips with the project's VLM judge (QwenJudge).

Example:
    python scripts/score_strips.py \
        --strips-dir random_rollout_strips \
        --api-urls http://10.24.6.85:8001/v1,http://10.24.6.85:8002/v1
"""

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import imageio.v2 as imageio
import numpy as np
import yaml

from prefrl.calibrate import score as compute_metrics
from prefrl.vlm import QwenJudge, clip_offsets, parse

DIVIDER_H = 4


def split_pair_strip(strip: np.ndarray, num_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Invert make_pair_strip() into (clip1, clip2) each shaped (num_frames, H, W, 3)."""
    total_h, total_w, c = strip.shape
    frame_h = (total_h - DIVIDER_H) // 2
    assert frame_h * 2 + DIVIDER_H == total_h, f"Unexpected strip height {total_h}"
    assert total_w % num_frames == 0, f"Strip width {total_w} not divisible by {num_frames}"
    frame_w = total_w // num_frames

    row1 = strip[:frame_h]
    row2 = strip[frame_h + DIVIDER_H :]
    clip1 = np.stack([row1[:, i * frame_w : (i + 1) * frame_w] for i in range(num_frames)], axis=0)
    clip2 = np.stack([row2[:, i * frame_w : (i + 1) * frame_w] for i in range(num_frames)], axis=0)
    return clip1, clip2


def mu_label(mu) -> str:
    if mu is None:
        return "NONE"
    return {1.0: "1", 0.0: "2", 0.5: "TIE"}.get(mu, str(mu))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strips-dir", type=Path, default=Path("random_rollout_strips"))
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/default.yaml")
    parser.add_argument("--prompts", type=Path, default=REPO_ROOT / "configs/prompts.yaml")
    parser.add_argument(
        "--api-urls",
        default="http://10.24.6.85:8001/v1,http://10.24.6.85:8002/v1",
        help="Comma-separated OpenAI-compatible VLM base URLs",
    )
    parser.add_argument("--prompt", default=None, help="Override prompts.yaml variant (default: from config)")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Override vlm.max_new_tokens")
    parser.add_argument(
        "--frames-per-clip",
        type=int,
        default=None,
        help="Frames to sample from each strip (default: from config, or 0 for all frames)",
    )
    parser.add_argument(
        "--frame-scale",
        type=float,
        default=1.0,
        help="Additional resize scale on top of strip frames (default: 1.0, since strips are already 0.5x = 300x200)",
    )
    parser.add_argument("--no-swap-check", action="store_true", help="Disable swap consistency check")
    parser.add_argument("--out", type=Path, default=None, help="Output JSON path (default: <strips-dir>/vlm_scores.json)")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    prompts = yaml.safe_load(args.prompts.read_text())
    meta = json.loads((args.strips_dir / "metadata.json").read_text())

    vlm_cfg = dict(cfg["vlm"])
    vlm_cfg["api_urls"] = [u.strip() for u in args.api_urls.split(",") if u.strip()]
    vlm_cfg["frame_scale"] = args.frame_scale
    if args.prompt is not None:
        vlm_cfg["prompt"] = args.prompt
    if args.max_new_tokens is not None:
        vlm_cfg["max_new_tokens"] = args.max_new_tokens
    if args.no_swap_check:
        vlm_cfg["swap_check"] = False

    first_num_frames = meta["pairs"][0]["num_frames"]
    if args.frames_per_clip is None:
        n_frames = vlm_cfg["frames_per_clip"]
    elif args.frames_per_clip == 0:
        n_frames = first_num_frames
    else:
        n_frames = args.frames_per_clip
    vlm_cfg["frames_per_clip"] = n_frames

    clip_pairs = []
    for rec in meta["pairs"]:
        strip = imageio.imread(args.strips_dir / rec["strip"])
        c1, c2 = split_pair_strip(strip, rec["num_frames"])
        if n_frames < rec["num_frames"]:
            idx = clip_offsets(rec["num_frames"], n_frames)
            c1, c2 = c1[idx], c2[idx]
        clip_pairs.append((c1, c2))

    judge = QwenJudge(vlm_cfg, prompts, segment_len=meta["segment_steps"])

    t0 = time.perf_counter()
    results = judge.label(clip_pairs)
    elapsed = time.perf_counter() - t0

    mus = [m for m, _ in results]
    replies = [r for _, r in results]
    oracle = np.array([rec["oracle_mu"] for rec in meta["pairs"]], dtype=float)
    gap = np.array([abs(rec["true_returns"][0] - rec["true_returns"][1]) for rec in meta["pairs"]], dtype=float)
    metrics = compute_metrics(mus, replies, oracle, gap)

    print(
        f"Model: {vlm_cfg['model']} | prompt={vlm_cfg['prompt']} | frames={n_frames}/{first_num_frames} | "
        f"swap_check={vlm_cfg['swap_check']} | {elapsed:.2f}s ({elapsed / len(clip_pairs):.2f}s/pair)\n"
    )
    header = f"{'Pair':>4}  {'Ret1':>7}  {'Ret2':>7}  {'Oracle':>6}  {'Fwd':>6}  {'Swp':>6}  {'VLM':>5}  {'Match':>5}"
    print(header)
    print("-" * len(header))

    pair_records = []
    for rec, (mu, pair_replies) in zip(meta["pairs"], results):
        fwd = parse(pair_replies[0])
        swp = parse(pair_replies[1]) if len(pair_replies) > 1 else "-"
        ora_lbl = mu_label(rec["oracle_mu"])
        vlm_lbl = mu_label(mu)
        if mu in (0.0, 1.0) and rec["oracle_mu"] != 0.5:
            match = "YES" if mu == rec["oracle_mu"] else "NO"
        elif mu == 0.5:
            match = "TIE"
        else:
            match = "FAIL"
        r1, r2 = rec["true_returns"]
        print(f"{rec['pair']:4d}  {r1:7.1f}  {r2:7.1f}  {ora_lbl:>6}  {fwd:>6}  {swp:>6}  {vlm_lbl:>5}  {match:>5}")
        pair_records.append(
            {
                "pair": rec["pair"],
                "strip": rec["strip"],
                "true_returns": rec["true_returns"],
                "oracle_mu": rec["oracle_mu"],
                "vlm_mu": mu,
                "fwd_answer": fwd,
                "swp_answer": swp,
                "match": match,
                "replies": pair_replies,
            }
        )

    print("\nSummary:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")

    out_path = args.out or (args.strips_dir / f"vlm_scores_{vlm_cfg['prompt']}.json")
    out_data = {
        "vlm_cfg": {
            "model": vlm_cfg["model"],
            "api_urls": vlm_cfg["api_urls"],
            "prompt": vlm_cfg["prompt"],
            "frames_per_clip": n_frames,
            "frame_scale": vlm_cfg["frame_scale"],
            "swap_check": vlm_cfg["swap_check"],
            "max_new_tokens": vlm_cfg["max_new_tokens"],
        },
        "elapsed_s": elapsed,
        "metrics": {k: (None if isinstance(v, float) and np.isnan(v) else float(v) if isinstance(v, (np.floating, float)) else int(v)) for k, v in metrics.items()},
        "pairs": pair_records,
    }
    out_path.write_text(json.dumps(out_data, indent=2))
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
