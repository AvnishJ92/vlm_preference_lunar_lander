#!/usr/bin/env python3
"""Generate random LunarLander rollout pairs and save frame strips polled at FPS / 2.

Example:
    python scripts/random_rollout_strips.py --out random_rollout_strips
    python scripts/random_rollout_strips.py --n-pairs 10 --duration 1.7 --poll-fps 25 --out random_rollout_strips
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from prefrl import env as E


def resize_frames(frames: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return frames
    h, w = frames.shape[1:3]
    size = (round(w * scale), round(h * scale))
    return np.stack([np.asarray(Image.fromarray(f).resize(size, Image.BICUBIC)) for f in frames])


def make_pair_strip(clip1: np.ndarray, clip2: np.ndarray, scale: float) -> np.ndarray:
    """Stack clip 1 (top row) and clip 2 (bottom row) horizontally into a single RGB strip."""
    row1 = np.concatenate(list(resize_frames(clip1, scale)), axis=1)
    row2 = np.concatenate(list(resize_frames(clip2, scale)), axis=1)
    divider = np.full((4, row1.shape[1], 3), 120, dtype=np.uint8)
    return np.concatenate([row1, divider, row2], axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-pairs", type=int, default=10, help="Number of random rollout pairs (default: 10)")
    parser.add_argument(
        "--duration",
        type=float,
        default=1.7,
        help=f"Clip duration in seconds at {E.FPS} FPS (default: 1.7 -> {round(1.7 * E.FPS)} steps)",
    )
    parser.add_argument(
        "--poll-fps",
        type=float,
        default=E.FPS / 2,
        help=f"Frame polling rate in FPS (default: {E.FPS / 2:g} = FPS/2)",
    )
    parser.add_argument("--n-envs", type=int, default=8, help="Parallel envs for collecting random rollouts")
    parser.add_argument("--rollout-steps", type=int, default=500, help="Steps per env to collect before sampling pairs")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--scale", type=float, default=0.5, help="Per-frame resize scale in the saved strip (default: 0.5)")
    parser.add_argument("--no-mark-out-of-view", action="store_true", help="Disable CRASHED / LANDER OUT OF VIEW banners")
    parser.add_argument("--save-video", action="store_true", help="Also save side-by-side MP4 videos at full FPS")
    parser.add_argument("--out", type=Path, default=Path("random_rollout_strips"), help="Output directory for strips")
    args = parser.parse_args()

    k = round(args.duration * E.FPS)
    stride = max(1, round(E.FPS / args.poll_fps))
    offsets = np.arange(0, k, stride, dtype=int)
    mark_oov = not args.no_mark_out_of_view

    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Collect purely random rollouts using the same HiddenReset + Experience wrapper as training.
    rollout_steps = max(args.rollout_steps, k * 3)
    xp = E.make_venv(args.n_envs, args.seed, capacity=rollout_steps, hide_episode_ends=True)
    xp.reset()
    for _ in range(rollout_steps):
        actions = rng.integers(xp.action_space.n, size=args.n_envs)
        xp.step(actions)

    pairs = xp.sample_pairs(rng, args.n_pairs, k)
    _, _, rets = xp.segments(pairs, k)
    oracle = 0.5 * (1 + np.sign(rets[:, 0] - rets[:, 1]))
    oracle_tag = {1.0: "1", 0.0: "2", 0.5: "tie"}

    renderer = E.Renderer()
    records = []
    for i in range(args.n_pairs):
        clip1 = renderer.frames(xp, *pairs[i, 0], k, offsets=offsets, mark_out_of_view=mark_oov)
        clip2 = renderer.frames(xp, *pairs[i, 1], k, offsets=offsets, mark_out_of_view=mark_oov)
        strip = make_pair_strip(clip1, clip2, scale=args.scale)

        tag = oracle_tag[float(oracle[i])]
        stem = args.out / f"pair_{i:02d}_oracle-{tag}"
        strip_path = stem.with_name(stem.name + "_strip.png")
        imageio.imwrite(strip_path, strip)

        if args.save_video:
            full1 = renderer.frames(xp, *pairs[i, 0], k, mark_out_of_view=mark_oov)
            full2 = renderer.frames(xp, *pairs[i, 1], k, mark_out_of_view=mark_oov)
            E.write_video(stem.with_suffix(".mp4"), np.concatenate([full1, full2], axis=2), fps=E.FPS)

        records.append(
            {
                "pair": i,
                "strip": strip_path.name,
                "segments": pairs[i].tolist(),
                "true_returns": [float(rets[i, 0]), float(rets[i, 1])],
                "oracle_mu": float(oracle[i]),
                "num_frames": int(len(offsets)),
            }
        )

    renderer.env.close()
    xp.close()

    meta = {
        "env_fps": E.FPS,
        "duration_s": args.duration,
        "segment_steps": k,
        "poll_fps": E.FPS / stride,
        "stride": stride,
        "frame_offsets": offsets.tolist(),
        "seed": args.seed,
        "pairs": records,
    }
    (args.out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(
        f"Saved {args.n_pairs} strips ({len(offsets)} frames/clip @ {E.FPS / stride:g} FPS, "
        f"{k} steps = {k / E.FPS:.2f}s) to {args.out}"
    )


if __name__ == "__main__":
    main()
