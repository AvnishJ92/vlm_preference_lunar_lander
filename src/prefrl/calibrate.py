"""Prompt tuning: how well does each VLM prompt / input variant agree with the true-reward oracle?

    python -m prefrl.calibrate --config configs/default.yaml --out <run dir>/outputs/calibration

Clip pairs come from landers of every skill level (LunarLander's built-in heuristic pilot with 0-100% of its actions
replaced by random ones), so they range from "crash vs clean landing" to near-ties. The model is loaded once and
every variant in `calibration.variants` labels the same pairs. Read calibration.csv, set `vlm.*` in default.yaml to
the best variant, commit, then launch the experiments.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from gymnasium.envs.box2d.lunar_lander import heuristic

from prefrl import env as E
from prefrl.train import load_config
from prefrl.vlm import QwenJudge, clip_offsets, load_qwen, parse

log = logging.getLogger("prefrl")
N_ENVS, STEPS = 8, 3000


def mixed_skill_experience(seed):
    """Env i follows the heuristic pilot but takes a random action with probability i / (N_ENVS - 1)."""
    rng, xp = np.random.default_rng(seed), E.make_venv(N_ENVS, seed, STEPS)
    obs = xp.reset()
    for _ in range(STEPS):
        pilot = [heuristic(env, o) for env, o in zip(xp.venv.envs, obs)]
        random = rng.random(N_ENVS) < np.linspace(0, 1, N_ENVS)
        obs = xp.step(np.where(random, rng.integers(4, size=N_ENVS), pilot))[0]
    return xp


def score(mu, replies, oracle, gap):
    """Metrics of one variant. mu: label per pair (None = can't tell); oracle: true-reward label; gap: |return gap|."""
    labelled = np.array([m is not None for m in mu])
    hard = np.array([m in (0.0, 1.0) for m in mu]) & (oracle != 0.5)
    right = hard & (np.array([m if m is not None else -1 for m in mu]) == oracle)
    big = gap >= np.median(gap)
    answers = [parse(r) for pair in replies for r in pair]
    picks = [a for a in answers if a in ("1", "2")]
    return {
        "labelled": labelled.mean(),  # share of queries that produced a label at all
        "ties": np.mean([m == 0.5 for m in mu]),
        "agreement": right.sum() / max(hard.sum(), 1),  # of the clear-cut labels, share matching the oracle
        "agreement_big_gap": (right & big).sum() / max((hard & big).sum(), 1),
        "agreement_small_gap": (right & ~big).sum() / max((hard & ~big).sum(), 1),
        "correct_per_query": right.mean(),  # what the reward predictor actually gets out of one query
        "wrong_per_query": (hard & ~right).mean(),
        "says_1": np.mean([a == "1" for a in picks]) if picks else float("nan"),  # 0.5 = no position bias
        "unsure": np.mean([a == "UNSURE" for a in answers]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config(args.config)
    k, n = cfg["pref"]["segment_len"], cfg["calibration"]["n_pairs"]
    args.out.mkdir(parents=True, exist_ok=True)

    xp, rng, renderer = mixed_skill_experience(args.seed), np.random.default_rng(args.seed), E.Renderer()
    pairs = xp.sample_pairs(rng, n, k)
    ret = xp.segments(pairs, k)[2]
    oracle, gap = 0.5 * (1 + np.sign(ret[:, 0] - ret[:, 1])), np.abs(ret[:, 0] - ret[:, 1])
    name = {1.0: "1", 0.0: "2", 0.5: "tie"}
    for i in range(min(n, 24)):  # a few pairs to look at: clip 1 left, clip 2 right
        clips = [renderer.frames(xp, *where, k) for where in pairs[i]]
        E.write_video(args.out / "pairs" / f"p{i:03d}_oracle-{name[oracle[i]]}.mp4", np.concatenate(clips, axis=2))

    model, processor = load_qwen(cfg["vlm"])
    rows = []
    for variant, overrides in cfg["calibration"]["variants"].items():
        vlm = {**cfg["vlm"], **overrides}
        judge, tic, labels = QwenJudge(model, processor, vlm, cfg["prompts"], k), time.time(), []
        for a in range(0, n, 32):  # bounds the frames held in memory
            shown = [
                tuple(renderer.frames(xp, *w, k, clip_offsets(k, vlm["frames_per_clip"], xp.ends(*w, k))) for w in pair)
                for pair in pairs[a : a + 32]
            ]
            labels += judge.label(shown)
        mu, replies = zip(*labels)
        row = {"variant": variant, **score(mu, replies, oracle, gap), "seconds_per_query": (time.time() - tic) / n}
        rows.append(row)
        log.info("%s", {key: round(v, 3) if isinstance(v, float) else v for key, v in row.items()})
        with (args.out / f"answers_{variant}.jsonl").open("w") as f:
            for i in range(n):
                record = dict(pair=i, segments=pairs[i].tolist(), true_returns=ret[i].tolist(),
                              oracle_mu=float(oracle[i]), mu=mu[i], replies=replies[i])  # fmt: skip
                f.write(json.dumps(record) + "\n")
        pd.DataFrame(rows).to_csv(args.out / "calibration.csv", index=False)  # rewritten after every variant
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
