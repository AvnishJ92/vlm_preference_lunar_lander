"""Prompt tuning: how well does each VLM prompt / input variant agree with the true-reward oracle?

    python -m prefrl.calibrate --config configs/default.yaml --out <run dir>/outputs/calibration

The clip pairs are the kind a training run asks about. A true-reward PPO run is trained from scratch, so its
experience goes from crashing through hovering to landing, and both clips of a pair come from the same PPO update
(tagged early / mid / late). A further set pairs a lander that has flown out of the picture with a clip from that
run. (An earlier calibration on landers of mixed skill gave 76% agreement and did not predict training, where the
VLM was at or below chance.) The pool is saved, so a second process, or another model, scores the same pairs.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

from prefrl import env as E
from prefrl.train import load_config
from prefrl.vlm import QwenJudge, clip_offsets, parse

log = logging.getLogger("prefrl")
GROUPS = ("early", "mid", "late", "away")
LOG_KEYS = ("obs", "act", "rew", "ep_seed", "ep_step")


def build_pool(cfg, seed, path):
    """Experience plus the pairs to judge, saved to `path`. Returns (xp, pairs (n, 2, 2), group (n,))."""
    c, k, n_envs, n_steps = cfg["calibration"], cfg["pref"]["segment_len"], cfg["n_envs"], cfg["ppo"]["n_steps"]
    updates = c["train_steps"] // (n_envs * n_steps)
    torch.set_num_threads(cfg["threads"])  # PPO's small networks crawl when torch grabs every core of the node
    rng, xp = np.random.default_rng(seed), E.make_venv(n_envs, seed, c["away_steps"] + (updates + 1) * n_steps)
    xp.reset()
    for _ in range(c["away_steps"]):  # main engine only: the lander climbs out of the picture
        xp.step(np.full(n_envs, 2))
    PPO("MlpPolicy", xp, seed=seed, device="cpu", **cfg["ppo"]).learn(updates * n_envs * n_steps)

    def windows(lo, hi, m):
        return np.stack([rng.integers(n_envs, size=m), rng.integers(lo, hi - k + 1, size=m)], -1)

    pairs, group, per = [], [], c["n_pairs"] // updates
    for u in range(updates):  # both clips from the same PPO update, as in a real query
        lo = c["away_steps"] + u * n_steps
        pairs.append(np.stack([windows(lo, lo + n_steps, per), windows(lo, lo + n_steps, per)], 1))
        group += [GROUPS[3 * u // updates]] * per
    gone = windows(0, c["away_steps"], 20 * c["n_away_pairs"])
    gone = gone[E.out_of_view(xp.segments(gone, k)[0]).mean(1) > 0.5][: c["n_away_pairs"]]  # mostly out of view
    pairs.append(np.stack([gone, windows(c["away_steps"], xp.t, len(gone))], 1))  # the away clip is clip 1
    group += ["away"] * len(gone)
    order = rng.permutation(len(group))
    pairs, group = np.concatenate(pairs)[order], np.array(group)[order]
    np.savez_compressed(path, pairs=pairs, group=group, t=xp.t, **{key: getattr(xp, key)[:, : xp.t] for key in LOG_KEYS})
    return xp, pairs, group


def load_pool(cfg, seed, path):
    saved = np.load(path)
    xp = E.make_venv(cfg["n_envs"], seed, int(saved["t"]))
    for key in LOG_KEYS:
        getattr(xp, key)[:] = saved[key]
    xp.t = int(saved["t"])
    return xp, saved["pairs"], saved["group"]


def score(mu, replies, oracle, gap):
    """Metrics of one variant on a set of pairs. mu: label per pair (None = can't tell); oracle: true-reward label."""
    if len(mu) == 0:
        return {}
    labelled = np.array([m is not None for m in mu])
    hard = np.array([m in (0.0, 1.0) for m in mu]) & (oracle != 0.5)
    right = hard & (np.array([m if m is not None else -1 for m in mu]) == oracle)
    answers = [parse(r) for pair in replies for r in pair]
    picks = [a for a in answers if a in ("1", "2")]
    return {
        "labelled": labelled.mean(),  # share of queries that produced a label at all
        "ties": np.mean([m == 0.5 for m in mu]),
        "clear": int(hard.sum()),  # labels that pick a clip, on pairs the oracle does not call a tie
        "agreement": right.sum() / max(hard.sum(), 1),  # of those, share matching the oracle
        "correct_per_query": right.mean(),  # what the reward predictor actually gets out of one query
        "wrong_per_query": (hard & ~right).mean(),
        "says_1": np.mean([a == "1" for a in picks]) if picks else float("nan"),  # 0.5 = no position bias
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--variants", nargs="*", help="only these (default: all); lets two processes share the work")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides, e.g. vlm.model=...")
    parser.add_argument("--tag", default="", help="prefix for this process's output names, e.g. 32b_")
    parser.add_argument("--limit", type=int, help="score only the first N pairs of the (shuffled) pool")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config(args.config, args.set)
    k = cfg["pref"]["segment_len"]
    args.out.mkdir(parents=True, exist_ok=True)

    pool = args.out / "pool.npz"
    xp, pairs, group = load_pool(cfg, args.seed, pool) if pool.exists() else build_pool(cfg, args.seed, pool)
    pairs, group = pairs[: args.limit], group[: args.limit]
    n, renderer = len(pairs), E.Renderer()
    ret = xp.segments(pairs, k)[2]
    oracle, gap = 0.5 * (1 + np.sign(ret[:, 0] - ret[:, 1])), np.abs(ret[:, 0] - ret[:, 1])
    log.info("pool: %s", {g: int((group == g).sum()) for g in GROUPS})
    if not (args.out / "pairs").exists():  # a few pairs to look at: clip 1 left, clip 2 right
        for i in range(min(n, 24)):
            clips = [renderer.frames(xp, *where, k) for where in pairs[i]]
            name = f"p{i:03d}_{group[i]}_oracle-{ {1.0: '1', 0.0: '2', 0.5: 'tie'}[oracle[i]] }.mp4"
            E.write_video(args.out / "pairs" / name, np.concatenate(clips, axis=2))

    for variant, overrides in cfg["calibration"]["variants"].items():
        if args.variants and variant not in args.variants:
            continue
        vlm, name = {**cfg["vlm"], **overrides}, args.tag + variant
        judge, tic, labels = QwenJudge(vlm, cfg["prompts"], k), time.time(), []
        for a in range(0, n, 32):  # bounds the frames held in memory
            shown = [
                tuple(
                    renderer.frames(xp, *w, k, clip_offsets(k, vlm["frames_per_clip"], xp.ends(*w, k)), vlm["mark_out_of_view"])
                    for w in pair
                )
                for pair in pairs[a : a + 32]
            ]
            labels += judge.label(shown)
        mu, replies = zip(*labels)
        row = {"variant": name, "pairs": n, **score(mu, replies, oracle, gap)}
        for g in GROUPS:  # the same, separately for each kind of pair
            idx = np.flatnonzero(group == g)
            part = score([mu[i] for i in idx], [replies[i] for i in idx], oracle[idx], gap[idx])
            row |= {f"{key}_{g}": part[key] for key in ("labelled", "clear", "agreement") if part}
        gone = [mu[i] for i in np.flatnonzero(group == "away") if mu[i] in (0.0, 1.0)]
        row["prefers_out_of_view_clip"] = float(np.mean([m == 1.0 for m in gone])) if gone else float("nan")
        row |= {"seconds_per_query": (time.time() - tic) / n}
        row = {key: v if key == "variant" else float(v) for key, v in row.items()}
        log.info("%s", {key: v if key == "variant" else round(v, 3) for key, v in row.items()})
        (args.out / f"score_{name}.json").write_text(json.dumps(row))
        with (args.out / f"answers_{name}.jsonl").open("w") as f:
            for i in range(n):
                record = dict(pair=i, group=str(group[i]), segments=pairs[i].tolist(), true_returns=ret[i].tolist(),
                              oracle_mu=float(oracle[i]), mu=mu[i], replies=replies[i])  # fmt: skip
                f.write(json.dumps(record) + "\n")
        # Rebuilt from the score files after every variant, so processes working on other variants are included.
        rows = [json.loads(path.read_text()) for path in sorted(args.out.glob("score_*.json"))]
        pd.DataFrame(rows).sort_values("agreement", ascending=False).to_csv(args.out / "calibration.csv", index=False)


if __name__ == "__main__":
    main()
