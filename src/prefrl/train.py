"""One run: PPO on LunarLander, rewarded by the true reward or by a predictor learned from pairwise preferences.

    python -m prefrl.train --config configs/default.yaml --condition vlm --seed 0 --out <run dir>
"""

import argparse
import csv
import json
import logging
import time
from pathlib import Path

import imageio
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

from prefrl import env as E
from prefrl.reward_model import PrefDB, RewardEnsemble
from prefrl.vlm import QwenJudge, clip_offsets, load_qwen

log = logging.getLogger("prefrl")
EVAL_SEED = 10_000  # evaluation episode i uses seed EVAL_SEED + i in every run: all conditions face the same landings


def load_config(path, overrides=()):
    """default.yaml plus the prompts next to it, with `a.b=value` overrides applied."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    cfg["prompts"] = yaml.safe_load((path.parent / "prompts.yaml").read_text())
    for item in overrides:
        key, value = item.split("=", 1)
        *parents, leaf = key.split(".")
        node = cfg
        for name in parents:
            node = node[name]
        if leaf not in node:
            raise KeyError(f"--set {key}: no such config key")
        node[leaf] = yaml.safe_load(value)
    return cfg


def append_row(path, row):
    new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def labels_due(t, total_t, n, pretrain_frac, tau_frac):
    """How many queries should have been asked by timestep t.

    `pretrain_frac` of the budget up front, on clips of the untrained policy; the rest at a rate proportional to
    tau / (t + tau) (paper A.1: 2e6 / (T + 2e6)), scaled so the budget n is spent exactly at total_t.
    """
    n_pre = round(pretrain_frac * n)
    tau = tau_frac * total_t
    return n_pre + int((n - n_pre) * np.log1p(min(t, total_t) / tau) / np.log1p(total_t / tau))


class Feedback:
    """Steps 2 and 3 of the paper's loop (2.2): choose queries, have them labelled, fit the reward predictor."""

    def __init__(self, cfg, oracle_fraction, xp, model, out, rng, judge):
        self.p, self.v, self.oracle_fraction = cfg["pref"], cfg["vlm"], oracle_fraction
        self.xp, self.model, self.out, self.rng, self.judge = xp, model, out, rng, judge
        self.db = PrefDB(self.p["ensemble"], rng)
        self.renderer = E.Renderer() if judge else None
        self.asked = 0  # queries sent to a labeller; len(self.db) is how many came back with a label
        self.mark = 0  # log position (steps per env) of the previous round; a round looks at what came after it
        self.vlm = dict(asked=0, labelled=0, ties=0, hard=0, agree=0)
        self.pred_acc = float("nan")

    def _shown(self, where):
        """The frames of one segment that the VLM gets to see."""
        k = self.p["segment_len"]
        offsets = clip_offsets(k, self.v["frames_per_clip"], self.xp.ends(*where, k))
        return self.renderer.frames(self.xp, *where, k, offsets)

    def _save_clip(self, qid, pair, shown, mu, oracle):
        """Side-by-side mp4 of a query (clip 1 left, clip 2 right) and a strip of exactly the frames the VLM saw."""
        name = {1.0: "1", 0.0: "2", 0.5: "tie", None: "none"}
        stem = self.out / "videos/queries" / f"q{qid:05d}_vlm-{name[mu]}_oracle-{name[oracle]}"
        full = [self.renderer.frames(self.xp, *where, self.p["segment_len"]) for where in pair]
        E.write_video(stem.with_suffix(".mp4"), np.concatenate(full, axis=2))
        strip = np.concatenate([np.concatenate(list(clip[:, ::2, ::2]), axis=1) for clip in shown], axis=0)
        imageio.imwrite(stem.with_name(stem.name + "_shown.png"), strip)

    def query(self, n, t):
        """Ask for n labels on pairs of segments recorded since the previous round."""
        k, tic = self.p["segment_len"], time.time()
        cand = self.xp.sample_pairs(self.rng, n * self.p["candidate_factor"], k, self.mark)
        obs, act, ret = self.xp.segments(cand, k)
        oracle = 0.5 * (1 + np.sign(ret[:, 0] - ret[:, 1]))  # true-reward preference: 1, 0, or 0.5 on an exact tie
        with torch.no_grad():
            p = self.model.prefer_first(obs, act)
        # None of these candidates is labelled yet, so this is a held-out accuracy of the current predictor.
        clear = oracle != 0.5
        self.pred_acc = float(((p.mean(0).numpy() > 0.5) == (oracle == 1))[clear].mean())
        top = p.var(0).argsort(descending=True)[:n].numpy()  # the pairs the ensemble disagrees on most (2.2.4)

        mu, replies = oracle[top].tolist(), [None] * n
        by_vlm = np.flatnonzero(self.rng.random(n) >= self.oracle_fraction)
        for part in (by_vlm[a : a + 32] for a in range(0, len(by_vlm), 32)):  # bounds the frames held in memory
            shown = [[self._shown(where) for where in cand[i]] for i in top[part]]
            for j, clips, (m, r) in zip(part, shown, self.judge.label([tuple(c) for c in shown])):
                mu[j], replies[j], o, s = m, r, float(oracle[top[j]]), self.vlm
                if s["asked"] % self.v["save_clip_every"] == 0:
                    self._save_clip(self.asked + j, cand[top[j]], clips, m, o)
                s["asked"] += 1
                s["labelled"] += m is not None
                s["ties"] += m == 0.5
                if m in (0.0, 1.0) and o != 0.5:
                    s["hard"] += 1
                    s["agree"] += m == o

        with (self.out / "queries.jsonl").open("a") as f:
            for j, i in enumerate(top):
                if mu[j] is not None:
                    self.db.add(obs[i], act[i], mu[j])
                record = dict(
                    query=self.asked + j, t=t, segments=cand[i].tolist(), true_returns=ret[i].tolist(),
                    oracle_mu=float(oracle[i]), labeler="oracle" if replies[j] is None else "vlm",
                    replies=replies[j], mu=mu[j],
                )  # fmt: skip
                f.write(json.dumps(record) + "\n")
        self.asked += n
        log.info("t=%d asked %d queries (%d by VLM) in %.0fs; %d labels in D", t, n, len(by_vlm), time.time() - tic, len(self.db))

    def fit(self, epochs, t):
        """Train the predictor, renormalise it on the experience since the previous round, and close the round."""
        train = val = float("nan")
        if len(self.db):  # a VLM that never commits to an answer leaves D empty; r-hat then stays as initialised
            train, val = self.model.fit(self.db, epochs, self.p["batch_size"], self.rng)
        obs, act, rew = self.xp.recent(self.xp.t - self.mark)
        self.model.set_normalisation(obs, act)
        self.mark = self.xp.t
        s = self.vlm
        row = dict(
            t=t, asked=self.asked, labels=len(self.db), train_loss=train, val_loss=val,
            l2=self.model.l2.mean().item(), rhat_true_corr=np.corrcoef(self.model.reward(obs, act), rew)[0, 1],
            pred_oracle_acc=self.pred_acc, vlm_asked=s["asked"], vlm_labelled=s["labelled"], vlm_ties=s["ties"],
            vlm_oracle_agree=s["agree"] / s["hard"] if s["hard"] else float("nan"),
        )  # fmt: skip
        append_row(self.out / "rounds.csv", row)
        log.info("t=%d predictor: %s", t, {k: round(v, 4) for k, v in row.items() if k != "t"})


class Loop(BaseCallback):
    """Runs a feedback round every `pref.round_every` timesteps (if rewards are learned) and the evaluations.

    Both schedules are checked after each PPO rollout (n_envs x n_steps timesteps).
    """

    def __init__(self, cfg, out, feedback=None):
        super().__init__()
        self.cfg, self.out, self.fb = cfg, out, feedback
        self.next_eval, self.next_round = cfg["eval"]["every"], cfg["pref"]["round_every"]

    def evaluate(self, t, n_episodes, n_videos, tag=None):
        rets, lens, landed = E.evaluate(
            self.model, EVAL_SEED + np.arange(n_episodes), self.out / "videos/eval", n_videos, tag or f"step{t:08d}"
        )
        row = dict(t=t, mean=rets.mean(), std=rets.std(), min=rets.min(), max=rets.max(),
                   landed=landed.mean(), length=lens.mean(), episodes=n_episodes)  # fmt: skip
        append_row(self.out / "eval.csv", row)
        log.info("t=%d eval: %s", t, {k: round(float(v), 2) for k, v in row.items() if k != "t"})
        return row

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        t, p, e = self.num_timesteps, self.cfg["pref"], self.cfg["eval"]
        if self.fb and t >= self.next_round:
            due = labels_due(t, self.cfg["total_timesteps"], p["n_queries"], p["pretrain_frac"], p["anneal_tau_frac"])
            if due > self.fb.asked:
                self.fb.query(due - self.fb.asked, t)
            self.fb.fit(p["epochs_per_round"], t)
            self.next_round += p["round_every"]
        if t >= self.next_eval:
            self.evaluate(t, e["episodes"], e["video_episodes"])
            self.next_eval += e["every"]


def run(cfg, condition, seed, out, judge=None):
    """Train one (condition, seed). `judge` lets tests stand in for the VLM. Returns the final evaluation row."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cond, p, n_envs = cfg["conditions"][condition], cfg["pref"], cfg["n_envs"]
    (out / "config.yaml").write_text(yaml.safe_dump(dict(cfg, condition=condition, seed=seed), sort_keys=False))
    torch.set_num_threads(cfg["threads"])
    # The adaptive l2 penalty shrinks unused predictor weights into the subnormal float range, where CPU arithmetic is
    # ~10x slower (measured: 28 ms per label per fit instead of 3.5 ms). Treat them as the zeros they effectively are.
    torch.set_flush_denormal(True)
    rng = np.random.default_rng(seed)

    learned = "oracle_fraction" in cond  # otherwise: PPO on the true reward
    update = n_envs * cfg["ppo"]["n_steps"]  # timesteps per policy update
    capacity = (cfg["total_timesteps"] + p["pretrain_steps"]) // n_envs + 2 * cfg["ppo"]["n_steps"]
    xp = E.make_venv(n_envs, seed, capacity, hide_episode_ends=cfg["hide_episode_ends"])
    policy = PPO("MlpPolicy", xp, seed=seed, device="cpu", **cfg["ppo"])
    policy.set_logger(configure(str(out), ["stdout", "csv"]))

    feedback = None
    if learned:
        reward_model = RewardEnsemble(
            xp.observation_space.shape[0], xp.action_space.n, p["ensemble"], p["hidden"], p["lr"],
            p["label_noise"], p["l2_init"], p["val_ratio"],
        )  # fmt: skip
        xp.reward_fn = reward_model.reward
        if judge is None and cond["oracle_fraction"] < 1:
            judge = QwenJudge(*load_qwen(cfg["vlm"]), cfg["vlm"], cfg["prompts"], p["segment_len"])
        feedback = Feedback(cfg, cond["oracle_fraction"], xp, reward_model, out, rng, judge)
        # Paper A.1: the first quarter of the comparisons is on clips of the randomly initialised policy.
        obs = xp.reset()
        for _ in range(p["pretrain_steps"] // n_envs):
            obs = xp.step(policy.predict(obs)[0])[0]
        feedback.query(labels_due(0, 1, p["n_queries"], p["pretrain_frac"], 1), 0)
        feedback.fit(p["pretrain_epochs"], 0)

    loop = Loop(cfg, out, feedback)
    loop.init_callback(policy)
    loop.evaluate(0, cfg["eval"]["episodes"], cfg["eval"]["video_episodes"])
    # progress.csv gets about ten rows per evaluation interval
    policy.learn(cfg["total_timesteps"], callback=loop, log_interval=max(1, cfg["eval"]["every"] // update // 10))

    e = cfg["eval"]
    final = loop.evaluate(policy.num_timesteps, e["final_episodes"], e["final_video_episodes"], tag="final")
    policy.save(out / "policy")
    if learned:
        torch.save(reward_model.state_dict(), out / "reward_model.pt")
        if len(feedback.db):
            arrays = dict(zip(["obs", "act", "mu", "bootstrap"], feedback.db.arrays()))
            np.savez_compressed(out / "preferences.npz", **arrays)
    # Written last: its presence is the success marker of a run.
    summary = dict(condition=condition, seed=seed, timesteps=policy.num_timesteps, final_eval=final)
    if learned:
        summary.update(asked=feedback.asked, labels=len(feedback.db), vlm=feedback.vlm)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    return final


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--condition", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides, e.g. pref.n_queries=700")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run(load_config(args.config, args.set), args.condition, args.seed, args.out)


if __name__ == "__main__":
    main()
