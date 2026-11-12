"""CPU-only checks; nothing here loads the VLM weights. Run: source scripts/env.sh && $PY -m pytest"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from prefrl import env as E
from prefrl import plot, run_all
from prefrl.reward_model import PrefDB, RewardEnsemble
from prefrl.train import labels_due, load_config, run
from prefrl.vlm import QwenJudge, clip_offsets, combine, parse

CONFIG = Path(__file__).parent.parent / "configs/default.yaml"


def rollout(n_envs=3, steps=400, seed=0):
    xp, rng = E.make_venv(n_envs, seed, steps), np.random.default_rng(seed)
    xp.reset()
    dones = [xp.step(rng.integers(4, size=n_envs))[2] for _ in range(steps)]
    return xp, rng, np.array(dones)


def test_episode_ends_are_hidden_but_logged():
    xp, _, dones = rollout()
    assert not dones.any()  # the agent never sees an episode end ...
    assert (xp.ep_step[:, 1:] == 0).sum() >= 3  # ... although random landers crashed and were reset
    crash = np.argwhere(xp.ep_step[:, 1:] == 0)[0]  # the step before a reset carries the crash penalty
    assert xp.rew[crash[0], crash[1]] == -100


def test_replay_renders_exactly_what_was_logged():
    xp, rng, _ = rollout()
    renderer, k = E.Renderer(), 75
    spans_reset = 0
    for env, start in xp.sample_pairs(rng, 6, k).reshape(-1, 2):
        ends = xp.ends(env, start, k)
        spans_reset += len(ends) > 0
        offsets = clip_offsets(k, 8, ends)
        sparse = renderer.frames(xp, env, start, k, offsets)  # raises if the replay leaves the logged trajectory
        assert sparse.shape == (8, 400, 600, 3) and sparse.dtype == np.uint8
        # Skipping frames must not change the picture (exhaust particles age only when rendered).
        assert (sparse == renderer.frames(xp, env, start, k)[offsets]).all()
    assert spans_reset  # the sample covered windows that cross a hidden reset


def test_replay_detects_a_trajectory_that_does_not_match():
    xp, _, _ = rollout(n_envs=1, steps=80)
    xp.act[0, 5:40] = 2  # pretend the lander fired its main engine: the log no longer matches the physics
    with pytest.raises(RuntimeError, match="diverged"):
        E.Renderer().frames(xp, 0, 0, 75)


def test_reward_ensemble_recovers_a_reward_from_preferences():
    """Oracle preferences over a hidden reward are enough to predict held-out preferences and the reward itself."""
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    def true_reward(obs, act):
        return 2 * obs[..., 0] - np.abs(obs[..., 1]) + (act == 2) * 0.5

    def pairs(n):
        obs = rng.normal(size=(n, 2, 20, 8)).astype(np.float32)
        act = rng.integers(4, size=(n, 2, 20))
        ret = true_reward(obs, act).sum(-1)
        return obs, act, (ret[:, 0] > ret[:, 1]).astype(np.float32)

    model, db = RewardEnsemble(8, 4), PrefDB(3, rng)
    for o, a, m in zip(*pairs(500)):
        db.add(o, a, m)
    weights = db.arrays()[3]
    assert 0.3 < (weights == 0).mean() < 0.44  # about 1/e of D is held out per member
    train, val = model.fit(db, epochs=60, batch_size=64, rng=rng)
    assert train < 0.45 and val < 0.6  # log 2 = 0.69 is chance

    obs, act, mu = pairs(400)
    with torch.no_grad():
        predicted = model.prefer_first(obs, act).mean(0).numpy() > 0.5
    assert (predicted == (mu == 1)).mean() > 0.9

    flat_obs, flat_act = obs.reshape(-1, 8), act.reshape(-1)
    model.set_normalisation(flat_obs, flat_act)
    r_hat = model.reward(flat_obs, flat_act)
    assert abs(r_hat.mean()) < 1e-3 and abs(r_hat.std() - 1) < 1e-3  # zero mean, unit std for the agent
    assert np.corrcoef(r_hat, true_reward(flat_obs, flat_act))[0, 1] > 0.9


def test_answer_parsing_and_swap_consistency():
    assert parse("Clip 1 drifts away, clip 2 lands.\nANSWER: 2") == "2"
    assert parse("answer: `tie`") == "TIE"
    assert parse("ANSWER: 1 ... on reflection\nANSWER: 2") == "2"  # the final line counts
    assert parse("I think the first one") == "UNSURE"
    assert combine("1", "2") == 1.0 and combine("2", "1") == 0.0  # same clip picked in both orders
    assert combine("TIE", "TIE") == 0.5
    assert combine("1", "1") is None  # always answering "1" is position bias, not a preference
    assert combine("1", "TIE") is None and combine("UNSURE", "UNSURE") is None
    assert combine("2") == 0.0 and combine("UNSURE") is None  # single-order mode


def test_clip_offsets_always_show_the_end_of_an_attempt():
    assert clip_offsets(75, 8).tolist() == [0, 11, 21, 32, 42, 53, 63, 74]
    with_end = clip_offsets(75, 8, ends=[37])
    assert 37 in with_end and len(with_end) == 8 and (np.diff(with_end) > 0).all()


def test_label_schedule_spends_the_budget_front_loaded():
    due = [labels_due(t, 1_000_000, 1400, 0.25, 0.1) for t in range(0, 1_100_000, 50_000)]
    assert due[0] == 350  # a quarter before RL starts
    assert due[20] == 1400 and due[21] == 1400  # all spent at the end, never more
    assert (np.diff(due) >= 0).all()
    assert due[10] - due[0] > 2 * (due[20] - due[10])  # the first half of training gets most of the rest


def test_judge_builds_model_inputs():
    transformers = pytest.importorskip("transformers")
    cfg = load_config(CONFIG)
    try:
        processor = transformers.AutoProcessor.from_pretrained(cfg["vlm"]["model"], revision=cfg["vlm"]["revision"])
    except OSError:
        pytest.skip("Qwen processor files not available offline")
    processor.tokenizer.padding_side = "left"
    clip = np.zeros((8, 400, 600, 3), np.uint8)
    judge = QwenJudge(None, processor, cfg["vlm"], cfg["prompts"], segment_len=75)
    inputs = judge.inputs([(clip, clip), (clip, clip)])
    # 8 frames of 392x588 -> 4 frame pairs x (28 x 42 patches / 4) tokens per clip, two clips per conversation
    assert (inputs["input_ids"] == processor.video_token_id).sum(1).tolist() == [2 * 4 * 294] * 2
    assert inputs["second_per_grid_ts"] == pytest.approx([2 * judge.dt] * 4)  # real frame spacing, not a default fps
    text = processor.tokenizer.decode(inputs["input_ids"][0]).replace("<|video_pad|>", "")
    assert text.index("Clip 1:") < text.index("Clip 2:") < text.index("ANSWER: 1") and "yellow flags" in text

    images = QwenJudge(None, processor, dict(cfg["vlm"], input_mode="images", frames_per_clip=6), cfg["prompts"], 75)
    inputs = images.inputs([(clip[:6], clip[:6])])
    assert (inputs["input_ids"] == processor.image_token_id).sum().item() == 12 * 294


class FakeJudge:
    """Stands in for the VLM: prefers the first clip, cannot tell on every third pair."""

    def label(self, pairs):
        assert all(clip.shape == (4, 400, 600, 3) for pair in pairs for clip in pair)
        return [(None if i % 3 == 0 else 1.0, ["ANSWER: 1", "ANSWER: 2"]) for i in range(len(pairs))]


def tiny_config():
    cfg = load_config(CONFIG)
    cfg.update(total_timesteps=1024, n_envs=2)
    cfg["ppo"].update(n_steps=256, batch_size=64, n_epochs=1)
    cfg["eval"].update(every=512, episodes=2, final_episodes=2, final_video_episodes=1)
    cfg["pref"].update(n_queries=24, segment_len=30, pretrain_steps=400, pretrain_epochs=2, round_every=512,
                       epochs_per_round=1)
    cfg["vlm"].update(frames_per_clip=4, save_clip_every=4)
    return cfg


def test_run_with_learned_reward_end_to_end(tmp_path):
    cfg = tiny_config()
    final = run(cfg, "mixed", seed=0, out=tmp_path, judge=FakeJudge())

    summary = json.loads((tmp_path / "summary.json").read_text())
    queries = [json.loads(line) for line in (tmp_path / "queries.jsonl").read_text().splitlines()]
    assert summary["asked"] == len(queries) == 24  # the whole label budget was spent
    by_vlm = [q for q in queries if q["labeler"] == "vlm"]
    assert 0 < len(by_vlm) < 24  # mixed: both raters were used
    assert all(q["mu"] == q["oracle_mu"] for q in queries if q["labeler"] == "oracle")
    unlabelled = sum(q["mu"] is None for q in queries)
    assert unlabelled > 0 and summary["labels"] == 24 - unlabelled  # "can't tell" answers never enter D
    assert len(np.load(tmp_path / "preferences.npz")["mu"]) == summary["labels"]

    rounds, evals = pd.read_csv(tmp_path / "rounds.csv"), pd.read_csv(tmp_path / "eval.csv")
    assert rounds.t.tolist() == [0, 512, 1024] and rounds.asked.iloc[0] == 6 and rounds.asked.iloc[-1] == 24
    assert evals.t.tolist() == [0, 512, 1024, 1024] and evals["mean"].iloc[-1] == final["mean"]
    assert len(list((tmp_path / "videos/eval").glob("*.mp4"))) == 4
    assert list((tmp_path / "videos/queries").glob("*.mp4")) and list((tmp_path / "videos/queries").glob("*_shown.png"))
    assert (tmp_path / "policy.zip").exists() and (tmp_path / "reward_model.pt").exists()
    assert "rollout/ep_rew_mean" in pd.read_csv(tmp_path / "progress.csv")  # true return of training episodes


def test_run_on_true_reward_end_to_end(tmp_path):
    run(tiny_config(), "pure_rl", seed=1, out=tmp_path)
    assert json.loads((tmp_path / "summary.json").read_text())["final_eval"]["episodes"] == 2
    assert not (tmp_path / "queries.jsonl").exists() and not (tmp_path / "rounds.csv").exists()


def test_run_all_then_plot(tmp_path, monkeypatch):
    """The launcher's path: config files on disk -> every (condition, seed) as a subprocess -> figures and table."""
    cfg = tiny_config()
    cfg.update(conditions={"pure_rl": {}, "oracle": {"oracle_fraction": 1.0}}, seeds=[0, 1])
    (tmp_path / "prompts.yaml").write_text(yaml.safe_dump(cfg.pop("prompts")))
    (tmp_path / "default.yaml").write_text(yaml.safe_dump(cfg))
    outputs = tmp_path / "outputs"
    (outputs / "oracle/seed_1").mkdir(parents=True)  # left behind by a run that died: must not be reused
    (outputs / "oracle/seed_1/eval.csv").write_text("t,mean\n0,-999\n")
    monkeypatch.setenv("PYTHONPATH", str(CONFIG.parent.parent / "src"))
    argv = ["run_all", "--config", str(tmp_path / "default.yaml"), "--out", str(outputs), "--logs", str(tmp_path / "logs")]
    monkeypatch.setattr(sys, "argv", [*argv, "--conditions", "pure_rl", "--set", "eval.final_episodes=3"])
    with pytest.raises(SystemExit):
        run_all.main()
    done = sorted(outputs.glob("*/seed_?/summary.json"))
    assert [path.parts[-3] for path in done] == ["pure_rl", "pure_rl"]  # only the requested condition ran ...
    assert json.loads(done[0].read_text())["final_eval"]["episodes"] == 3  # ... with the override applied
    stamp = done[0].stat().st_mtime_ns
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as finished:
        run_all.main()
    assert finished.value.code == 0 and done[0].stat().st_mtime_ns == stamp  # finished runs are not repeated
    assert len(list(outputs.glob("*/seed_?/summary.json"))) == 4 and len(list((tmp_path / "logs").glob("*.log"))) == 4
    assert len(list(outputs.glob("oracle/seed_1.incomplete-*"))) == 1
    assert pd.read_csv(outputs / "oracle/seed_1/eval.csv")["mean"].min() > -999

    plot.main(outputs)
    made = {path.name for path in (outputs / "plots").iterdir()}
    assert made == {"learning_curves.png", "learning_curves_per_seed.png", "training_return.png",
                    "final_performance.png", "reward_predictor.png", "summary.csv"}  # fmt: skip
    table = pd.read_csv(outputs / "plots/summary.csv")
    assert table.seeds.tolist() == [2, 2] and table.queries_asked.tolist() == [0, 24]
    assert (table.ci95_low < table.final_return_mean).all() and (table.final_return_mean < table.ci95_high).all()
    # two seeds one apart: the 95% t interval of their mean is +-12.706 * (sd / sqrt 2) = +-6.353
    assert plot.ci95(np.std([0.0, 1.0], ddof=1), 2) == pytest.approx(6.353) and plot.ci95(np.nan, 1) == 0
