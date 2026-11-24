"""CPU-only checks; nothing here loads the VLM weights. Run: source scripts/env.sh && $PY -m pytest"""

import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from prefrl import env as E
from prefrl import plot, run_all
from prefrl.reward_model import PrefDB, RewardEnsemble
from prefrl.train import labels_due, load_config, run
from prefrl.vlm import QwenJudge, _side_by_side_video, clip_offsets, combine, parse

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
    crash = np.argwhere(xp.ep_step[:, 1:] == 0)[0]  # the 75-step crash hold carries CRASH_HOLD_PENALTY up to reset
    assert xp.rew[crash[0], crash[1]] == E.CRASH_HOLD_PENALTY


def test_episode_ends_can_be_left_visible():
    xp, rng = E.make_venv(2, 0, 300, hide_episode_ends=False), np.random.default_rng(0)
    xp.reset()
    dones = np.array([xp.step(rng.integers(4, size=2))[2] for _ in range(300)])
    assert dones.sum() >= 2 and ((xp.ep_step[:, 1:] == 0) == dones[:-1].T).all()  # a done is followed by a new episode
    E.Renderer().frames(xp, 0, 100, 75)  # and the log still replays exactly


def _heuristic_action(obs):
    """Simple PD controller from Gymnasium LunarLander to reach the pad cleanly in tests."""
    angle_targ = np.clip(obs[0] * 0.5 + obs[2] * 1.0, -0.4, 0.4)
    hover_targ = 0.55 * abs(obs[0])
    angle_todo = (angle_targ - obs[4]) * 0.5 - obs[5] * 1.0
    hover_todo = (hover_targ - obs[1]) * 0.5 - obs[3] * 0.5
    if obs[6] > 0.5 or obs[7] > 0.5:
        angle_todo = 0.0
        hover_todo = -obs[3] * 0.5
    if hover_todo > abs(angle_todo) and hover_todo > 0.05:
        return 2
    if angle_todo < -0.05:
        return 3
    if angle_todo > 0.05:
        return 1
    return 0


def test_pad_hold_and_oracle_shaping_rewards():
    # 1) Mid-air hovering incurs HOVER_COAST_PENALTY (action=0) and HOVER_FIRE_PENALTY (action!=0).
    for test_action, expected_penalty in ((0, E.HOVER_COAST_PENALTY), (2, E.HOVER_FIRE_PENALTY)):
        hr = E.HiddenReset(gym.make(E.ENV_ID), seed=0, hide=False)
        raw = gym.make(E.ENV_ID)
        obs, _ = hr.reset()
        raw.reset(seed=hr.ep_seed)
        while not (obs[6] < 0.5 and obs[7] < 0.5 and obs[3] >= -0.05 and abs(obs[0]) < 0.8):
            obs, *_ = hr.step(2)
            raw.step(2)
        _, r_hr, *_ = hr.step(test_action)
        _, r_raw, *_ = raw.step(test_action)
        assert r_raw - r_hr == pytest.approx(expected_penalty)

    # 2) Landing holds on the pad for HOLD_STEPS (beyond Box2D sleep), paying PAD_NOOP_REWARD for action=0.
    xp = E.make_venv(1, 0, 6000, hide_episode_ends=True)
    hr = xp.venv.envs[0]
    obs = xp.reset()
    for _ in range(2500):
        if E.stable_on_pad(obs[0]) and hr.ep_step < 850:
            break
        obs = xp.step(np.array([_heuristic_action(obs[0])]))[0]
    assert E.stable_on_pad(obs[0]) and hr.ep_step < 850

    saw_asleep_hold = False
    for _ in range(E.HOLD_STEPS + 10):
        obs = xp.step(np.array([0]))[0]
        if hr.ep_step == 0:
            break
        if not hr.env.unwrapped.lander.awake:
            saw_asleep_hold = True
    assert saw_asleep_hold and hr.ep_step == 0
    land_end_t = xp.t
    hold_slice = slice(land_end_t - (E.HOLD_STEPS - 1), land_end_t)
    assert np.allclose(xp.rew[0, hold_slice], E.PAD_NOOP_REWARD)
    assert (np.diff(xp.ep_step[0, land_end_t - E.HOLD_STEPS : land_end_t]) == 1).all()

    # 3) Firing a thruster while stably on the pad pays PAD_FIRE_PENALTY.
    for _ in range(2500):
        if E.stable_on_pad(obs[0]):
            break
        obs = xp.step(np.array([_heuristic_action(obs[0])]))[0]
    assert E.stable_on_pad(obs[0])
    xp.step(np.array([2]))
    assert xp.rew[0, xp.t - 1] == pytest.approx(E.PAD_FIRE_PENALTY)

    # 4) Replay renders a window spanning the 75-step pad hold and post-hold reset without diverging.
    frames = E.Renderer().frames(xp, 0, land_end_t - E.HOLD_STEPS + 5, E.HOLD_STEPS)
    assert frames.shape == (E.HOLD_STEPS, 400, 600, 3)

    # 5) Crashing holds for HOLD_STEPS (50) steps at CRASH_HOLD_PENALTY (-2.0/step = -100 total) with frozen obs,
    #    marks CRASHED when mark_out_of_view=True, and replays cleanly both across the crash and inside the hold.
    xp_crash = E.make_venv(1, 0, 500, hide_episode_ends=True)
    hr_crash = xp_crash.venv.envs[0]
    xp_crash.reset()
    for _ in range(400):
        xp_crash.step(np.array([0]))
        if xp_crash.t > 10 and hr_crash.ep_step == 0:
            break
    assert hr_crash.ep_step == 0
    crash_end = xp_crash.t
    assert np.allclose(xp_crash.rew[0, crash_end - E.HOLD_STEPS : crash_end], E.CRASH_HOLD_PENALTY)
    assert xp_crash.rew[0, crash_end - E.HOLD_STEPS : crash_end].sum() == pytest.approx(E.HOLD_STEPS * E.CRASH_HOLD_PENALTY)
    assert not np.isclose(xp_crash.rew[0, crash_end - E.HOLD_STEPS - 1], E.CRASH_HOLD_PENALTY)
    assert np.all(xp_crash.obs[0, crash_end - E.HOLD_STEPS + 1 : crash_end, 6:8] == 0.0)
    r = E.Renderer()
    span_start = crash_end - E.HOLD_STEPS - 5
    full_crash = r.frames(xp_crash, 0, span_start, E.HOLD_STEPS)
    offsets = clip_offsets(E.HOLD_STEPS, 16, xp_crash.ends(0, span_start, E.HOLD_STEPS))
    assert np.array_equal(full_crash[offsets], r.frames(xp_crash, 0, span_start, E.HOLD_STEPS, offsets))
    marked_crash = r.frames(xp_crash, 0, span_start, E.HOLD_STEPS, mark_out_of_view=True)
    changed_crash = (full_crash != marked_crash).any((1, 2, 3))
    assert not changed_crash[:5].any() and changed_crash[5:].all()
    in_hold_start = crash_end - E.HOLD_STEPS + 10
    assert r.frames(xp_crash, 0, in_hold_start, E.HOLD_STEPS - 15).shape == (E.HOLD_STEPS - 15, 400, 600, 3)


def test_sample_pairs_mixes_recent_and_history():
    xp, rng, _ = rollout(n_envs=2, steps=400, seed=1)
    pairs = xp.sample_pairs(rng, 200, k=75, since=200)
    starts = pairs[..., 1]
    assert ((starts >= 200).any(axis=1)).all()  # every pair includes a clip from the recent rollout
    has_hist = (starts < 200).any(axis=1)
    assert 0.15 < has_hist.mean() < 0.55  # half of clip-2 samples draw from full history [0, hi)
    assert (starts[:, 0] < 200).any() and (starts[:, 1] < 200).any()  # slot order is randomized
    for env, start in pairs.reshape(-1, 2):
        assert len(xp.ends(env, start, 75)) == 0  # no sampled clip crosses a reset
        assert not xp.hold[env, start : start + E.MIN_FLIGHT_STEPS].any()  # >= 0.5s active flight at start
        assert xp.obs[env, start, 6] < 0.5 and xp.obs[env, start, 7] < 0.5


def test_vlm_side_by_side_video_and_prompts():
    cfg = load_config(CONFIG)
    assert cfg["vlm"]["frames_per_clip"] == 16 and cfg["vlm"]["frame_scale"] == pytest.approx(0.7)
    assert "neither improves nor gets worse" not in cfg["prompts"]["variants"]["progress"]
    a = np.zeros((16, 400, 600, 3), np.uint8)
    b = np.full((16, 400, 600, 3), 20, np.uint8)
    stacked = _side_by_side_video(a, b, scale=cfg["vlm"]["frame_scale"])
    assert stacked.shape == (16, 280, 840, 3) and stacked.dtype == np.uint8
    assert not np.array_equal(stacked[:, :40, :150], 0)  # CLIP 1 label stamped in top-left
    assert not np.array_equal(stacked[:, :40, 420:570], 20)  # CLIP 2 label stamped in top-left of right half

    judge = QwenJudge(cfg["vlm"], cfg["prompts"], segment_len=cfg["pref"]["segment_len"])
    assert "CRASHED" in judge.system and "LANDER OUT OF VIEW" in judge.system
    msgs = judge.messages(a, b)
    image_items = [c for c in msgs[1]["content"] if c["type"] == "image_url"]
    assert len(image_items) == 16
    assert image_items[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_replay_renders_exactly_what_was_logged():
    xp, rng, _ = rollout()
    renderer, k = E.Renderer(), 75
    for env, start in xp.sample_pairs(rng, 6, k).reshape(-1, 2):
        ends = xp.ends(env, start, k)
        assert len(ends) == 0  # sampled clips never cross a hidden reset
        offsets = clip_offsets(k, 8, ends)
        sparse = renderer.frames(xp, env, start, k, offsets)  # raises if the replay leaves the logged trajectory
        assert sparse.shape == (8, 400, 600, 3) and sparse.dtype == np.uint8
        # Skipping frames must not change the picture (exhaust particles age only when rendered).
        assert (sparse == renderer.frames(xp, env, start, k)[offsets]).all()
    # Also verify Renderer replays accurately across an explicit episode boundary
    reset_step = int(np.flatnonzero(xp.ep_step[0, 1:] == 0)[0]) + 1
    start = max(0, reset_step - 30)
    ends = xp.ends(0, start, k)
    assert len(ends) > 0
    offsets = clip_offsets(k, 8, ends)
    assert (renderer.frames(xp, 0, start, k, offsets) == renderer.frames(xp, 0, start, k)[offsets]).all()


def test_out_of_view_frames_are_marked_only_when_asked():
    xp = E.make_venv(1, 0, 400)
    xp.reset()
    for _ in range(400):
        xp.step(np.array([2]))  # main engine only: the lander climbs out of the picture
    gone = E.out_of_view(xp.obs[0])
    assert gone.any() and not gone[:20].any()
    start = int(np.flatnonzero(gone)[0]) - 10  # a window that begins in view and ends out of it
    renderer = E.Renderer()
    plain, marked = renderer.frames(xp, 0, start, 30), renderer.frames(xp, 0, start, 30, mark_out_of_view=True)
    changed = (plain != marked).any((1, 2, 3))
    # frame j shows the state after action j, which is what the next logged observation records
    assert not changed[:8].any() and (changed[:-1] == gone[start + 1 : start + 30]).all()


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


def test_judge_api_requests_and_swap_check():
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading

    seen = []
    lock = threading.Lock()

    def make_handler(server_tag):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                assert self.path == "/v1/chat/completions"
                length = int(self.headers["Content-Length"])
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                with lock:
                    seen.append((server_tag, body))
                # Pick whichever side has higher mean pixel brightness (left=1, right=2)
                n_imgs = sum(1 for c in body["messages"][1]["content"] if c["type"] == "image_url")
                assert body["temperature"] == 0.0 and n_imgs == 16
                reply = "ANSWER: 1" if len(seen) % 2 == 1 else "ANSWER: 2"
                resp = json.dumps({"choices": [{"message": {"content": reply}}]}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)

            def log_message(self, *_):
                pass

        return Handler

    s1 = ThreadingHTTPServer(("127.0.0.1", 0), make_handler("s1"))
    s2 = ThreadingHTTPServer(("127.0.0.1", 0), make_handler("s2"))
    t1 = threading.Thread(target=s1.serve_forever, daemon=True)
    t2 = threading.Thread(target=s2.serve_forever, daemon=True)
    t1.start(), t2.start()
    try:
        cfg = load_config(CONFIG)
        cfg["vlm"]["api_urls"] = [f"http://127.0.0.1:{s1.server_port}/v1", f"http://127.0.0.1:{s2.server_port}/v1"]
        cfg["vlm"]["concurrency"] = 4
        clip = np.zeros((16, 400, 600, 3), np.uint8)
        judge = QwenJudge(cfg["vlm"], cfg["prompts"], segment_len=75)
        results = judge.label([(clip, clip)])
        assert len(seen) == 2  # (A, B) and (B, A) swap check
        assert {tag for tag, _ in seen} == {"s1", "s2"}  # round-robins across both API URLs
        assert results == [(1.0, ["ANSWER: 1", "ANSWER: 2"])]
    finally:
        s1.shutdown(), s2.shutdown()
        s1.server_close(), s2.server_close()


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
    cfg["pref"].update(n_queries=24, budget_by_labels=False, segment_len=30, pretrain_steps=400, pretrain_epochs=2,
                       round_every=512, epochs_per_round=1)
    cfg["vlm"].update(frames_per_clip=4, save_clip_every=4)
    return cfg


def test_budget_by_labels_retries_until_target(tmp_path):
    cfg = tiny_config()
    cfg["pref"].update(budget_by_labels=True, max_retry_factor=4)
    run(cfg, "vlm", seed=0, out=tmp_path, judge=FakeJudge())
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["labels"] >= 24 and summary["asked"] > 24


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
