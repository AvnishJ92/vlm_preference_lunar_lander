"""LunarLander with hidden episode ends (paper App. A), a transition log, and exact re-rendering of any logged window."""

import gymnasium as gym
import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnvWrapper

ENV_ID = "LunarLander-v3"
FPS = 50
HOLD_STEPS = 50  # 1.0s hold after landing or crashing (50 steps * +-2.0 = +-100.0 total)
MIN_FLIGHT_STEPS = 25  # 0.5s minimum active flight at the start of every 1.5s (75-step) clip
PAD_NOOP_REWARD = 2.0  # step reward when stably on the pad with action == 0
PAD_FIRE_PENALTY = -3.0  # step reward when stably on the pad with action != 0
CRASH_HOLD_PENALTY = -2.0  # step reward during the 50-step crash hold (total -100.0)
HOVER_COAST_PENALTY = 1.0  # subtracted per mid-air step when vy >= -0.05 and action == 0
HOVER_FIRE_PENALTY = 2.0  # subtracted per mid-air step when vy >= -0.05 and action != 0


def stable_on_pad(obs):
    """True when the lander is resting upright and stationary on both legs between the pad flags."""
    return bool(
        obs[6] > 0.5
        and obs[7] > 0.5
        and abs(obs[0]) <= 0.25
        and abs(obs[4]) < 0.2
        and abs(obs[2]) < 0.2
        and abs(obs[3]) < 0.2
        and abs(obs[5]) < 0.2
    )


class HiddenReset(gym.Wrapper):
    """Never reports done: on a real episode end we reset silently and hand back the next episode's first observation.

    Paper App. A: termination conditions leak the task, so the agent gets one continuous episode. Each real episode is
    reset with an explicit seed so that it can be replayed, and therefore rendered, later.

    Oracle reward shaping for training trajectories and preference clips:
      1. Stably on the pad with action == 0 pays `PAD_NOOP_REWARD` (+2.0) per step.
      2. Stably on the pad with action != 0 pays `PAD_FIRE_PENALTY` (-3.0) per step.
      3. Mid-air hovering/climbing (both legs off ground and vy >= -0.05) incurs a hover penalty.
      4. Instead of resetting immediately on Box2D sleep (+100 spike) or crash (-100 spike), hold on the pad or at
         the frozen crash state for `hold_steps` (75 steps = 1 clip, +150 or -150 total) before terminating/resetting.

    With hide=False the episode ends are passed through as usual (the standard, easier task; used to measure what
    hiding them costs) and the caller resets.
    """

    def __init__(self, env, seed, hide=True, hold_steps=HOLD_STEPS):
        super().__init__(env)
        self._seeds, self.hide, self.hold_steps = np.random.default_rng(seed), hide, hold_steps

    def reset(self, **_):
        self.ep_seed = int(self._seeds.integers(2**31))
        self.ep_step, self.ep_return, self.pad_steps = 0, 0.0, 0
        self.crash_steps, self._crash_obs = 0, None
        obs, info = self.env.reset(seed=self.ep_seed)
        self._prev_obs = obs
        return obs, info

    def step(self, action):
        if self._crash_obs is not None:
            obs = self._crash_obs.copy()
            self._prev_obs = obs
            reward = CRASH_HOLD_PENALTY
            self.crash_steps += 1
            terminated = self.crash_steps >= self.hold_steps
            truncated = False
        else:
            prev_obs = self._prev_obs
            obs, reward, terminated, truncated, _ = self.env.step(action)
            unwrapped = self.env.unwrapped
            crashed = bool(unwrapped.game_over or abs(obs[0]) >= 1.0)
            asleep = bool(not unwrapped.lander.awake and not crashed)
            on_pad = stable_on_pad(prev_obs) or (asleep and abs(obs[0]) <= 0.25)

            if crashed:
                obs = obs.copy()
                # Clear leg contact flags so a hard vertical crash onto the pad cannot look like a landed state.
                obs[6:8] = 0.0
                self._crash_obs = obs.copy()
                self._prev_obs = obs
                self.crash_steps = 1
                reward = CRASH_HOLD_PENALTY
                terminated = self.crash_steps >= self.hold_steps
            elif on_pad:
                self._prev_obs = obs
                reward = PAD_NOOP_REWARD if int(action) == 0 else PAD_FIRE_PENALTY
                if stable_on_pad(obs) or (asleep and abs(obs[0]) <= 0.25):
                    self.pad_steps += 1
                else:
                    self.pad_steps = 0
                terminated = self.pad_steps >= self.hold_steps
            else:
                self._prev_obs = obs
                self.pad_steps = 1 if stable_on_pad(obs) else 0
                if asleep:
                    # Settled outside the pad: terminate without awarding the +100 landing bonus.
                    reward = 0.0
                    terminated = True
                elif prev_obs[6] < 0.5 and prev_obs[7] < 0.5 and prev_obs[3] >= -0.05:
                    reward -= HOVER_FIRE_PENALTY if int(action) != 0 else HOVER_COAST_PENALTY

        info = {
            "ep_seed": self.ep_seed,
            "ep_step": self.ep_step,
            "hold": bool(self.crash_steps > 0 or self.pad_steps > 0),
        }
        self.ep_step += 1
        self.ep_return += reward
        if terminated or truncated:
            # SB3 averages info["episode"] into rollout/ep_rew_mean: the true return of real training episodes.
            info["episode"] = {"r": self.ep_return, "l": self.ep_step}
            if self.hide:
                obs, _ = self.reset()
                terminated = truncated = False
        return obs, float(reward), terminated, truncated, info


class Experience(VecEnvWrapper):
    """Logs every transition of every env and, when `reward_fn` is set, swaps the reward the agent sees for r-hat.

    Arrays are (n_envs, capacity, ...). `rew` is the true reward: read only by the oracle labeller and diagnostics.
    """

    def __init__(self, venv, capacity, reward_fn=None):
        super().__init__(venv)
        n = venv.num_envs
        self.obs = np.zeros((n, capacity, *venv.observation_space.shape), np.float32)
        self.act = np.zeros((n, capacity), np.int64)
        self.rew = np.zeros((n, capacity), np.float32)
        self.ep_seed = np.zeros((n, capacity), np.int64)
        self.ep_step = np.zeros((n, capacity), np.int64)
        self.hold = np.zeros((n, capacity), bool)
        self.t = 0  # steps logged per env
        self.reward_fn = reward_fn

    def reset(self):
        self._obs = self.venv.reset()
        return self._obs

    def step_async(self, actions):
        self._act = actions
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rew, done, infos = self.venv.step_wait()
        t = self.t
        self.obs[:, t], self.act[:, t], self.rew[:, t] = self._obs, self._act, rew
        self.ep_seed[:, t] = [i["ep_seed"] for i in infos]
        self.ep_step[:, t] = [i["ep_step"] for i in infos]
        self.hold[:, t] = [i.get("hold", False) for i in infos]
        self.t += 1
        if self.reward_fn is not None:
            rew = self.reward_fn(self._obs, self._act)
        self._obs = obs
        return obs, rew, done, infos

    def sample_pairs(self, rng, n, k, since=0):
        """n random pairs of k-step windows. Returns (n, 2, 2) of [env, start].

        Every sampled window stays within a single episode (no mid-clip reset) and begins with at least
        `MIN_FLIGHT_STEPS` (0.5s) of active mid-air flight before any crash or landing hold. When `since > 0`, pairs
        combine one clip from the recent rollout (`>= since`) with 50% recent clips and 50% historical clips from
        `[0, t - k + 1)` in randomized slot order.
        """
        hi = self.t - k + 1
        mf = min(MIN_FLIGHT_STEPS, max(1, k // 2))
        no_reset = (self.ep_step[:, :hi] + (k - 1)) == self.ep_step[:, k - 1 : self.t]
        in_air = (self.obs[:, :hi, 6] < 0.5) & (self.obs[:, :hi, 7] < 0.5)
        active_start = (~self.hold[:, :hi]) & (~self.hold[:, mf - 1 : hi + mf - 1]) & in_air
        valid = no_reset & active_start

        def _draw(lo, count):
            sub = valid[:, lo:]
            if not sub.any():
                sub = no_reset[:, lo:] if no_reset[:, lo:].any() else np.ones((self.num_envs, hi - lo), bool)
            coords = np.argwhere(sub)
            coords[:, 1] += lo
            return coords[rng.integers(len(coords), size=count)]

        if since <= 0:
            return _draw(0, 2 * n).reshape(n, 2, 2)
        c0 = _draw(since, n)
        n_recent = n // 2
        c1 = rng.permutation(np.concatenate([_draw(since, n_recent), _draw(0, n - n_recent)], axis=0))
        pairs = np.stack([c0, c1], axis=1)
        swap = rng.random(n) < 0.5
        pairs[swap] = pairs[swap, ::-1]
        return pairs

    def segments(self, where, k):
        """where (..., 2) of [env, start] -> obs (..., k, d), act (..., k), true return (...)."""
        env, idx = where[..., 0, None], where[..., 1, None] + np.arange(k)
        return self.obs[env, idx], self.act[env, idx], self.rew[env, idx].sum(-1)

    def recent(self, n):
        """Flat (obs, act, true reward) of the last n steps of every env."""
        s = slice(max(self.t - n, 0), self.t)
        return self.obs[:, s].reshape(-1, self.obs.shape[-1]), self.act[:, s].ravel(), self.rew[:, s].ravel()

    def ends(self, env, start, k):
        """Offsets within a window after which a hidden reset happened (the last step of a real episode)."""
        return np.flatnonzero(self.ep_step[env, start + 1 : start + k] == 0)


def make_venv(n_envs, seed, capacity, reward_fn=None, hide_episode_ends=True):
    def make(i):
        return lambda: HiddenReset(gym.make(ENV_ID), seed=[seed, i], hide=hide_episode_ends)

    return Experience(DummyVecEnv([make(i) for i in range(n_envs)]), capacity, reward_fn)


def out_of_view(obs):
    """True where the lander has left the rendered picture. obs[..., 0] spans the picture width as -1..1; the lander
    spawns at obs[..., 1] ~= 1.40 and its legs clear the top edge of the 400px viewport above obs[..., 1] = 1.50."""
    return (np.abs(obs[..., 0]) > 1) | (obs[..., 1] > 1.5)


def _stamp_banner(frame, text, color=(255, 60, 60), size=40):
    """Write a status banner at top center so a frame carries out-of-view, crash, or landed state visually."""
    image = Image.fromarray(frame)
    ImageDraw.Draw(image).text((frame.shape[1] // 2, 65), text, fill=color, anchor="mm",
                               font=ImageFont.load_default(size=size))  # fmt: skip
    return np.asarray(image)


def _tick_particles(lander):
    """What LunarLander.render() does to exhaust particles, minus the drawing, so skipped frames age them the same."""
    for p in lander.particles:
        p.ttl -= 0.15
    lander._clean_particles(False)


class Renderer:
    """Renders logged experience by replaying it: same episode seed + same actions = same physics.

    This avoids rendering (or storing frames for) the millions of steps nobody looks at; every replayed observation
    is checked against the log, so a clip can never show something other than what the agent did.
    """

    def __init__(self):
        self.env = gym.make(ENV_ID, render_mode="rgb_array")

    def frames(self, xp, env, start, k, offsets=None, mark_out_of_view=False, mark_landed=False, mark_firing_on_pad=False):
        """uint8 (len(offsets), 400, 600, 3) for the window's steps `offsets` (default: all k).

        Frame j shows the world right after action j (its engine flames included), freezing physics during any crash
        hold. With mark_out_of_view, frames outside the picture carry CRASHED and crash-hold frames
        carry CRASHED. With mark_landed, frames stably landed on the pad with engines off (or asleep on pad)
        carry LANDED in green. With mark_firing_on_pad, frames stably on the pad while firing thrusters
        (action != 0) carry FIRING ON PAD in orange-red.
        """
        wanted = set(range(k) if offsets is None else np.asarray(offsets).tolist())
        lander, out, crashed = self.env.unwrapped, [], False

        def _advance(action):
            nonlocal obs, crashed
            if crashed:
                return
            obs = self.env.step(int(action))[0]
            crashed = bool(lander.game_over or abs(obs[0]) >= 1.0)
            if crashed:
                obs = obs.copy()
                obs[6:8] = 0.0

        for j in range(k):
            g = start + j
            s = xp.ep_step[env, g]
            if j == 0 or s == 0:  # (re)enter an episode: replay it up to this step
                obs, _ = self.env.reset(seed=int(xp.ep_seed[env, g]))
                crashed = False
                for a in xp.act[env, g - s : g]:
                    _advance(a)
                    _tick_particles(lander)
            if not np.allclose(obs, xp.obs[env, g], atol=1e-5):
                raise RuntimeError(f"replay diverged from the log at env {env} step {g}")
            act = int(xp.act[env, g])
            _advance(act)
            if j in wanted:
                frame = self.env.render()
                if mark_out_of_view:
                    if out_of_view(obs) or crashed:
                        frame = _stamp_banner(frame, "CRASHED")
                if not crashed and not out_of_view(obs):
                    asleep = not lander.lander.awake
                    if mark_landed and ((stable_on_pad(obs) and act == 0) or (asleep and abs(obs[0]) <= 0.25)):
                        frame = _stamp_banner(frame, "LANDED", color=(60, 235, 90), size=64)
                    elif mark_firing_on_pad and stable_on_pad(obs) and act != 0:
                        frame = _stamp_banner(frame, "FIRING ON PAD", color=(255, 110, 30), size=52)
                out.append(frame)
            else:
                _tick_particles(lander)
        return np.stack(out)


def write_video(path, frames, fps=FPS):
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, frames, fps=fps, macro_block_size=1)  # keep 600x400 instead of padding to 16s


def evaluate(policy, seeds, video_dir=None, n_videos=0, tag=""):
    """True-reward returns of the deterministic policy on the standard (episodic) env, one episode per seed.

    Returns (returns, lengths, landed) arrays; the first n_videos episodes are written to video_dir as mp4.
    """
    env = gym.make(ENV_ID, render_mode="rgb_array" if n_videos else None)
    rets, lens, landed = [], [], []
    for i, seed in enumerate(seeds):
        obs, _ = env.reset(seed=int(seed))
        ret, n, done, frames = 0.0, 0, False, []
        while not done:
            obs, r, terminated, truncated, _ = env.step(int(policy.predict(obs, deterministic=True)[0]))
            ret, n, done = ret + r, n + 1, terminated or truncated
            if i < n_videos:
                frames.append(env.render())
        if frames:
            write_video(video_dir / f"{tag}_ep{i}_return{ret:+.0f}.mp4", frames)
        rets.append(ret), lens.append(n), landed.append(terminated and r == 100)
    env.close()
    return np.array(rets), np.array(lens), np.array(landed)
