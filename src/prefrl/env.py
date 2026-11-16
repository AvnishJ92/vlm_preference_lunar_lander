"""LunarLander with hidden episode ends (paper App. A), a transition log, and exact re-rendering of any logged window."""

import gymnasium as gym
import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnvWrapper

ENV_ID = "LunarLander-v3"
FPS = 50
HOLD_STEPS = 75  # steps to hold on the pad after landing (one 1.5s clip at 50 FPS)
PAD_NOOP_REWARD = 2.0  # step reward when stably on the pad with action == 0
PAD_FIRE_PENALTY = -3.0  # step reward when stably on the pad with action != 0
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
      4. Instead of resetting immediately on Box2D sleep with a single +100 spike, hold on the pad for `hold_steps`
         (75 steps = 1 clip) before terminating/resetting so clips capture pure landed trajectories.

    With hide=False the episode ends are passed through as usual (the standard, easier task; used to measure what
    hiding them costs) and the caller resets.
    """

    def __init__(self, env, seed, hide=True, hold_steps=HOLD_STEPS):
        super().__init__(env)
        self._seeds, self.hide, self.hold_steps = np.random.default_rng(seed), hide, hold_steps

    def reset(self, **_):
        self.ep_seed = int(self._seeds.integers(2**31))
        self.ep_step, self.ep_return, self.pad_steps = 0, 0.0, 0
        obs, info = self.env.reset(seed=self.ep_seed)
        self._prev_obs = obs
        return obs, info

    def step(self, action):
        prev_obs = self._prev_obs
        obs, reward, terminated, truncated, _ = self.env.step(action)
        self._prev_obs = obs

        unwrapped = self.env.unwrapped
        crashed = bool(unwrapped.game_over or abs(obs[0]) >= 1.0)
        asleep = bool(not unwrapped.lander.awake and not crashed)
        on_pad = stable_on_pad(prev_obs) or (asleep and abs(obs[0]) <= 0.25)

        if crashed:
            reward = -100.0
            terminated = True
        elif on_pad:
            reward = PAD_NOOP_REWARD if int(action) == 0 else PAD_FIRE_PENALTY
            if stable_on_pad(obs) or (asleep and abs(obs[0]) <= 0.25):
                self.pad_steps += 1
            else:
                self.pad_steps = 0
            terminated = self.pad_steps >= self.hold_steps
        else:
            self.pad_steps = 1 if stable_on_pad(obs) else 0
            if asleep:
                # Settled outside the pad: terminate without awarding the +100 landing bonus.
                reward = 0.0
                terminated = True
            elif prev_obs[6] < 0.5 and prev_obs[7] < 0.5 and prev_obs[3] >= -0.05:
                reward -= HOVER_FIRE_PENALTY if int(action) != 0 else HOVER_COAST_PENALTY

        info = {"ep_seed": self.ep_seed, "ep_step": self.ep_step}
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
        self.t += 1
        if self.reward_fn is not None:
            rew = self.reward_fn(self._obs, self._act)
        self._obs = obs
        return obs, rew, done, infos

    def sample_pairs(self, rng, n, k, since=0):
        """n random pairs of k-step windows from steps >= since. Returns (n, 2, 2) of [env, start].

        Windows are cut from the continuous stream, so they may span a hidden reset (as in the paper's Atari setup);
        that is what puts crashes and landings inside clips often enough to learn their reward.
        """
        env = rng.integers(self.num_envs, size=(n, 2))
        start = rng.integers(max(since, 0), self.t - k + 1, size=(n, 2))
        return np.stack([env, start], -1)

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
    """True where the lander is outside the rendered picture. obs[..., 0] spans the picture width as -1..1; the top
    edge is at obs[..., 1] = 1.41 (viewport height 13.33 world units, pad at 3.33, legs 0.6 long, scaled by 6.67)."""
    return (np.abs(obs[..., 0]) > 1) | (obs[..., 1] > 1.41)


def _stamp_out_of_view(frame):
    """Write on a frame what someone watching the whole video sees for themselves: the lander has left the picture."""
    image = Image.fromarray(frame)
    ImageDraw.Draw(image).text((frame.shape[1] // 2, 60), "LANDER OUT OF VIEW", fill=(255, 60, 60), anchor="mm",
                               font=ImageFont.load_default(size=40))  # fmt: skip
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

    def frames(self, xp, env, start, k, offsets=None, mark_out_of_view=False):
        """uint8 (len(offsets), 400, 600, 3) for the window's steps `offsets` (default: all k).

        Frame j shows the world right after action j (its engine flames included). With mark_out_of_view, frames in
        which the lander is outside the picture carry the words LANDER OUT OF VIEW.
        """
        wanted = set(range(k) if offsets is None else np.asarray(offsets).tolist())
        lander, out = self.env.unwrapped, []
        for j in range(k):
            g = start + j
            s = xp.ep_step[env, g]
            if j == 0 or s == 0:  # (re)enter an episode: replay it up to this step
                obs, _ = self.env.reset(seed=int(xp.ep_seed[env, g]))
                for a in xp.act[env, g - s : g]:
                    obs = self.env.step(int(a))[0]
                    _tick_particles(lander)
            if not np.allclose(obs, xp.obs[env, g], atol=1e-5):
                raise RuntimeError(f"replay diverged from the log at env {env} step {g}")
            obs = self.env.step(int(xp.act[env, g]))[0]
            if j in wanted:
                frame = self.env.render()
                out.append(_stamp_out_of_view(frame) if mark_out_of_view and out_of_view(obs) else frame)
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
