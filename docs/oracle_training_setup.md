# Oracle Reward-Based Training Setup

In this project, the **`oracle`** condition implements the synthetic-oracle preference-learning setup from Christiano et al. (2017, *Deep Reinforcement Learning from Human Preferences*, [arXiv:1706.03741](https://arxiv.org/abs/1706.03741)).

Crucially, **`oracle` is not RL on the true environment reward** (that baseline is `pure_rl`). Instead, the PPO agent never sees the environment reward—it optimizes a learned reward predictor trained solely on pairwise preferences over 1.5-second clips, where each preference query is answered by comparing the clips' true returns.

---

## 1. Environment Wrapping & Experience Logging

Defined in [`src/prefrl/env.py`](../src/prefrl/env.py):

* **Hidden episode ends & oracle reward shaping (`HiddenReset`, lines 31–97; `stable_on_pad`, lines 18–28):** With `hide_episode_ends: true` ([`configs/default.yaml`](../configs/default.yaml#L6)), the wrapper intercepts `terminated` and `truncated`, silently resets the underlying `LunarLander-v3` environment, and returns `terminated = truncated = False` to PPO. This prevents termination signals from leaking the task objective. To eliminate credit smearing and hover/pad-camping hacks in the oracle reward (while leaving `evaluate()` on unmodified `LunarLander-v3`), `HiddenReset` applies four rules:
  1. **Pad idle reward (`PAD_NOOP_REWARD: +2.0`):** Each step stably on the pad (`stable_on_pad(prev_obs)` or asleep within `|x| <= 0.25`) with `action == 0` pays `+2.0`.
  2. **Pad thruster penalty (`PAD_FIRE_PENALTY: -3.0`):** Each step stably on the pad with `action != 0` pays `-3.0`.
  3. **Mid-air hover penalty (`HOVER_COAST_PENALTY: 1.0`, `HOVER_FIRE_PENALTY: 2.0`):** On non-crash mid-air steps (`leg1 = leg2 = 0`) with `vy >= -0.05`, subtracts `1.0` (`action == 0`) or `2.0` (`action != 0`) from the step reward.
  4. **75-step pad hold (`HOLD_STEPS: 75`):** Instead of immediately resetting with a single `+100` spike when Box2D sets `lander.awake = False` on the pad, holds the episode on the pad for `75` steps (`1` clip) before resetting. Crashes keep `-100.0`; sleeping off-pad (`|x| > 0.25`) terminates with `0.0`.
* **Transition buffer & reward swap (`Experience`, lines 100–159):** A `VecEnvWrapper` over `n_envs: 16` `DummyVecEnv` workers logs `(obs, act, true_rew, ep_seed, ep_step)` at every step. Because `reward_fn` is set to `reward_model.reward` ([`src/prefrl/train.py`](../src/prefrl/train.py#L216)), `step_wait` stores the shaped oracle reward in `self.rew` (for oracle labeling and diagnostics only) and returns the predicted reward $\hat{r}(o, a)$ to PPO.

---

## 2. Pre-training Before RL Starts

Defined in [`src/prefrl/train.py`](../src/prefrl/train.py#L220-L225):

Before `policy.learn(...)` begins:
1. The randomly initialized PPO policy is stepped for `pref.pretrain_steps: 16000` timesteps (`1,000` steps per env across 16 envs).
2. `Feedback.query` asks `pretrain_frac * n_queries` = `0.25 * 1400 = 350` initial preference queries from these random-policy transitions.
3. `Feedback.fit` trains the `RewardEnsemble` on those 350 labels for `pref.pretrain_epochs: 100` epochs and computes the initial reward normalization constants.

---

## 3. Feedback Loop, Acquisition & Oracle Labeling

Defined in [`src/prefrl/train.py`](../src/prefrl/train.py#L55-L189):

During training, `Loop._on_rollout_end` triggers a feedback round every `pref.round_every: 16384` timesteps (matching one PPO rollout of `16 envs × 1024 n_steps`).

* **Annealed query schedule (`labels_due`, lines 55–63):** After the initial 350 queries, the remaining `1,050` queries are spent at a rate proportional to $\frac{\tau}{t + \tau}$ with $\tau = 0.1 \times 2{,}000{,}000 = 200{,}000$ timesteps, exhausting the `1,400`-query budget at `total_timesteps: 2,000,000`.
* **Candidate sampling (`sample_pairs` in `env.py`, lines 137–145):** When `due > self.fb.asked`, `Feedback.query` samples `n * candidate_factor` (`10 * n`) pairs of `segment_len: 75`-step (1.5 s at 50 FPS) windows from steps recorded since the previous round (`self.mark`). Windows are sliced from the continuous stream and can span a `HiddenReset`, which puts crashes inside clips, while the 75-step pad hold allows pure landed clips (`+150` return for `action=0`).
* **Ensemble-variance acquisition (`Feedback.query`, lines 100–105):** The 3-member ensemble computes $P(\text{clip 1} \succ \text{clip 2})$ for all candidates. It logs held-out predictor accuracy (`pred_oracle_acc`) on all non-tie candidates, then selects the `n` pairs with the highest ensemble variance `p.var(0)`.
* **Oracle labeling (`Feedback.query`, lines 99–131):**
  * The oracle label is computed from the summed shaped rewards `ret` of each 75-step clip (rewarding pad rest with `action=0`, penalizing pad thruster taps and mid-air hovering):
    $$\mu = 0.5 \cdot (1 + \operatorname{sign}(R_1 - R_2)) \in \{0.0, 0.5, 1.0\}$$
  * Because `conditions.oracle.oracle_fraction` is `1.0` ([`configs/default.yaml`](../configs/default.yaml#L63)), `by_vlm` is empty: neither `QwenJudge` nor `Renderer` is instantiated, no frames are rendered during training, and all `1,400` queries are answered by the oracle and added to `PrefDB` (unlike VLM conditions, which drop `UNSURE` / inconsistent swap-check queries).

---

## 4. Reward Predictor Ensemble & Normalization

Defined in [`src/prefrl/reward_model.py`](../src/prefrl/reward_model.py):

* **Bootstrap database (`PrefDB`, lines 8–29):** Each `(obs_pair, act_pair, mu)` triple receives a `Poisson(1.0)` count per ensemble member. Triples with count `0` (~$1/e$ of the dataset) serve as that member's held-out validation set.
* **Architecture (`RewardEnsemble`, lines 31–55):** `3` independent MLPs taking `[obs, one_hot(act)]` through two 64-unit `LeakyReLU(0.01)` hidden layers to a scalar $\hat{r}_m(o, a)$.
* **Bradley–Terry objective with label noise (`prefer_first`, lines 72–78):** Sums each member's predicted rewards over the 75 steps of each segment and models:
  $$P_m(\text{clip 1} \succ \text{clip 2}) = (1 - \epsilon)\,\sigma(\hat{R}_{m,1} - \hat{R}_{m,2}) + \frac{\epsilon}{2}, \quad \epsilon = 0.1$$
  trained with cross-entropy against $\mu$ using Adam (`lr: 1e-3`, `batch_size: 64`) for `epochs_per_round: 10` epochs per round.
* **Adaptive L2 regularization (`fit`, lines 87–108):** Each member maintains its own L2 coefficient (initialized at `1e-4`, clamped to `[1e-7, 1e-1]`). After every epoch, if `val_loss > 1.5 * train_loss`, `l2` is multiplied by `1.1`; if `val_loss < 1.1 * train_loss`, `l2` is divided by `1.1`. (`torch.set_flush_denormal(True)` is enabled in [`src/prefrl/train.py`](../src/prefrl/train.py#L200) to avoid ~10× CPU slowdown from subnormal weights.)
* **Per-round re-normalization (`set_normalisation` & `reward`, lines 58–70):** At the end of `Feedback.fit`, `mean` and `std` per member plus overall `scale` are recomputed on the recent transitions since `self.mark`. During rollouts, `RewardEnsemble.reward` z-scores each member, averages across the 3 members, and divides by `scale` so PPO receives zero-mean, unit-std rewards.

---

## 5. PPO Policy & Evaluation

Defined in [`configs/default.yaml`](../configs/default.yaml#L9-L26) and [`src/prefrl/train.py`](../src/prefrl/train.py#L165-L236):

* **Policy:** Stable-Baselines3 `PPO("MlpPolicy")` on CPU with separate `[64, 64]` `tanh` actor/critic networks, `n_envs=16`, `n_steps=1024`, `batch_size=64`, `n_epochs=4`, `lr=3e-4`, `gamma=0.995`, `gae_lambda=0.97`, and `ent_coef=0.01`, trained for `2,000,000` timesteps.
* **Evaluation:** Every `50,000` timesteps (and at `0` and `2,000,000`), `Loop.evaluate` runs the deterministic policy on **standard episodic `LunarLander-v3`** (20 episodes during training, 100 at the end) using fixed evaluation seeds (`EVAL_SEED + i`) and records the **true** environment return and landing rate in `eval.csv`.
* **Observed outcome in this repo ([`docs/results.md`](results.md#L10-L44)):** Across 5 seeds, `oracle` obtained all 1,400/1,400 labels and achieved 71–78% held-out predictor accuracy against the oracle, with a mean final return of `103 [-8, 213]` and `38%` landings (seed 3 solved the task at `246`, while the other four seeds reached `25–127`, compared to `273 [266, 281]` for `pure_rl`).
