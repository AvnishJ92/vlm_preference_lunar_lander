# Deep RL from preferences on LunarLander, with a VLM as the human

An implementation of Christiano et al. 2017, *Deep Reinforcement Learning from Human Preferences*
([arXiv:1706.03741](https://arxiv.org/abs/1706.03741)), on Gymnasium `LunarLander-v3`. The agent never sees the
environment reward. A reward predictor is fit to pairwise preferences over 1.5 s clips, and PPO maximises the
predicted reward. The preferences come from the true reward (the paper's synthetic oracle) or from
`Qwen/Qwen2.5-VL-7B-Instruct` watching the two clips (standing in for the paper's human raters).

Four conditions, five seeds each:

| condition | reward the agent optimises | who answers the queries |
|---|---|---|
| `pure_rl` | true environment reward | nobody |
| `oracle` | learned predictor | oracle: the clip with the higher true return wins |
| `vlm` | learned predictor | Qwen2.5-VL |
| `mixed` | learned predictor | each query: oracle with probability `oracle_fraction`, else Qwen2.5-VL |

**Status (2026-11-12):** gates passed, main experiment launched; see *External runs* at the bottom.

Gate results that shaped the configuration:

- **GPU smoke test** (run `...-7bf969`): the whole VLM path works. It also showed the describe-then-answer prompt
  running into its token limit, which led to the calibration grid below.
- **Prompt calibration** (run `...-7a4e47`, 200 clip pairs from landers of mixed skill, each asked in both orders):

  | variant | labelled | ties | agrees with oracle | correct / query | wrong / query | answers "1" |
  |---|---|---|---|---|---|---|
  | answer only, 8-frame video (**chosen**) | 59% | 0% | 76% | 0.45 | 0.14 | 31% |
  | answer only, 6 separate images | 69% | 0% | 73% | 0.50 | 0.19 | 47% |
  | answer only, 16-frame video | 54% | 1% | 76% | 0.41 | 0.13 | 28% |
  | answer only, 8 frames upscaled 1.5x | 51% | 0% | 76% | 0.39 | 0.12 | 26% |
  | one line per clip, then answer | 69% | 17% | 78% | 0.40 | 0.12 | 57% |
  | describe, compare, then answer | 72% | 30% | 80% | 0.34 | 0.09 | 47% |

  "Labelled" = the two orders gave the same answer. "Agrees" is among clear-cut labels. Differences in agreement
  are within noise (about 4 points); the answer-only video prompt gives the most signal per query and costs about
  0.4 s per model call against 1.0 s for the long prompt. The paper assumes a rater who is wrong about 5% of the
  time; this one is wrong about 24% of the time, and 87% right only when the two clips differ a lot.
- **Baselines at 1M timesteps** (run `...-7bf969`): PPO on the true reward ended at -18 to 159 (mean 89) and was
  still rising, so the experiment runs for 2M. Oracle-preference runs tracked or beat it at equal timesteps.

## How it follows the paper

| paper | here |
|---|---|
| Policy optimised on predicted reward; entropy bonus 0.01, gamma 0.995, lambda 0.97 (A.1) | same values, with PPO instead of TRPO |
| No episode-end signal reaches the agent (A) | `HiddenReset`: silent reset, never `done`; the crash/landing reward stays on the last step |
| Clips of 1.5 s (A.1) | 75 steps at 50 FPS, cut from the continuous stream, so a clip can contain a crash and the restart |
| Preference, tie, or can't tell (2.2.2) | VLM answers `1`, `2`, `TIE` or `UNSURE`; ties give mu = 0.5, can't-tell is dropped |
| Bradley-Terry cross-entropy on summed predicted reward (2.2.3, Eq. 1) | `RewardEnsemble.prefer_first` |
| Ensemble of 3, each on a bootstrap sample of D, 1/e held out (2.2.3, A) | Poisson(1) count per label and member; count 0 = that member's validation set |
| l2 adapted to keep validation loss at 1.1-1.5x training loss (2.2.3) | per member, after every epoch |
| 10% chance the rater answers at random (2.2.3) | `label_noise: 0.1` |
| Members normalised separately, averaged, zero mean / unit std (2.2.1, 2.2.3, A.1) | `RewardEnsemble.reward` |
| 10x candidate pairs, ask the ones with the highest ensemble variance (2.2.4, A) | `candidate_factor: 10` |
| 25% of comparisons on the untrained policy, then rate ~ tau/(T+tau) (A.1) | `pretrain_frac`, `anneal_tau_frac` |
| Two-layer, 64-unit, leaky-ReLU predictor on (observation, action) (A.1) | same |
| 350 / 700 / 1400 synthetic labels, 700 human (3.1.1) | `n_queries: 1400` |
| 5 runs per setting, evaluated on the true reward (Fig. 2) | 5 seeds; evaluation on standard episodic LunarLander |

Differences worth knowing:

- **The oracle/VLM mixing ratio is not from the paper.** The paper never mixes raters, so there is no ratio to copy.
  `conditions.mixed.oracle_fraction` is 0.5 as a placeholder.
- PPO replaces TRPO, and the predictor is trained in a feedback round after every PPO update (16,384 timesteps)
  rather than asynchronously.
- The policy and value networks each have one hidden layer of 64 tanh units (`ppo.policy_kwargs.net_arch`).
- The query budget counts queries asked. A VLM query that comes back "can't tell" is spent but adds no label, so the
  VLM conditions end with fewer labels than the oracle condition.
- Every VLM query is asked in both clip orders and kept only if the two answers agree (`swap_check`). This removes
  position bias at the cost of two model calls per query.
- LunarLander's fuel cost stays in the true reward (engine exhaust is visible in the clips).
- Label budget, annealing constant and predictor epochs are choices for LunarLander that were not tuned; the
  timestep budget (2M) was set from the 1M baselines above.

## Layout

```
configs/default.yaml     every hyperparameter, the four conditions, the calibration grid
configs/prompts.yaml     the VLM instructions (system prompt + answer-format variants)
src/prefrl/env.py        HiddenReset, the transition log, replay-based clip rendering, evaluation
src/prefrl/reward_model.py   preference database and reward ensemble
src/prefrl/vlm.py        Qwen2.5-VL judge: prompt building, answer parsing, swap consistency
src/prefrl/train.py      one (condition, seed) run
src/prefrl/run_all.py    all runs with bounded concurrency
src/prefrl/calibrate.py  prompt tuning against the oracle
src/prefrl/plot.py       figures and summary table
slurm/launch.sh          stages inside the node-40 allocation
tests/                   CPU-only tests
```

Clips are not stored during training. Every real episode is reset with a recorded seed, so any window can be
re-rendered later by replaying its actions; the replay is checked against the logged observations step by step.

## Running it

```bash
source scripts/env.sh && $PY -m pytest      # builds the locked venv outside the repo on first use; ~40 s of tests

slurm/launch.sh prepare                     # commit first; prints the new run directory <run>
slurm/launch.sh model <run>                 # ~17 GB of weights into $HF_HOME, once
slurm/launch.sh calibrate <run>             # GPU. Read <run>/outputs/calibration/calibration.csv
slurm/launch.sh smoke <run>                 # GPU. Passes if <run>/smoke/*/summary.json appears
slurm/launch.sh train <run> all             # the 20 runs (or e.g. `baselines --conditions pure_rl oracle`)
slurm/launch.sh plot <run>                  # can be run while training is in progress
```

`calibrate` scores each variant in `calibration.variants` on 200 clip pairs from landers of mixed skill. The column
to compare is `correct_per_query`; `agreement` near 0.5 means the VLM cannot judge these clips and the `vlm`
condition should not be expected to learn. To adopt a variant, copy its settings into `vlm:` in `default.yaml`,
commit, and `prepare` a new run.

GPU stages run as `srun --overlap` steps in Slurm job `ALLOC` (default 26911, the 1-GPU / 16-CPU allocation on
bodhanai-node040). Runs execute a read-only snapshot of the committed code under `<run>/code`, never this checkout.

## What a run writes

`<run>/outputs/<condition>/seed_<n>/`:

| file | content |
|---|---|
| `summary.json` | written last; its presence means the run finished. Final 100-episode evaluation, label counts, VLM agreement |
| `eval.csv` | true-reward evaluation every 50k steps (deterministic policy, 20 fixed episodes); last row is the final evaluation |
| `progress.csv` | PPO statistics per update; `rollout/ep_rew_mean` is the true return of the last 100 training episodes |
| `rounds.csv` | per feedback round: queries asked, labels held, predictor losses, predictor-vs-oracle accuracy on unseen pairs, r-hat vs true reward correlation, VLM agreement with the oracle |
| `queries.jsonl` | every query: the two segments, their true returns, the oracle's label, who answered, the VLM's raw replies, the label used |
| `videos/eval/*.mp4` | one evaluation episode per evaluation point and five from the final policy, return in the filename |
| `videos/queries/*.mp4`, `*_shown.png` | every 10th VLM query: both clips side by side, and the exact frames the VLM saw, with its answer and the oracle's in the filename |
| `policy.zip`, `reward_model.pt`, `preferences.npz` | final policy, reward ensemble, preference database |

`<run>/outputs/plots/` (curves are the mean over seeds with a 95% Student-t confidence band): `learning_curves.png`, `learning_curves_per_seed.png`, `training_return.png`,
`final_performance.png`, `reward_predictor.png`, `vlm_labels.png`, `summary.csv`.
Logs: `<run>/attempts/001/logs/`.

## External runs

- [20261112-prefrl-lunarlander-7bf969](</projects/data/ttsteam/avnish/artifacts/vlm_preference_lunar_lander/runs/20261112-prefrl-lunarlander-7bf969/summary.md>) — Deep RL from preferences on LunarLander-v3: PPO on the true reward vs a reward learned from oracle, Qwen2.5-VL-7B and mixed preferences; 5 seeds each; code `7bf96907a283`.

- [20261112-prefrl-lunarlander-7a4e47](</projects/data/ttsteam/avnish/artifacts/vlm_preference_lunar_lander/runs/20261112-prefrl-lunarlander-7a4e47/summary.md>) — Deep RL from preferences on LunarLander-v3: PPO on the true reward vs a reward learned from oracle, Qwen2.5-VL-7B and mixed preferences; 5 seeds each; code `7a4e4703fa02`.

- [20261112-prefrl-lunarlander-26ecf3](</projects/data/ttsteam/avnish/artifacts/vlm_preference_lunar_lander/runs/20261112-prefrl-lunarlander-26ecf3/summary.md>) — Deep RL from preferences on LunarLander-v3: PPO on the true reward vs a reward learned from oracle, Qwen2.5-VL-7B and mixed preferences; 5 seeds each; code `26ecf31cc15a`.
