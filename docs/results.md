# Results

Run `20261112-prefrl-lunarlander-42961b`, 12-13 November 2026. Four conditions, five seeds each, 2M timesteps per
run, 1,400 preference queries per learned-reward run, policy and value networks with two hidden layers of 64.
Method and configuration are in the [README](../README.md). The first attempt, with a one-layer policy, is kept
[at the end](#first-attempt-superseded-one-hidden-layer-policy).

## Summary

| condition | final return, mean [95% CI] | per seed | successful landings | labels obtained |
|---|---|---|---|---|
| PPO on true reward | 273 [266, 281] | 273, 278, 264, 273, 279 | 97% | none needed |
| Oracle preferences | 103 [-8, 213] | 25, 127, 72, 246, 43 | 38% | 1,400 of 1,400 |
| VLM preferences (Qwen2.5-VL-7B) | 53 [-177, 282] | -107, 4, 259, 236, -128 | 37% | 586 of 1,400 on average |
| Oracle + VLM, 50/50 | 173 [54, 293] | 6, 248, 183, 207, 222 | 56% | 1,141 of 1,400 on average |

Final return is the mean over 100 evaluation episodes of the deterministic final policy, scored with the true
environment reward; 200 counts as solved. Intervals are Student-t 95% intervals of the mean over the five seeds.

- PPO on the true reward solves the task in every seed.
- Learning from preferences works in some seeds and not in others, for every rater. The oracle solves it in one
  seed of five, the VLM alone in two, the 50/50 mix in three (a fourth reaches 183).
- The only difference the intervals support is that the true reward beats oracle preferences. The three
  learned-reward conditions cannot be ranked with five seeds, and neither the VLM nor the mix can be separated
  from the true-reward baseline, because their intervals are so wide.
- The VLM agrees with the oracle on only about half of its labels, yet two VLM-only seeds land in 90% of episodes.
  The VLM teaches a different reward from the environment's, and that reward can still lead to landing.

![Evaluation return during training](../results/plots/learning_curves.png)

![Final return and landing rate](../results/plots/final_performance.png)

Other figures in [results/plots/](../results/plots/): `learning_curves_per_seed.png`, `training_return.png`,
`reward_predictor.png`, `vlm_labels.png`, and the table `summary.csv`.

## What happened in each condition

**True reward.** All five seeds pass 200 between 0.46M and 0.70M timesteps and end at 264-279.

**Oracle preferences.** All 1,400 queries were answered. The reward predictor ranks unseen clip pairs like the
oracle 71-78% of the time. Seed 3 is solved by 0.46M timesteps (246, 89% landings). The other four improve slowly
to 25-127 and land in 0-53% of episodes. The paper reports oracle labels close to RL;
this implementation does not reach that, and the cause was not found (label budget, label schedule and predictor
settings were not tuned; changing the l2 coefficient did not change held-out accuracy).

**VLM preferences.** 42% of queries produced a label, because both clip orders have to give the same answer.

| seed | final return | landings | what the final policy does |
|---|---|---|---|
| 2 | 259 | 91% | lands on the pad |
| 3 | 236 | 90% | lands on the pad |
| 1 | 4 | 0% | reaches the pad and stays there without shutting down until the 1,000-step limit; it scored 179 at 1.75M |
| 0 | -107 | 0% | falls and crashes next to the pad within about 70 steps |
| 4 | -128 | 2% | crashes within about 160 steps |

No seed flies out of view any more (the worst single episode of the final evaluations is -377; in the first attempt four seeds averaged
-1,600 to -4,800). The out-of-view marker fixed that failure.

**Oracle + VLM.** Half the queries went to each rater. Seeds 1, 3 and 4 end at 207-248 with 66-90% landings. Seed 2
ends at 183 (best evaluation 267) and seed 0 at 6; seed 0 sits on the pad without shutting down, and most of seed
2's episodes also run close to the time limit. The VLM's labels
in these runs agree with the oracle only 39-46% of the time.

## What the VLM's labels are

**Agreement with the oracle falls to chance or below once training starts.**

| VLM-answered queries | phase | queries | got a label | agrees with oracle |
|---|---|---|---|---|
| VLM-only runs | before RL (random policy) | 1,750 | 41% | 68% of 703 |
| | first 0.5M timesteps | 2,715 | 31% | 51% of 830 |
| | after 0.5M timesteps | 2,535 | 54% | 48% of 1,374 |
| mixed runs | before RL (random policy) | 884 | 41% | 72% of 355 |
| | first 0.5M timesteps | 1,364 | 60% | 42% of 818 |
| | after 0.5M timesteps | 1,270 | 82% | 30% of 1,036 |

Ties are almost gone (35 of 10,518 VLM queries; the first attempt had over 1,300).

**The VLM rewards where the lander ends up; the environment rewards how much it improved.** LunarLander's reward
pays for getting closer, slower and more upright during the clip, so a lander already resting on the pad earns
nothing. Scoring each clip with the environment's own "how good is this state" term, for queries after 0.5M
timesteps:

| the preferred clip is the one that ... | VLM, VLM-only runs | oracle, same pairs | VLM, mixed runs | oracle, same pairs |
|---|---|---|---|---|
| ends in the better state | 65% | 60% | 82% | 39% |
| improves more during the clip | 44% | 74% | 35% | 87% |

This is why low agreement did not stop learning. A reward that pays for being low, slow, upright and on the pad
is not the environment's reward, but maximising it also produces landings. In the two VLM-only seeds that solved
the task, the learned reward ranks clips like the oracle only 39-40% of the time. Agreement with the oracle is
therefore a poor guide to whether VLM labels are useful here.

It also explains the typical failure: a policy that reaches the pad and stays there with engines running (VLM
seed 1, mixed seed 0) is in a "good state" in every clip. Why the mix did better than either rater alone
is not established; one reading is that the oracle's labels reward the approach and the VLM's reward arriving.

## Prompt tuning before the runs

Each variant was scored against the oracle on clip pairs from a true-reward PPO run (early, mid and late thirds of
training) and on pairs where one clip shows a lander that has flown out of view ("away"). Agreement is over labels
that pick a clip. Scores: [results/calibration/calibration.csv](../results/calibration/calibration.csv).

| model | variant | early | mid | late | away | prefers the out-of-view clip |
|---|---|---|---|---|---|---|
| 7B | `direct` (first attempt's setup) | 56% | 30% | 31% | 57% | 43% |
| 7B | `direct` + marker | 58% | 31% | 27% | 93% | 7% |
| 7B | **`progress` + marker (used)** | 55% | 34% | 27% | 100% | 0% |
| 7B | `reasoned` + marker | 64% | 23% | 6% | 100% | 0% |
| 7B | start/end frames only + marker | 47% | 31% | 22% | 100% | 0% |
| 32B | `progress` + marker | 60% | 33% | 27% | 100% | 0% |
| 32B | start/end frames only + marker | 55% | 40% | 42% | 100% | 0% |

The marker writes "LANDER OUT OF VIEW" on frames where the lander has left the picture. It is a presentation aid,
not information about the reward. No prompt and neither model size raised agreement on mid and late clips above
chance, for the reason in the previous section. The 7B rows use 334 pairs, the 32B rows 160. The start/end
variant was a one-off test and is not in `configs/prompts.yaml`.

## Caveats

- Five seeds, wide intervals: see the summary for what they support.
- The 50/50 mixing ratio is a placeholder. The paper never mixes raters, so it gives no ratio.
- The oracle condition is well below the paper's result and below the true-reward baseline.
- PPO replaces the paper's TRPO; the policy has two hidden layers of 64 (see the first attempt for why).
- One VLM, 8 frames per clip, answer-only prompt. The VLM conditions end with fewer labels than the oracle
  condition because unanswered queries still count against the budget.
- "Agreement with the oracle" measures agreement with LunarLander's reward, which rewards progress within a clip.

## How it was run

Slurm steps inside the 1-GPU allocation on bodhanai-node040 (job 26911), code revision `42961bd`:

| step | what | when (IST) | result |
|---|---|---|---|
| 26911.397 | true-reward and oracle runs (CPU) | 12 Nov 19:34-20:06 | exit 0, 10 runs |
| 26911.398-402 | calibration, 7B and 32B | 12 Nov 19:34-20:08 | exit 0 (one failed start, rerun) |
| 26911.405 | VLM and mixed runs | 12 Nov 23:54-00:35 | stopped on request; partial runs set aside |
| 26911.434 | VLM and mixed runs, from the start | 13 Nov 14:27-16:33 | exit 0, 10 runs |

The VLM runs were launched with `--set vlm.prompt=progress vlm.mark_out_of_view=true`; these are now the defaults
in `configs/default.yaml`. Three VLM runs shared the GPU at a time (about 22 GB each). All 20 runs have a
`summary.json`.

Everything is copied to `results/`: per-run files and videos under `results/runs/<condition>/seed_<n>/` (on disk
only, 178 MB, 1,974 videos), figures and `summary.csv` under `results/plots/`, calibration under
`results/calibration/`. Figures were regenerated with `python -m prefrl.plot`.

---

# First attempt (superseded): one-hidden-layer policy

> These are the results of the first attempt, kept as a record of why the setup changed. Its runs, videos and
> figures were deleted on 2026-11-12 when the experiment was restarted with a two-layer policy; the figures are in
> git history (revision `8744d48`, `results/plots/`).

Main run `20261112-prefrl-lunarlander-26ecf3`, 12 November 2026. Four conditions, five seeds each, 2M timesteps per
run, 1,400 preference queries per learned-reward run. Method and configuration are described in the
[README](../README.md).

## Summary

With Qwen2.5-VL-7B as the only rater, the agent did not learn to land: every seed ended with no successful landings,
and four of five learned to fly away from the pad. With the true-reward oracle as rater the method worked but stayed
well below PPO trained directly on the true reward. A 50/50 mix of oracle and VLM labels landed in between the oracle's
range, with the largest spread across seeds. No condition reached the "solved" score of 200 on average.

| condition | final return, mean [95% CI] | per seed | successful landings | labels in the database |
|---|---|---|---|---|
| PPO on true reward | 166 [134, 197] | 176, 202, 159, 153, 137 | 68% | none needed |
| Oracle preferences | 39 [-60, 138] | 135, 40, 94, -66, -8 | 29% | 1,400 of 1,400 |
| VLM preferences | -2,114 [-4,204, -23] | -2,051, -4,779, -1,625, -1,996, -118 | 0% | 659 of 1,400 on average |
| Oracle + VLM, 50/50 | 84 [-91, 259] | -117, 171, -3, 144, 225 | 40% | 1,070 of 1,400 on average |

Final return is the mean over 100 evaluation episodes of the deterministic final policy, scored with the true
environment reward. Intervals are Student-t 95% intervals of the mean over the five seeds. With five seeds they are
wide: the only differences they support are that the VLM-only condition is far worse than the other three, and that
the oracle condition is below the true-reward baseline. Oracle and mixed cannot be told apart.

## What happened in each condition

**True reward.** Returns rose steadily to 137-202. The policies land in 60-80% of episodes; the rest crash or hover
until the time limit.

**Oracle preferences.** All 1,400 queries were answered. The reward predictor ranked unseen clip pairs like the oracle
63-79% of the time. Policies improved to a best evaluation of 52-182 depending on the seed, and two seeds (3 and 4)
ended below zero. The paper reports oracle labels nearly matching RL; this implementation does not reach that.

**VLM preferences.** Only 47% of queries produced a label (the two clip orders had to give the same answer), and 39%
of those labels were ties. Where the VLM picked a clip it agreed with the oracle 50-59% of the time in four seeds and
72% in the fifth. The four seeds near chance learned to fly up and out of view: in their clips from the last quarter
of training the lander is off-screen for 92-98% of the steps, and final returns are -1,600 to -4,800. Seed 4 never
left the picture; it crashed within about 70 steps in every episode (return -118).

**Oracle + VLM.** Half the queries went to each rater. Three seeds reached 144-225 with 49-80% landings; two did not
learn to land. In four of the five seeds the VLM's labels agreed with the oracle only 29-41% of the time, which is
worse than chance; the next section explains why.

## Why the VLM labels did not help

The VLM is useful on the clips it was calibrated on and on the first, random-policy clips. It stops being useful as
soon as training starts, for two different reasons.

**1. It cannot rank two bad clips, and almost all clips are bad.** In the VLM-only runs 81% of all queries were
pairs in which both clips lose more than 80 reward: two crashes early on, two clips of the lander flying off later.
On those it gave no label for 52%, a tie for 23%, and agreed with the oracle on 53% of the rest.

| VLM-only runs, by phase | queries | got a label | of which ties | agrees with oracle (clear labels) |
|---|---|---|---|---|
| before RL (random policy) | 1,750 | 33% | 0 | 78% of 580 |
| first 0.5M timesteps | 2,715 | 39% | 290 | 52% of 772 |
| after 0.5M timesteps | 2,535 | 65% | 1,011 | 53% of 642 |

**2. It prefers "not crashing", including flying out of view.** When one clip lost more than 80 reward and the other
did not, the VLM agreed with the oracle 79% of the time. But among pairs that differed in how long the lander was off-screen,
it preferred the clip that was off-screen longer in 71% of cases (the oracle: 47%), despite the prompt calling this
as bad as a crash. A reward learned from such labels pays for leaving the picture. Once the lander is gone, both clips
show an empty sky and the VLM answers "tie" (the 1,011 ties above), so nothing corrects the policy.

**3. It rates the situation, while the environment reward rates the change.** This is what makes its labels worse
than chance once policies are competent. LunarLander's reward pays for getting closer, slowing down and straightening
up during the clip; a lander that is already sitting on the ground earns nothing. The VLM prefers the lander that
sits on the ground. In the mixed runs after 0.5M timesteps (984 clear VLM labels):

| the preferred clip is the one that ... | VLM | oracle |
|---|---|---|
| has its legs on the ground for more of the clip | 80% | 26% |
| is closer to the pad on average | 82% | 34% |
| is lower on average | 73% | 27% |
| is slower on average | 78% | 30% |
| gets closer to the pad during the clip | 36% | 74% |
| has the main engine on for more of the clip | 20% | 76% |

(Each row leaves out pairs in which the two clips are equal on that property.)

The prompt asked for exactly the oracle's view ("judge how the situation changes during the clip"), and the
answer-only prompt did not follow it. The VLM's preference is not unreasonable for a person watching; it is simply
not the quantity this experiment scores.

| mixed runs, VLM-answered queries, by phase | queries | got a label | agrees with oracle (clear labels) |
|---|---|---|---|
| before RL (random policy) | 884 | 33% | 76% of 288 |
| first 0.5M timesteps | 1,374 | 43% | 40% of 573 |
| after 0.5M timesteps | 1,255 | 78% | 29% of 984 |

The calibration set did not reveal any of this. Its clips came from landers of very different skill (a scripted
pilot with varying amounts of random actions), where 76% agreement was measured. The clips an agent produces during
training are either all crashes or all competent, and those are the two regimes where the VLM fails.

## Why the true-reward baseline is below 200: the policy network

Run `20261112-prefrl-ablation-aa3621`: PPO on the true reward only, 2M timesteps, three seeds per variant, everything
else as in the main run.

| episode ends | policy network | final return, mean [95% CI] | per seed | landings | first evaluation at 200 or more |
|---|---|---|---|---|---|
| hidden (paper) | 1 layer of 64 (main run, 5 seeds) | 166 [134, 197] | 176, 202, 159, 153, 137 | 68% | 1.7M in one seed |
| hidden (paper) | 2 layers of 64 | 272 [253, 290] | 273, 278, 264 | 96% | 0.46M to 0.66M |
| hidden (paper) | 2 layers of 256 | 271 [250, 292] | 278, 262, 274 | 95% | 0.41M to 0.46M |
| visible | 1 layer of 64 | 117 [94, 141] | 128, 114, 110 | 63% | never |
| visible | 2 layers of 64 | 277 [272, 282] | 279, 276, 276 | 99% | 0.70M to 0.85M |

The single hidden layer is the bottleneck. With two hidden layers PPO solves the task in the paper's setting, in about
a quarter of the timesteps. Hiding the episode ends, which the paper requires, costs little or nothing here. All
learned-reward conditions in the main run used the same one-layer policy, so their results are held back by it too;
how much is not known until they are rerun.

## Caveats

- Five seeds give wide intervals; see the summary for what they do and do not support.
- The true-reward baseline itself averages 166, below 200, because the policy has a single hidden layer of 64 units
  (see the ablation above). Every condition shares that limit.
- The oracle condition underperforms the paper's result. Label budget, label schedule, predictor regularisation and
  PPO settings were not tuned; the adaptive l2 coefficient sat at its ceiling during pretraining.
- The 50/50 mixing ratio is a placeholder; the paper has no such condition.
- One VLM, one prompt, 8 frames per clip. Nothing here says what a larger model, a reasoning prompt or a different
  clip presentation would do.
- "Agreement with the oracle" measures agreement with LunarLander's reward, which rewards progress within a clip.

## How it was run

One Slurm step (26911.390) inside the 1-GPU allocation on bodhanai-node040, 15:35-17:45 IST (2 h 10 min), exit 0,
all 20 runs complete. Code revision `26ecf31`; figures regenerated with revision `a14a796`. Up to three VLM runs
shared the GPU (22 GB each); the ten runs without a VLM ran on CPU alongside them.
