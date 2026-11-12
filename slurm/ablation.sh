#!/bin/bash
# True-reward PPO only, 3 seeds per variant, all at once (CPU): what do the hidden episode ends and the policy size
# cost? Reference is the main experiment's pure_rl condition (hidden ends, one hidden layer of 64).
# Usage: ablation.sh <run dir> <python>     -> <run dir>/outputs/ablation/<variant>/seed_<n>/
set -f # the overrides contain [ ]
RUN=$1 PY=$2
variants=(
  "hidden_2x64 ppo.policy_kwargs.net_arch=[64,64]"
  "hidden_2x256 ppo.policy_kwargs.net_arch=[256,256]"
  "episodic_1x64 hide_episode_ends=false"
  "episodic_2x64 hide_episode_ends=false ppo.policy_kwargs.net_arch=[64,64]"
)
for variant in "${variants[@]}"; do
  read -r name overrides <<< "$variant"
  for seed in 0 1 2; do
    "$PY" -m prefrl.train --config "$RUN/config/default.yaml" --condition pure_rl --seed "$seed" \
      --out "$RUN/outputs/ablation/$name/seed_$seed" \
      --set threads=1 eval.video_episodes=0 eval.final_video_episodes=0 $overrides \
      > "$RUN/attempts/001/logs/ablation_${name}_seed$seed.log" 2>&1 &
  done
done
wait
ls "$RUN"/outputs/ablation/*/seed_*/summary.json | wc -l
