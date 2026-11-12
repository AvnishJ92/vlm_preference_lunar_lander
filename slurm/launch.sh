#!/bin/bash
# Launcher for the 1-GPU allocation on bodhanai-node040 (Slurm job "avnish"). Nothing runs until you call a stage.
#
#   slurm/launch.sh prepare            snapshot the committed code + configs into a new external run directory
#   slurm/launch.sh model     <run>    download the pinned Qwen2.5-VL weights into the HF cache (login node, no GPU)
#   slurm/launch.sh calibrate <run> [name --variants ...]   GPU: score prompt variants against the oracle
#                                                           -> outputs/calibration/
#   slurm/launch.sh smoke     <run>    GPU: short end-to-end run of the VLM condition    -> smoke/
#   slurm/launch.sh train     <run> <name> [run_all options]   runs, e.g. `baselines --conditions pure_rl oracle`
#                                      or `all`                -> outputs/<condition>/seed_<n>/
#   slurm/launch.sh plot      <run>    figures + summary table                           -> outputs/plots/
#
# calibrate / smoke / train are `srun --overlap` steps inside the allocation, started in the background. Follow one
# with `tail -f <run>/attempts/001/logs/<stage>.out` and `squeue -s -j $ALLOC`; stop one with `scancel <job>.<step>`.
set -euo pipefail
ALLOC="${ALLOC:-26911}"
ARTIFACTS="${ARTIFACTS:-/projects/data/ttsteam/avnish/artifacts}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
stage="${1:?usage: launch.sh prepare | launch.sh <model|calibrate|smoke|train|plot> <run dir> [...]}"

if [ "$stage" = prepare ]; then
  source "$REPO/scripts/env.sh"
  id="${RUN_ID:-$(date -u +%Y%m%d)-prefrl-lunarlander-$(git -C "$REPO" rev-parse --short=6 HEAD)}"
  run=$(agent-run prepare --repo "$REPO" --root "$ARTIFACTS" --project vlm_preference_lunar_lander --id "$id" \
    --purpose "Deep RL from preferences on LunarLander-v3: PPO on the true reward vs a reward learned from oracle, Qwen2.5-VL-7B and mixed preferences; 5 seeds each" \
    --config "$REPO/configs/default.yaml" --config "$REPO/configs/prompts.yaml" --environment "$REPO/uv.lock" \
    ${PARENT_RUN:+--parent-run "$PARENT_RUN"})
  mkdir "$run/code" "$run/outputs" "$run/smoke"
  tar -xf "$run/provenance/source.tar" -C "$run/code"
  chmod -R a-w "$run/code" # the run executes this snapshot, not the development tree
  echo "$UV_PROJECT_ENVIRONMENT" > "$run/provenance/venv_path.txt"
  uv pip freeze --python "$PY" > "$run/provenance/env_freeze.txt"
  agent-run check-paths --code "$REPO" --run-dir "$run" \
    --output "$run/outputs" --output "$run/smoke" --output "$run/attempts" > /dev/null
  echo "prepared: $run   (agent-run added it to README.md; commit that line)"
  exit
fi

RUN="$(cd "${2:?run directory (printed by: launch.sh prepare)}" && pwd)"
PY="$(cat "$RUN/provenance/venv_path.txt")/bin/python"
CFG="$RUN/config/default.yaml"
LOGS="$RUN/attempts/001/logs"
export PYTHONPATH="$RUN/code/src" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy

step() { # step <name> <cpus> <command...>: one background step inside the allocation, logged under the attempt
  local name=$1 cpus=$2
  shift 2
  [ "$(squeue -h -j "$ALLOC" -o '%T %N')" = "RUNNING bodhanai-node040" ] ||
    { echo "job $ALLOC is not the running node040 allocation; set ALLOC=<job id>" >&2; exit 1; }
  [ -d "$RUN/attempts/001" ] || agent-run attempt --run-dir "$RUN" --job-id "$ALLOC" \
    --command-json "[\"slurm/launch.sh\",\"$name\",\"$RUN\"]" > /dev/null
  # HF_HUB_OFFLINE: the weights come from the cache filled by the `model` stage, or the step fails at once.
  nohup srun --jobid="$ALLOC" --overlap --ntasks=1 --cpus-per-task="$cpus" --job-name="prefrl_$name" \
    env HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$@" > "$LOGS/$name.out" 2>&1 &
  echo "started step prefrl_$name in job $ALLOC; log: $LOGS/$name.out"
}

case "$stage" in
  model) # network and disk only, so it runs here; goes to the account's HF cache ($HF_HOME), about 17 GB
    read -r model revision < <("$PY" -c 'import sys, yaml; v = yaml.safe_load(open(sys.argv[1]))["vlm"]
print(v["model"], v["revision"])' "$CFG")
    "$(dirname "$PY")/hf" download "$model" --revision "$revision"
    ;;
  calibrate) # optional: a step name and `--variants a b` to split the variants over two concurrent steps
    step "calibrate${3:+_$3}" 4 "$PY" -m prefrl.calibrate --config "$CFG" --out "$RUN/outputs/calibration" "${@:4}"
    ;;
  smoke) # exercises model load, clip rendering, VLM labelling, predictor training, PPO, evaluation and videos
    step smoke 4 "$PY" -m prefrl.train --config "$CFG" --condition vlm --seed 0 --out "$RUN/smoke/vlm_$(date +%H%M%S)" \
      --set total_timesteps=32768 ppo.n_steps=256 pref.round_every=8192 pref.n_queries=48 pref.pretrain_steps=4000 \
      pref.pretrain_epochs=5 eval.every=16384 eval.final_episodes=5 eval.final_video_episodes=1
    ;;
  train) # runs of prefrl.run_all; a second call must name other conditions than one still running
    step "train_${3:?step name, e.g. baselines}" 16 "$PY" -m prefrl.run_all --config "$CFG" --out "$RUN/outputs" \
      --logs "$LOGS" "${@:4}"
    ;;
  plot)
    "$PY" -m prefrl.plot "$RUN/outputs"
    ;;
  *)
    echo "unknown stage: $stage" >&2
    exit 2
    ;;
esac
