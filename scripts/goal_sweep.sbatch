#!/bin/bash
# Live goal alerts: causal multitask spotters with different look-ahead budgets, selected on Goal AP,
# then speed-vs-accuracy on test (scripts/goal_latency.py).
#   sbatch scripts/goal_sweep.sbatch
#SBATCH -J varflash-goal
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/VARFlash}"
source .venv/bin/activate
mkdir -p logs models/sweep
SEEDS=${SEEDS:-"0 1"}
DELAYS=${DELAYS:-"0 1 2 3 5 10"}

run() {  # name, then train args
  local name=$1; shift
  [ -f "models/sweep/$name.pt" ] && return
  varflash -v spotting train --workers 2 --out "models/sweep/$name.pt" "$@" > "logs/train_$name.log" 2>&1
}
jobs_limit() { while [ "$(jobs -rp | wc -l)" -ge 4 ]; do sleep 5; done; }

for s in $SEEDS; do
  for d in $DELAYS; do
    jobs_limit; run goal_causal_d${d}_s$s --classes goal_multitask --seed $s --causal --delay $d &
  done
done
wait

python scripts/goal_latency.py models/sweep/goal_causal_d*_s*.pt models/live_d10.pt
echo done
