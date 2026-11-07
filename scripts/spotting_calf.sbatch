#!/bin/bash
# Context-aware loss (Cioppa et al., CVPR 2020) vs the plain BCE baselines already in models/sweep
# ({all17,multitask}_bce_s{0,1,2}.pt): same model and data, only --loss calf. Then test eval + comparison.
#   sbatch scripts/spotting_calf.sbatch
#SBATCH -J varflash-calf
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 03:00:00
#SBATCH -o logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/VARFlash}"
source .venv/bin/activate
mkdir -p logs models/sweep outputs
SEEDS=${SEEDS:-"0 1 2"}

run() {  # name, then train args
  local name=$1; shift
  [ -f "models/sweep/$name.pt" ] && return
  varflash -v spotting train --workers 2 --out "models/sweep/$name.pt" "$@" > "logs/train_$name.log" 2>&1
}
jobs_limit() { while [ "$(jobs -rp | wc -l)" -ge 6 ]; do sleep 5; done; }

for s in $SEEDS; do
  jobs_limit; run all17_bce_calf_s$s --classes all --loss calf --seed $s &
  jobs_limit; run multitask_bce_calf_s$s --classes cards_multitask --loss calf --seed $s &
done
wait

for m in models/sweep/{all17,multitask}_bce_calf_s*.pt; do
  n=$(basename "$m" .pt)
  [ -f "outputs/spotting_eval_${n}_test/ap.csv" ] && continue
  jobs_limit
  varflash spotting eval --model "$m" --split test > "logs/eval_$n.log" 2>&1 &
done
wait
python scripts/compare_spotting.py all17_bce multitask_bce
echo done
