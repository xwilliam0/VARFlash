#!/bin/bash
# Epoch selection for the card spotters: Yellow card only (yc) and/or the selection score smoothed
# over 3 epochs (sm3), versus the default (mean of Yellow card + Sending off, single epoch).
# Recipes are the deployed ones (models/cards.pt, live_d10.pt, cards_live.pt). Then test eval + ensembles;
# compare with: python scripts/bootstrap_compare.py spotter <baseline family> <family>_yc ...
#   sbatch scripts/spotting_select.sbatch
#SBATCH -J varflash-select
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

AUG="--time-mask 10 --ema 0.999 --label-sigma 2"
declare -A FAMILY=(
  [multitask_bce_tm_ema_g2]="--classes cards_multitask $AUG"
  [multitask_causal_d10_tm_ema_g2]="--classes cards_multitask --causal --delay 10 $AUG"
  [multitask_causal_d30]="--classes cards_multitask --causal --delay 30"
)
declare -A SELECT=(
  [yc]="--select-classes Yellow\ card"
  [sm3]="--select-smooth 3"
  [yc_sm3]="--select-classes Yellow\ card --select-smooth 3"
)

for s in $SEEDS; do
  for f in "${!FAMILY[@]}"; do
    for v in "${!SELECT[@]}"; do
      eval "args=(${FAMILY[$f]} ${SELECT[$v]})"
      jobs_limit; run ${f}_${v}_s$s "${args[@]}" --seed $s &
    done
  done
done
wait

for m in models/sweep/*.pt; do
  n=$(basename "$m" .pt)
  [ -f "outputs/spotting_eval_${n}_test/ap.csv" ] && continue
  jobs_limit
  varflash spotting eval --model "$m" --split test > "logs/eval_$n.log" 2>&1 &
done
wait
bash scripts/spotting_ensemble.sbatch
for f in "${!FAMILY[@]}"; do
  python scripts/bootstrap_compare.py spotter $f ${f}_yc ${f}_sm3 ${f}_yc_sm3 > "logs/bootstrap_select_$f.log" 2>&1
done
echo done
