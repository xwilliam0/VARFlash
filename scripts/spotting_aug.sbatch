#!/bin/bash
# Spotter data-augmentation ablation (mixup, time masking) vs the spotting_sweep.sbatch baselines.
#   sbatch scripts/spotting_aug.sbatch
# Baselines reuse models/sweep/{all17_bce,cards_bce,multitask_bce,cards_causal_d10}_s*.pt;
# only the missing all17 seeds are retrained.
#SBATCH -J varflash-aug
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/VARFlash}"
source .venv/bin/activate
mkdir -p logs models/sweep outputs
SEEDS=${SEEDS:-"0 1 2"}
MIX=${MIX:-0.2}
TM=${TM:-10}  # frames (5 s at 2 fps)

run() {  # name, then train args
  local name=$1; shift
  [ -f "models/sweep/$name.pt" ] && return
  varflash -v spotting train --workers 2 --out "models/sweep/$name.pt" "$@" > "logs/train_$name.log" 2>&1
}

jobs_limit() { while [ "$(jobs -rp | wc -l)" -ge 4 ]; do sleep 5; done; }

declare -A AUG=([mix]="--mixup $MIX" [tm]="--time-mask $TM" [mixtm]="--mixup $MIX --time-mask $TM")

for s in $SEEDS; do
  jobs_limit; run all17_bce_s$s --classes all --seed $s &
  for a in mix tm mixtm; do
    jobs_limit; run all17_bce_${a}_s$s --classes all --seed $s ${AUG[$a]} &
    jobs_limit; run cards_bce_${a}_s$s --classes cards --seed $s ${AUG[$a]} &
  done
  jobs_limit; run multitask_bce_mixtm_s$s --classes cards_multitask --seed $s ${AUG[mixtm]} &
  for a in tm mixtm; do
    jobs_limit; run cards_causal_d10_${a}_s$s --classes cards --seed $s --loss focal --oversample 20 --causal --delay 10 \
      ${AUG[$a]} &
  done
done
wait

for m in models/sweep/*.pt; do
  n=$(basename "$m" .pt)
  [ -f "outputs/spotting_eval_${n}_test/ap.csv" ] || varflash spotting eval --model "$m" --split test > "logs/eval_$n.log" 2>&1
done
for a in mix tm mixtm; do
  varflash spotting official-eval --model models/sweep/all17_bce_${a}_s0.pt --split test > logs/official_all17_bce_${a}_s0.log 2>&1
done
echo done
