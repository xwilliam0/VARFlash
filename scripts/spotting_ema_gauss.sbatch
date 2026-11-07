#!/bin/bash
# Weight EMA and Gaussian targets on top of time masking, for the four deployed spotter families
# (models/spotter.pt, cards.pt, live_d10.pt, cards_live.pt), then test eval and seed ensembles.
#   sbatch scripts/spotting_ema_gauss.sbatch
#SBATCH -J varflash-emag
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 04:00:00
#SBATCH -o logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/VARFlash}"
source .venv/bin/activate
mkdir -p logs models/sweep outputs
SEEDS=${SEEDS:-"0 1 2"}
TM=${TM:-10}
EMA=${EMA:-0.999}
SIGMA=${SIGMA:-2}

run() {  # name, then train args
  local name=$1; shift
  [ -f "models/sweep/$name.pt" ] && return
  varflash -v spotting train --workers 2 --out "models/sweep/$name.pt" "$@" > "logs/train_$name.log" 2>&1
}
jobs_limit() { while [ "$(jobs -rp | wc -l)" -ge 6 ]; do sleep 5; done; }

declare -A FAMILY=(
  [all17_bce]="--classes all"
  [multitask_bce]="--classes cards_multitask"
  [multitask_causal_d10]="--classes cards_multitask --causal --delay 10"
  [multitask_causal_d30]="--classes cards_multitask --causal --delay 30"
)
declare -A VARIANT=(
  [tm]="--time-mask $TM"
  [tm_ema]="--time-mask $TM --ema $EMA"
  [tm_g2]="--time-mask $TM --label-sigma $SIGMA"
  [tm_ema_g2]="--time-mask $TM --ema $EMA --label-sigma $SIGMA"
)

for s in $SEEDS; do
  for f in "${!FAMILY[@]}"; do
    for v in "${!VARIANT[@]}"; do
      jobs_limit; run ${f}_${v}_s$s ${FAMILY[$f]} ${VARIANT[$v]} --seed $s &
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
echo done
