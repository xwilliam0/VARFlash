#!/bin/bash
# Spotter experiments on one GPU: 17-class baseline + card-model ablations, then evaluation.
#   sbatch scripts/spotting_sweep.sbatch            # new job
#   srun --jobid=<salloc job> --overlap bash scripts/spotting_sweep.sbatch   # inside an allocation
#SBATCH -J varflash-sweep
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 06:00:00
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

# Several small models share the GPU; run them 4 at a time.
jobs_limit() { while [ "$(jobs -rp | wc -l)" -ge 4 ]; do sleep 5; done; }

run all17_bce_s0 --classes all &
for s in $SEEDS; do
  jobs_limit; run cards_bce_s$s           --classes cards --seed $s &
  jobs_limit; run cards_focal_os20_s$s    --classes cards --seed $s --loss focal --oversample 20 &
  for d in 0 10 30; do
    jobs_limit; run cards_causal_d${d}_s$s --classes cards --seed $s --loss focal --oversample 20 --causal --delay $d &
  done
  # All SoccerNet classes as auxiliary tasks (reds merged), selected on card mAP.
  jobs_limit; run multitask_bce_s$s          --classes cards_multitask --seed $s &
  jobs_limit; run multitask_os20_s$s         --classes cards_multitask --seed $s --oversample 20 &
  for d in 10 30; do
    jobs_limit; run multitask_causal_d${d}_s$s --classes cards_multitask --seed $s --causal --delay $d &
  done
done
wait

for m in models/sweep/*.pt; do
  n=$(basename "$m" .pt)
  [ -f "outputs/spotting_eval_${n}_test/ap.csv" ] || varflash spotting eval --model "$m" --split test > "logs/eval_$n.log" 2>&1
done
varflash spotting official-eval --model models/sweep/all17_bce_s0.pt --split test > logs/official_all17_bce_s0.log 2>&1
echo done
