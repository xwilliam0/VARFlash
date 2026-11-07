#!/bin/bash
# Offside: does the spotter improve when every frame sees the restart that follows it?
#   sbatch scripts/offside_sweep.sbatch
# Baselines are models/sweep/all17_bce*_s*.pt (60 s windows, edge-averaged inference).
#SBATCH -J varflash-offside
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -o logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/VARFlash}"
source .venv/bin/activate
mkdir -p logs models/offside outputs
SEEDS=${SEEDS:-"0 1 2"}
CTX="--window 240 --edge 60 --dilations 1 2 4 8 16 32"

run() {  # name, then train args
  local name=$1; shift
  [ -f "models/offside/$name.pt" ] && return
  varflash -v spotting train --workers 2 --out "models/offside/$name.pt" "$@" > "logs/train_$name.log" 2>&1
}
jobs_limit() { while [ "$(jobs -rp | wc -l)" -ge 4 ]; do sleep 5; done; }

for s in $SEEDS; do
  jobs_limit; run all17_ctx_s$s   --classes all --seed $s $CTX &
  jobs_limit; run all17_ctxtm_s$s --classes all --seed $s $CTX --time-mask 10 &
  jobs_limit; run all17_edge_s$s  --classes all --seed $s --window 120 --edge 30 &
done
# Inference-only: the existing baselines scored from window centres.
for s in $SEEDS; do
  jobs_limit; varflash spotting eval --model models/sweep/all17_bce_s$s.pt --edge 30 --split test \
    > logs/eval_all17_bce_s${s}_edge30.log 2>&1 &
done
wait

for m in models/offside/*.pt; do
  n=$(basename "$m" .pt)
  [ -f "outputs/spotting_eval_${n}_test/ap.csv" ] || varflash spotting eval --model "$m" --split test > "logs/eval_$n.log" 2>&1 &
  jobs_limit
done
wait
OMP_NUM_THREADS=8 python scripts/offside_context.py models/sweep/all17_bce_s0.pt > logs/offside_context_base_s0.log 2>&1 &
OMP_NUM_THREADS=8 python scripts/offside_errors.py models/offside/all17_ctx_s0.pt > logs/offside_errors_ctx_s0.log 2>&1 &
wait
echo done
