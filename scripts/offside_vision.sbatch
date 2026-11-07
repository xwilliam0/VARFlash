#!/bin/bash
# Visual offside verifier: train on raw-frame clips, then fuse with the spotter (fit on valid, AP on test).
#   SEED=0 sbatch scripts/offside_vision.sbatch
# Needs scripts/offside_clips.sbatch first (and, for the "_ctx" candidates, a run with
# --tag _ctx --spotter models/offside/all17_ctxtm_s0.pt).
#SBATCH -J varflash-offvis
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 06:00:00
#SBATCH -o logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/VARFlash}"
source .venv/bin/activate
SEED=${SEED:-0}
M=models/offside/offside_vision_s$SEED.pt
[ -f "$M" ] || varflash -v offside train --epochs 10 --seed "$SEED" --workers 14 --out "$M"
for tag in "" _ctx; do
  [ -f "outputs/offside/candidates${tag}_test.parquet" ] || continue
  varflash offside rescore --model "$M" --tag "$tag" --workers 14 > "logs/offside_rescore_s${SEED}${tag}.log" 2>&1
done
echo done
