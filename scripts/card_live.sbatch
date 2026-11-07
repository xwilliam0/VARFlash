#!/bin/bash
# Causal (live) visual card verifier: train with 0 / 2 / 4 s of look-ahead after the card,
# then score each live (AP and recall by seconds after the card) and through the spotter fusion.
#   sbatch scripts/card_live.sbatch
#SBATCH -J varflash-cardlive
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 06:00:00
#SBATCH -o logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/VARFlash}"
source .venv/bin/activate
mkdir -p logs models
nvidia-smi --query-gpu=name,memory.total --format=csv

for d in 0 2 4; do
  m=models/card_vision_live_d$d.pt
  [ -f $m ] || varflash -v cards train --causal --delay $d --workers 14 --out $m
  varflash cards live-eval --model $m --workers 14
  varflash cards rescore --model $m --tag _live_d$d --workers 14
done
echo done
