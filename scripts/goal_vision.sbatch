#!/bin/bash
# Visual goal detector: extract 12.5 fps shot clips, train the causal clip model, evaluate speed vs accuracy.
#   sbatch scripts/goal_vision.sbatch
#SBATCH -J varflash-goalvis
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

varflash -v goals build-clips --workers 15
[ -f models/goal_vision.pt ] || varflash -v goals train --workers 14 --out models/goal_vision.pt
varflash goals eval --model models/goal_vision.pt --workers 14
echo done
