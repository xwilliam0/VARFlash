#!/bin/bash
# Card candidates scored by every visual seed, with and without mirror test-time augmentation,
# so scripts/card_fusion.py can compare seed ensembles / TTA. The untagged scores are seed 1.
# Also trains seed 2 for a three-model ensemble.
#   sbatch scripts/card_rescore_variants.sbatch
#SBATCH -J varflash-cardvar
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH --mem=96G
#SBATCH -t 05:00:00
#SBATCH -o logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/VARFlash}"
source .venv/bin/activate

score() {  # model, tag, extra args
  [ -f outputs/cards/scored_test$2.parquet ] || varflash cards rescore --model $1 --tag $2 --workers 11 "${@:3}"
}
score models/card_vision.pt _s0
score models/card_vision_s1.pt _s1_tta --tta
score models/card_vision.pt _s0_tta --tta
[ -f models/card_vision_s2.pt ] || varflash -v cards train --seed 2 --workers 11 --out models/card_vision_s2.pt
score models/card_vision_s2.pt _s2
score models/card_vision_s2.pt _s2_tta --tta
echo done
