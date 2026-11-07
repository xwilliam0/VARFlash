#!/bin/bash
# Candidates from the long-context spotter (valid + test), their clips, then fusion with each visual model.
#   sbatch scripts/offside_ctx_rescore.sbatch
#SBATCH -J varflash-offctx
#SBATCH -p mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 04:00:00
#SBATCH --requeue
#SBATCH -o logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/VARFlash}"
source .venv/bin/activate
SPOTTER=${SPOTTER:-models/offside/all17_ctxtm_s0.pt}
[ -f outputs/offside/candidates_ctx_test.parquet ] && [ -f data/soccernet/offside_clips/index_ctx_test.parquet ] ||
  OMP_NUM_THREADS=2 varflash -v offside build-clips --spotter "$SPOTTER" --splits valid test --tag _ctx \
    --confuser-neg 0 --random-neg 0 --workers 15
for m in models/offside/offside_vision_s*.pt; do
  varflash offside rescore --model "$m" --tag _ctx --workers 14 \
    > "logs/offside_rescore_$(basename "$m" .pt | sed 's/offside_vision_//')_ctx.log" 2>&1
done
echo done
