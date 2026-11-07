#!/bin/bash
# Raw-frame clips for the visual offside verifier (CPU only: video decoding + JPEG).
#   SPOTTER=models/offside/all17_ctx_s0.pt sbatch scripts/offside_clips.sbatch
# Clips are cached by id, so re-running with a new spotter only extracts its new candidates.
#SBATCH -J varflash-offclips
#SBATCH -p mit_normal
#SBATCH -c 32
#SBATCH --mem=96G
#SBATCH -t 08:00:00
#SBATCH -o logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/VARFlash}"
source .venv/bin/activate
export OMP_NUM_THREADS=4
SPOTTER=${SPOTTER:-models/sweep/all17_bce_s0.pt}
varflash -v offside build-clips --spotter "$SPOTTER" --workers 30 --device cpu
echo done
