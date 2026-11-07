#!/bin/bash
# Bootstrap CIs once the context-aware-loss sweep and the live card verifier jobs are done:
#   sbatch --dependency=afterany:<calf job>:<cardlive job> scripts/bootstrap_after.sbatch
#SBATCH -J varflash-boot
#SBATCH -p mit_normal
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/VARFlash}"
source .venv/bin/activate
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

for fam in all17_bce multitask_bce; do
  ls models/sweep/${fam}_calf_s[0-9].pt >/dev/null 2>&1 && python scripts/bootstrap_compare.py spotter $fam ${fam}_calf
done

tags=("")
for d in 0 2 4; do
  [ -f outputs/cards/scored_test_live_d$d.parquet ] && tags+=("_live_d$d")
done
python scripts/bootstrap_compare.py cards "${tags[@]}"
echo done
