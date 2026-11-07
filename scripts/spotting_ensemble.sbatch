#!/bin/bash
# Seed ensembles of every models/sweep/<name>_s<k>.pt family with >= 2 seeds, evaluated on test.
#   sbatch scripts/spotting_ensemble.sbatch
#SBATCH -J varflash-ens
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/VARFlash}"
source .venv/bin/activate
mkdir -p logs outputs

jobs_limit() { while [ "$(jobs -rp | wc -l)" -ge 4 ]; do sleep 5; done; }

for s0 in models/sweep/*_s0.pt; do
  stem=$(basename "$s0" _s0.pt)
  mapfile -t members < <(ls models/sweep/${stem}_s[0-9].pt)
  n=${#members[@]}
  [ "$n" -ge 2 ] || continue
  name=${stem}_ens$n
  [ -f "outputs/spotting_eval_${name}_test/ap.csv" ] && continue
  jobs_limit
  varflash -v spotting eval --model "${members[@]}" --name "$name" --split test > "logs/eval_$name.log" 2>&1 &
  if [[ $stem == all17_* ]]; then
    jobs_limit
    varflash spotting official-eval --model "${members[@]}" --name "$name" --split test > "logs/official_$name.log" 2>&1 &
  fi
done
wait
echo done
