#!/bin/bash
# Replay sessions for SoccerNet test-split games (unseen in training): goals, red cards, penalties.
#   sbatch scripts/replay_games.sbatch
#SBATCH -J varflash-replay
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -o logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/VARFlash}"
source .venv/bin/activate
mkdir -p logs
SN=data/soccernet

GAMES=(
  "napoli-milan|italy_serie-a/2016-2017/2016-08-27 - 21-45 Napoli 4 - 2 AC Milan"
  "barcelona-psg|europe_uefa-champions-league/2016-2017/2017-03-08 - 22-45 Barcelona 6 - 1 Paris SG"
  "realmadrid-schalke|europe_uefa-champions-league/2014-2015/2015-03-10 - 22-45 Real Madrid 3 - 4 Schalke"
  "bayern-realmadrid|europe_uefa-champions-league/2016-2017/2017-04-12 - 21-45 Bayern Munich 1 - 2 Real Madrid"
  "dortmund-leverkusen|germany_bundesliga/2016-2017/2017-03-04 - 17-30 Dortmund 6 - 2 Bayer Leverkusen"
  "westbrom-chelsea|england_epl/2015-2016/2015-08-23 - 15-30 West Brom 2 - 3 Chelsea"
  "liverpool-westham|england_epl/2015-2016/2015-08-29 - 17-00 Liverpool 0 - 3 West Ham"
  "inter-fiorentina|italy_serie-a/2015-2016/2015-09-27 - 21-45 Inter 1 - 4 Fiorentina"
  "realmadrid-valencia|spain_laliga/2015-2016/2016-05-08 - 18-00 Real Madrid 3 - 2 Valencia"
)

for entry in "${GAMES[@]}"; do
  name=${entry%%|*}; dir=$SN/${entry#*|}
  [ -z "${FORCE:-}" ] && [ -f "outputs/replay/$name/session.json" ] && continue
  while [ "$(jobs -rp | wc -l)" -ge 4 ]; do sleep 5; done
  varflash -v replay build --name "$name" --game-dir "$dir" --device cuda > "logs/replay_$name.log" 2>&1 &
done

# Argentina-Switzerland: official highlights anchored to the Kalshi "to advance" market.
HIGHLIGHTS=${HIGHLIGHTS:-$HOME/orcd/scratch/varflash_data/matches/argsui_highlights.mp4}
if [ -f "$HIGHLIGHTS" ] && { [ -n "${FORCE:-}" ] || [ ! -f outputs/replay/argsui-highlights/session.json ]; }; then
  varflash -v replay build --name argsui-highlights --video "$HIGHLIGHTS" \
    --clips config/replay_clips/argsui_highlights.csv --event KXWCADVANCE-26JUL11ARGSUI --hold-min 2 \
    --note "$(cat config/replay_clips/argsui_highlights_note.txt)" --device cuda > logs/replay_argsui-highlights.log 2>&1 &
fi
wait
ls outputs/replay
