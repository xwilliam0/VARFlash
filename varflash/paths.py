from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
KALSHI_DATA = DATA / "kalshi"
SOCCERNET_DATA = DATA / "soccernet"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"
KICKOFFS_CSV = ROOT / "config" / "kickoffs.csv"
