"""Replace the deployed single-seed spotters with the best seed ensemble of their family.

    python scripts/deploy_ensembles.py            # pick by validation mAP, then write
    python scripts/deploy_ensembles.py --dry-run  # only show the picks

For each slot, every ``<family>[_<variant>]_ens<N>`` evaluated by ``spotting_ensemble.sbatch``
is a candidate; the one with the best *validation* mAP (from its eval log) wins, so the test
numbers stay unbiased. The previous checkpoint is moved to ``models/single/`` (once) and the
ensemble is written under the old name, so every caller of ``load_spotter`` picks it up.
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from compare_spotting import parse  # noqa: E402

SLOTS = {"spotter": "all17_bce", "cards": "multitask_bce", "live_d10": "multitask_causal_d10",
         "cards_live": "multitask_causal_d30"}


def candidates(family: str) -> list[tuple[float, str, list[Path]]]:
    out = []
    for log in (ROOT / "logs").glob(f"eval_{family}*_ens*.log"):
        name = log.stem[len("eval_"):]
        stem, _ = re.fullmatch(r"(.+)_ens(\d+)", name).groups()
        if parse(f"{stem}_s0")[0] != family:
            continue
        m = re.search(r"valid mAP ([0-9.]+)", log.read_text())
        members = sorted((ROOT / "models/sweep").glob(f"{stem}_s[0-9].pt"))
        if m and members:
            out.append((float(m.group(1)), stem, members))
    return sorted(out, reverse=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    picks = {}
    for slot, family in SLOTS.items():
        c = candidates(family)
        if not c:
            print(f"{slot}: no ensemble evals for {family}, skipping")
            continue
        print(f"{slot} ({family}):")
        for v, stem, members in c:
            print(f"  {'*' if (v, stem) == c[0][:2] else ' '} {stem:<36} valid mAP {v:.4f}  ({len(members)} seeds)")
        picks[slot] = c[0][2]
    if a.dry_run:
        return
    from varflash.spotting.train import save_ensemble

    (ROOT / "models/single").mkdir(exist_ok=True)
    for slot, members in picks.items():
        cur, keep = ROOT / f"models/{slot}.pt", ROOT / f"models/single/{slot}.pt"
        if cur.exists() and not keep.exists():
            shutil.move(cur, keep)
        save_ensemble(members, cur, a.device)


if __name__ == "__main__":
    main()
