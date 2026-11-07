"""CPU smoke test of the causal card verifier: tiny backbone, a few real clips, one epoch.

    python scripts/card_live_smoke.py
"""

import logging
from pathlib import Path

import pandas as pd

from varflash.cards.clips import CLIPS_DIR
from varflash.cards.train import CardTrainConfig, clip_live, clip_scores, live_report, load_card_model, train


def pick(df: pd.DataFrame, n: int) -> pd.DataFrame:
    red = df[df.events.str.contains(r", 1\]")].head(n // 3)
    yellow = df[df.events.str.contains(r", 0\]")].head(n // 3)
    neg = df[df.events == "[]"].head(n // 3)
    return pd.concat([red, yellow, neg]).drop_duplicates("clip_id")


def main():
    logging.basicConfig(level=logging.INFO)
    tr, va, te = (pick(pd.read_parquet(CLIPS_DIR / f"index_{s}.parquet"), n)
                  for s, n in (("train", 24), ("valid", 18), ("test", 18)))
    cfg = CardTrainConfig(backbone="resnet18", frames=24, batch_size=2, epochs=1, samples_per_epoch=4,
                          causal=True, delay_s=2.0, num_workers=2)
    path = train(cfg, tr, va, out=Path("/tmp/card_live_smoke.pt"))
    model, ck = load_card_model(path, "cpu")
    print("causal", model.causal, "delay", ck["config"]["delay_s"])
    lv, lt = clip_live(model, va, 24, num_workers=2), clip_live(model, te, 24, num_workers=2)
    print("live shape", lv.shape)
    print(live_report(lv, va, lt, te).round(3).head(6).to_string(index=False))
    print("rescore scores", clip_scores(model, te.assign(events="[]"), 24, num_workers=2, delay=8).shape)
    print("SMOKE OK")


if __name__ == "__main__":
    main()
