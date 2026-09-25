"""Stage 1: fine-tune the blocking bi-encoder on encoder-fold (E) positive pairs.

Usage: python -m ber.train_biencoder [--max-pairs N] [--epochs 1]
"""
import argparse

import numpy as np
import polars as pl

from .biencoder import train_biencoder
from .config import WORK_DIR, split_dir
from .splits import enc_mask


def main(max_pairs: int, epochs: int, batch: int):
    d = split_dir("train")
    s1 = pl.read_parquet(d / "source1.parquet", columns=["idx", "text"])
    gt = pl.read_parquet(d / "gt.parquet")
    em = enc_mask(len(s1))
    gt = gt.filter(pl.col("s1_idx").is_in(np.flatnonzero(em)))
    texts = {k: pl.read_parquet(d / f"source{k}.parquet", columns=["text"])["text"].to_list() for k in (2, 3)}
    s1t = s1["text"].to_list()
    gt = gt.sample(fraction=1.0, shuffle=True, seed=0)
    if max_pairs and len(gt) > max_pairs:
        gt = gt.head(max_pairs)
    anchors = [s1t[i] for i in gt["s1_idx"].to_list()]
    pos = [texts[k][j] for k, j in zip(gt["tgt"].to_list(), gt["tgt_idx"].to_list())]
    print(f"[biencoder] {len(anchors):,} positive pairs from {em.sum():,} E-fold S1", flush=True)
    train_biencoder(anchors, pos, str(WORK_DIR / "biencoder"), epochs=epochs, batch_size=batch)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pairs", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=512)
    a = ap.parse_args()
    main(a.max_pairs, a.epochs, a.batch)
