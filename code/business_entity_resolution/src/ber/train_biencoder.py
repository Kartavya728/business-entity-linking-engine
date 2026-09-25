"""Stage 1: fine-tune the blocking bi-encoder on encoder-fold (E) positive pairs.

Usage: python -m ber.train_biencoder [--max-pairs N] [--epochs 1]
"""
import argparse

import numpy as np
import polars as pl

from .biencoder import train_biencoder
from .config import WORK_DIR, split_dir
from .splits import enc_mask


def main(max_pairs: int, epochs: int, batch: int, hard: bool, init: str, out: str, lr: float):
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
    hn = None
    if hard:
        # one mined hard negative (top dense non-match of the same S1) per positive pair
        neg = pl.read_parquet(d / "hardneg_e.parquet")
        neg = neg.with_columns(pl.int_range(pl.len()).over("s1_idx").alias("r"),
                               pl.len().over("s1_idx").alias("n"))
        pick = gt.select("s1_idx").with_row_index("i").with_columns(
            pl.Series("u", np.random.default_rng(1).integers(0, 1 << 30, len(gt))))
        pick = pick.join(neg.select("s1_idx", "n").unique(), on="s1_idx", how="left")
        pick = pick.with_columns((pl.col("u") % pl.col("n").fill_null(1)).alias("r"))
        pick = pick.join(neg.select("s1_idx", "r", "tgt", "tgt_idx"), on=["s1_idx", "r"], how="left").sort("i")
        hn = [texts[k][j] if k is not None else "" for k, j in zip(pick["tgt"].to_list(), pick["tgt_idx"].to_list())]
        print(f"[biencoder] hard negatives attached ({sum(1 for h in hn if h):,} non-empty)", flush=True)
    print(f"[biencoder] {len(anchors):,} positive pairs from {em.sum():,} E-fold S1", flush=True)
    train_biencoder(anchors, pos, out, epochs=epochs, batch_size=batch, hard_negs=hn, init=init, lr=lr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pairs", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--hard", action="store_true", help="add mined hard negatives (work/train/hardneg_e.parquet)")
    ap.add_argument("--init", default="intfloat/multilingual-e5-small")
    ap.add_argument("--out", default=str(WORK_DIR / "biencoder"))
    ap.add_argument("--lr", type=float, default=5e-5)
    a = ap.parse_args()
    main(a.max_pairs, a.epochs, a.batch, a.hard, a.init, a.out, a.lr)
