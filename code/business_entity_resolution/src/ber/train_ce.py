"""Train the cross-encoder on encoder-fold (E) candidates, and score pairs with it.

Usage:
  python -m ber.train_ce train [--n-s1 150000]      # fit on E-fold blocking candidates
  python -m ber.train_ce score --split train|test   # score pairs with p1 >= CE_MIN_P1
"""
import argparse

import numpy as np
import polars as pl

from .config import WORK_DIR, split_dir
from .crossencoder import CrossEncoder, train_crossencoder
from .splits import ENC_FOLDS, s1_folds

CE_PATH = WORK_DIR / "crossencoder"
CE_MIN_P1 = 0.003
HARD_RANK = 12  # negatives: candidates within this fused rank


def _texts(split):
    d = split_dir(split)
    return {k: pl.read_parquet(d / f"source{k}.parquet", columns=["text"])["text"].to_list() for k in (1, 2, 3)}


def train(n_s1: int):
    d = split_dir("train")
    cand = pl.read_parquet(d / "candidates.parquet", columns=["s1_idx", "tgt", "tgt_idx", "rrf_r"])
    gt = pl.read_parquet(d / "gt.parquet").with_columns(pl.lit(1, pl.Int8).alias("y"))
    folds = s1_folds(pl.read_parquet(d / "source1.parquet", columns=["idx"]).height)
    e_s1 = np.flatnonzero(np.isin(folds, ENC_FOLDS))
    e_s1 = np.random.default_rng(0).choice(e_s1, size=min(n_s1, len(e_s1)), replace=False)
    c = cand.filter(pl.col("s1_idx").is_in(e_s1))
    c = c.join(gt, on=["s1_idx", "tgt", "tgt_idx"], how="left").with_columns(pl.col("y").fill_null(0))
    c = c.filter((pl.col("y") == 1) | (pl.col("rrf_r") <= HARD_RANK))
    print(f"[ce] {len(c):,} training pairs, pos rate {c['y'].mean():.3f}", flush=True)
    T = _texts("train")
    a = [T[1][i] for i in c["s1_idx"].to_list()]
    b = [T[k][j] for k, j in zip(c["tgt"].to_list(), c["tgt_idx"].to_list())]
    train_crossencoder(a, b, c["y"].to_numpy(), str(CE_PATH))


def score(split: str):
    d = split_dir(split)
    p = pl.read_parquet(d / "p1.parquet")  # s1_idx, tgt, tgt_idx, p1
    p = p.filter(pl.col("p1") >= CE_MIN_P1)
    print(f"[ce] scoring {len(p):,} {split} pairs", flush=True)
    T = _texts(split)
    a = [T[1][i] for i in p["s1_idx"].to_list()]
    b = [T[k][j] for k, j in zip(p["tgt"].to_list(), p["tgt_idx"].to_list())]
    ce = CrossEncoder.load(str(CE_PATH)).predict(a, b)
    p.select("s1_idx", "tgt", "tgt_idx").with_columns(pl.Series("ce", ce)).write_parquet(d / "ce.parquet")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "score"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--n-s1", type=int, default=150000)
    a = ap.parse_args()
    train(a.n_s1) if a.cmd == "train" else score(a.split)
