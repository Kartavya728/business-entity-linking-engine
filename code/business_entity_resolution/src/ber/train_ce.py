"""Train the cross-encoder on encoder-fold (E) candidates, and score pairs with it.

Usage:
  python -m ber.train_ce train [--n-s1 150000] [--model small|base]   # fit on E-fold candidates
  python -m ber.train_ce score --split train|test [--model small|base] # score pairs with p1 >= CE_MIN_P1
"""
import argparse

import numpy as np
import polars as pl

from .config import WORK_DIR, split_dir
from .crossencoder import CrossEncoder, train_crossencoder
from .splits import ENC_FOLDS, s1_folds

CE_PATH = WORK_DIR / "crossencoder"
# name -> (base checkpoint, output dir, score column, score file)
CE_MODELS = {
    "small": ("intfloat/multilingual-e5-small", WORK_DIR / "crossencoder", "ce", "ce.parquet"),
    "base": ("intfloat/multilingual-e5-base", WORK_DIR / "crossencoder_base", "ceb", "ce_b.parquet"),
    "base2": ("intfloat/multilingual-e5-base", WORK_DIR / "crossencoder_base2", "ceb2", "ce_b2.parquet"),
    # canonical text (filler-free name + legal form + normalised address), see fillers.py
    "canon": ("intfloat/multilingual-e5-small", WORK_DIR / "crossencoder_canon", "cec", "ce_c.parquet"),
    # field-structured input (raw name | address || core name ; legal | normalised address) on the
    # large multilingual backbone (XLM-R large, stronger on unseen languages)
    "large": ("intfloat/multilingual-e5-large", WORK_DIR / "crossencoder_large", "cel", "ce_l.parquet"),
    # different architecture (DeBERTa-v3, MIT) for ensemble diversity, same field-structured input
    "mdeberta": ("microsoft/mdeberta-v3-base", WORK_DIR / "crossencoder_mdeberta", "cemd", "ce_md.parquet"),
}
TEXT_COL = {"canon": "ctext", "large": "ftext", "mdeberta": "ftext"}
TRAIN_CFG = {  # model -> (batch size, lr, sparse-rank negatives, side-swap augmentation, max_len)
    "small": (256, 3e-5, 0, False, 128), "canon": (256, 3e-5, 0, False, 128),
    "base": (128, 2e-5, 0, False, 128), "base2": (128, 2e-5, 0, False, 128),
    "large": (64, 1e-5, 6, True, 160),
    "mdeberta": (128, 3e-5, 6, True, 160),
}
CE_MIN_P1 = 0.003
HARD_RANK = 12  # negatives: candidates within this fused rank


def _texts(split, col="text"):
    d = split_dir(split)
    if col == "ftext":  # field-structured text: raw view || canonical view
        return {k: [f"{a} || {b}" for a, b in pl.read_parquet(d / f"source{k}.parquet", columns=["text", "ctext"]).iter_rows()]
                for k in (1, 2, 3)}
    return {k: pl.read_parquet(d / f"source{k}.parquet", columns=[col])[col].to_list() for k in (1, 2, 3)}


def train(n_s1: int, model: str = "small", seed: int = 0):
    d = split_dir("train")
    cand = pl.read_parquet(d / "candidates.parquet", columns=["s1_idx", "tgt", "tgt_idx", "dense_r2", "sparse_r2"])
    bs, lr, sparse_neg, swap, max_len = TRAIN_CFG[model]
    gt = pl.read_parquet(d / "gt.parquet").with_columns(pl.lit(1, pl.Int8).alias("y"))
    folds = s1_folds(pl.read_parquet(d / "source1.parquet", columns=["idx"]).height)
    e_s1 = np.flatnonzero(np.isin(folds, ENC_FOLDS))
    e_s1 = np.random.default_rng(seed).choice(e_s1, size=min(n_s1, len(e_s1)), replace=False)
    c = cand.filter(pl.col("s1_idx").is_in(e_s1))
    c = c.join(gt, on=["s1_idx", "tgt", "tgt_idx"], how="left").with_columns(pl.col("y").fill_null(0))
    c = c.filter((pl.col("y") == 1) | (pl.col("dense_r2") < HARD_RANK) | (pl.col("sparse_r2") < sparse_neg))
    print(f"[ce] {len(c):,} training pairs, pos rate {c['y'].mean():.3f}", flush=True)
    T = _texts("train", TEXT_COL.get(model, "text"))
    a = [T[1][i] for i in c["s1_idx"].to_list()]
    b = [T[k][j] for k, j in zip(c["tgt"].to_list(), c["tgt_idx"].to_list())]
    base, out, _, _ = CE_MODELS[model]
    train_crossencoder(a, b, c["y"].to_numpy(), str(out), batch_size=bs, lr=lr, base=base, swap=swap,
                       max_len=max_len, seed=42 + seed)


def score(split: str, model: str = "small"):
    d = split_dir(split)
    p = pl.read_parquet(d / "p1.parquet")  # s1_idx, tgt, tgt_idx, p1
    p = p.filter(pl.col("p1") >= CE_MIN_P1).select("s1_idx", "tgt", "tgt_idx")
    _, path, col, fname = CE_MODELS[model]
    old = None
    if (d / fname).exists():  # incremental: only score pairs not scored before
        old = pl.read_parquet(d / fname)
        p = p.join(old.select("s1_idx", "tgt", "tgt_idx"), on=["s1_idx", "tgt", "tgt_idx"], how="anti")
    print(f"[ce] scoring {len(p):,} new {split} pairs with {model}", flush=True)
    if len(p):
        T = _texts(split, TEXT_COL.get(model, "text"))
        a = [T[1][i] for i in p["s1_idx"].to_list()]
        b = [T[k][j] for k, j in zip(p["tgt"].to_list(), p["tgt_idx"].to_list())]
        bs = {"large": 256, "mdeberta": 512}.get(model, 512 if "base" in model else 1024)
        ce = CrossEncoder.load(str(path), max_len=TRAIN_CFG[model][4]).predict(a, b, batch_size=bs)
        p = p.with_columns(pl.Series(col, ce))
    out = pl.concat([old, p]) if old is not None else p
    out.write_parquet(d / fname)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "score"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--n-s1", type=int, default=150000)
    ap.add_argument("--model", default="small", choices=list(CE_MODELS))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    train(a.n_s1, a.model, a.seed) if a.cmd == "train" else score(a.split, a.model)
