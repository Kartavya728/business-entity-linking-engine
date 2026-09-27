"""Train the two-stage ranker and tune the decision rule on out-of-fold predictions.

  python -m ber.train_ranker stage1   # full pair features of the training subset
  python -m ber.train_ranker stage2   # p1 + competition context + CE + compact features
Stage 2 context is computed over *all* train candidates (p1.parquet from the streaming
pass), so every pair sees its true competitors. Training/evaluation rows are the subset
S1 entities (random 500K of the ranker folds); OOF predictions give the validation score.
"""
import argparse
import os
import itertools
import json
import time

import numpy as np
import polars as pl

from .config import split_dir
from .decide import exclusive, score, select_expected_f, select_threshold
from .features import group_context

from .ranker import MODEL_DIR, feat_cols, fit_stage, fit_stage_lgb, stage2_context, subset_cv

N_CV = 4
S2_MIN_P1 = float(os.environ.get("BER_S2_MIN_P1", "0.0005"))


def tune(oof: pl.DataFrame, gt: pl.DataFrame, universe: np.ndarray, p: str = "p2"):
    """Grid over decision rules; returns (best score, config)."""
    oof = oof.rename({p: "p2"}) if p != "p2" else oof
    res = []
    for excl in (False, True):
        d = exclusive(oof) if excl else oof
        for t in (0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95):
            res.append((score(select_threshold(d, t=t), gt, universe), dict(rule="thr", excl=excl, t=t)))
        for floor, eb in itertools.product((0.3, 0.4, 0.5, 0.6), (1.0, 1.3, 1.6)):
            s = score(select_expected_f(d, floor=floor, empty_bias=eb), gt, universe)
            res.append((s, dict(rule="ef", excl=excl, floor=floor, empty_bias=eb)))
    res.sort(key=lambda r: -r[0])
    for s, c in res[:6]:
        print(f"  {s:.5f} {c}", flush=True)
    return res[0]


def stage1():
    d = split_dir("train")
    t = time.time()
    df = pl.read_parquet(d / "features.parquet")
    df = df.with_columns(pl.Series("cv", subset_cv(df["s1_idx"].to_numpy(), d, N_CV)))
    print(f"[train] stage-1 features {df.shape} in {time.time() - t:.0f}s; pos rate {df['y'].mean():.4f}", flush=True)
    df = df.with_columns(pl.Series("p1", fit_stage(df, "stage1", feat_cols(df), N_CV)))
    gt = pl.read_parquet(d / "gt.parquet")
    universe = np.load(d / "subset_s1.npy")
    print("[train] stage-1 OOF decision grid:")
    tune(df.select("s1_idx", "tgt", "tgt_idx", "p1"), gt, universe, p="p1")


def stage2_frame(split: str) -> pl.DataFrame:
    """p1 table (+ CE) with competition context over all candidates of the split."""
    d = split_dir(split)
    df = pl.read_parquet(d / "p1.parquet")
    # stage 2 only re-ranks pairs stage 1 considers possible (same rule for train and test);
    # rows below never reach the decision floor and dominated RAM (100M train rows -> 16M)
    df = df.filter(pl.col("p1") >= S2_MIN_P1)
    from .ce_registry import CE_MODELS
    skip = set(os.environ.get("BER_CE_SKIP", "qwen05,qwen05_lora").split(","))  # e.g. BER_CE_SKIP=large2,bgem3
    for name, m in CE_MODELS.items():  # every cross-encoder whose score file exists
        fname, col, pref = m["file"], m["col"], m["col"] + "ctx"
        if name not in skip and (d / fname).exists():
            df = df.join(pl.read_parquet(d / fname), on=["s1_idx", "tgt", "tgt_idx"], how="left")
            df = group_context(df.with_columns(pl.col(col).fill_null(-1.0)), col, pref)
    df = stage2_context(df, "p1")
    if os.environ.get("BER_DT") == "1" and split == "test":  # word-difference features (see adapt.py)
        from .adapt import diff_token_feats
        df = diff_token_feats(df, split, procs=12)
    return df


def stage2():
    d = split_dir("train")
    t = time.time()
    df = stage2_frame("train")
    df = df.with_columns(pl.Series("cv", subset_cv(df["s1_idx"].to_numpy(), d, N_CV))).filter(pl.col("cv") >= 0)
    gt = pl.read_parquet(d / "gt.parquet").with_columns(pl.lit(1, pl.Int8).alias("y"))
    df = df.join(gt, on=["s1_idx", "tgt", "tgt_idx"], how="left").with_columns(pl.col("y").fill_null(0))
    if os.environ.get("BER_DT") == "1":  # after the subset filter: only training rows need them
        from .adapt import diff_token_feats
        df = diff_token_feats(df, "train", procs=12)
    print(f"[train] stage-2 frame {df.shape} in {time.time() - t:.0f}s", flush=True)
    cols2 = feat_cols(df) + ["p1"]
    px = fit_stage(df, "stage2", cols2, N_CV)
    cands = {"xgb": px}
    if os.environ.get("BER_LGB") == "1":
        pl_ = fit_stage_lgb(df, "stage2lgb", cols2, N_CV)
        cands["lgb"] = pl_
        cands["avg"] = 0.5 * (px + pl_)
    universe = np.load(d / "subset_s1.npy")
    gt = gt.drop("y")
    print("[train] stage-1 (p1) OOF on full context rows:")
    tune(df.select("s1_idx", "tgt", "tgt_idx", "p1"), gt, universe, p="p1")
    best = None
    for blend, pv in cands.items():
        print(f"[train] stage-2 OOF decision grid ({blend}):")
        s_, c_ = tune(df.select("s1_idx", "tgt", "tgt_idx").with_columns(pl.Series("p2", pv)), gt, universe)
        if best is None or s_ > best[0]:
            best = (s_, {**c_, "blend": blend}, pv)
    best_s, best_c, pbest = best
    df.select("s1_idx", "tgt", "tgt_idx", "y", "cv", "p1").with_columns(pl.Series("p2", pbest)).write_parquet(d / "oof.parquet")
    json.dump(best_c, open(MODEL_DIR / "decision.json", "w"))
    print(f"[train] best OOF macro F0.5 = {best_s:.5f} with {best_c}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["stage1", "stage2"])
    a = ap.parse_args()
    stage1() if a.stage == "stage1" else stage2()
