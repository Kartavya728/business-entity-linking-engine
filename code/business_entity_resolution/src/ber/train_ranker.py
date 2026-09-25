"""Train the two-stage ranker on train features and tune the decision rule on OOF.

Usage: python -m ber.train_ranker stage1|stage2 [--n-cv 4]
Writes work/models/*, work/train/oof.parquet, work/models/decision.json
"""
import argparse
import itertools
import json
import time

import numpy as np
import polars as pl

from .config import split_dir
from .decide import exclusive, score, select_expected_f, select_threshold
from .features import group_context
from .ranker import MODEL_DIR, feat_cols, fit_stage, stage2_context
from .splits import ENC_FOLDS, s1_folds


def tune(oof: pl.DataFrame, gt: pl.DataFrame, universe: np.ndarray):
    """Grid over decision rules; returns best config and its macro F0.5."""
    res = []
    for excl in (False, True):
        d = exclusive(oof) if excl else oof
        for t in (0.3, 0.4, 0.5, 0.6, 0.7):
            res.append((score(select_threshold(d, t=t), gt, universe), dict(rule="thr", excl=excl, t=t)))
        for floor, eb in itertools.product((0.05, 0.1, 0.2, 0.3), (0.8, 1.0, 1.2)):
            s = score(select_expected_f(d, floor=floor, empty_bias=eb), gt, universe)
            res.append((s, dict(rule="ef", excl=excl, floor=floor, empty_bias=eb)))
    res.sort(key=lambda r: -r[0])
    for s, c in res[:8]:
        print(f"  {s:.5f} {c}")
    return res[0]


def load_train(n_cv: int):
    d = split_dir("train")
    t = time.time()
    df = pl.read_parquet(d / "features.parquet")
    n_s1 = pl.read_parquet(d / "source1.parquet", columns=["idx"]).height
    folds = s1_folds(n_s1)
    fold = folds[df["s1_idx"].to_numpy()]
    cv = np.where(np.isin(fold, ENC_FOLDS), -1, (fold - 3) % n_cv).astype(np.int8)
    df = df.with_columns(pl.Series("cv", cv))
    print(f"[train] features {df.shape} loaded in {time.time() - t:.0f}s; pos rate {df['y'].mean():.4f}", flush=True)
    return df, folds


def stage1(n_cv: int):
    """Fit stage 1 and write p1 for every train candidate (OOF on R, fold-average on E)."""
    d = split_dir("train")
    df, folds = load_train(n_cv)
    df = df.with_columns(pl.Series("p1", fit_stage(df, "stage1", feat_cols(df), n_cv)))
    df.select("s1_idx", "tgt", "tgt_idx", "y", "cv", "p1").write_parquet(d / "p1.parquet")
    gt = pl.read_parquet(d / "gt.parquet")
    universe = np.flatnonzero(~np.isin(folds, ENC_FOLDS))
    print("[train] stage-1 decision grid:")
    tune(df.filter(pl.col("cv") >= 0).select("s1_idx", "tgt", "tgt_idx", pl.col("p1").alias("p2")), gt, universe)


def attach_stage2_inputs(df: pl.DataFrame, d) -> pl.DataFrame:
    """Join p1 (+ cross-encoder score when available) and add p1 competition context."""
    df = df.join(pl.read_parquet(d / "p1.parquet").select("s1_idx", "tgt", "tgt_idx", "p1"),
                 on=["s1_idx", "tgt", "tgt_idx"], how="left")
    if (d / "ce.parquet").exists():
        df = df.join(pl.read_parquet(d / "ce.parquet"), on=["s1_idx", "tgt", "tgt_idx"], how="left")
        df = group_context(df.with_columns(pl.col("ce").fill_null(-1.0)), "ce", "cectx")
    return stage2_context(df, "p1")


def stage2(n_cv: int):
    d = split_dir("train")
    df, folds = load_train(n_cv)
    df = attach_stage2_inputs(df, d)
    cols2 = feat_cols(df) + ["p1"]
    df = df.with_columns(pl.Series("p2", fit_stage(df, "stage2", cols2, n_cv)))
    oof = df.select("s1_idx", "tgt", "tgt_idx", "y", "cv", "p1", "p2")
    oof.write_parquet(d / "oof.parquet")
    gt = pl.read_parquet(d / "gt.parquet")
    universe = np.flatnonzero(~np.isin(folds, ENC_FOLDS))
    print("[train] stage-2 decision grid:")
    best_s, best_c = tune(oof.filter(pl.col("cv") >= 0), gt, universe)
    json.dump(best_c, open(MODEL_DIR / "decision.json", "w"))
    print(f"[train] best OOF macro F0.5 = {best_s:.5f} with {best_c}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["stage1", "stage2"])
    ap.add_argument("--n-cv", type=int, default=4)
    a = ap.parse_args()
    stage1(a.n_cv) if a.stage == "stage1" else stage2(a.n_cv)
