"""Stage 3 driver: pair features, computed in streamed chunks (bounded RAM/disk).

  python -m ber.build_features subset            # train: features for a sample of R-fold S1
  python -m ber.build_features score --split S   # all candidates: features -> stage-1 p1,
                                                 # keep keys + p1 + KEEP columns only
Retrieval-group context (dctx_*) is computed on the full candidate table first, so it
is identical whether a pair is processed in the subset or in the streaming pass.
"""
import argparse
import time

import numpy as np
import polars as pl

from .config import split_dir
from .features import TfidfBank, group_context, name_freq, pair_features
from .splits import ENC_FOLDS, s1_folds

CHUNK = 5_000_000
SUBSET_S1 = 500_000
# compact pair features carried into stage 2 (besides p1 / context / CE)
KEEP = ["dense_s", "dense_r2", "rdense_r", "n_tset", "n_ratio", "nk_ratio", "a_tset", "a_ratio",
        "n_idf", "a_idf", "n_c3", "a_c3", "hno_big_miss", "hno_anyfix", "zip_eq", "leg_conflict",
        "leg_eq", "a_empty2", "nfreq1", "nfreq2", "num_miss1", "num_miss2", "nns_ratio"]


def load_candidates(split: str) -> pl.DataFrame:
    d = split_dir(split)
    cand = pl.read_parquet(d / "candidates.parquet")
    cand = group_context(cand, "dense_s", "dctx")
    return cand.with_columns(pl.col(pl.Float64).cast(pl.Float32))


def feature_chunks(split: str, cand: pl.DataFrame):
    """Yield feature frames for `cand`, per target source, in CHUNK-row pieces."""
    d = split_dir(split)
    s1 = pl.read_parquet(d / "source1.parquet")
    for k in (2, 3):
        ck = cand.filter(pl.col("tgt") == k)
        if len(ck) == 0:
            continue
        tg = pl.read_parquet(d / f"source{k}.parquet")
        f1, f2 = name_freq(s1, tg)
        t = time.time()
        bank = TfidfBank(s1, tg)
        print(f"[feat] {split} S{k}: tfidf bank in {time.time() - t:.0f}s", flush=True)
        for s in range(0, len(ck), CHUNK):
            t = time.time()
            c = ck.slice(s, CHUNK)
            f = pair_features(c, s1, tg, bank)
            f = f.with_columns(pl.Series("nfreq1", f1[c["s1_idx"].to_numpy()]),
                               pl.Series("nfreq2", f2[c["tgt_idx"].to_numpy()]))
            f = f.with_columns(pl.col(pl.Float64).cast(pl.Float32))
            print(f"[feat] {split} S{k} rows {s:,}-{s + len(c):,}/{len(ck):,} in {time.time() - t:.0f}s", flush=True)
            yield f
        del tg, bank


def label(f: pl.DataFrame, gt: pl.DataFrame) -> pl.DataFrame:
    return f.join(gt, on=["s1_idx", "tgt", "tgt_idx"], how="left").with_columns(pl.col("y").fill_null(0))


def subset():
    """Full features for a random sample of R-fold S1 entities (stage-1 training set)."""
    d = split_dir("train")
    cand = load_candidates("train")
    n_s1 = pl.read_parquet(d / "source1.parquet", columns=["idx"]).height
    folds = s1_folds(n_s1)
    r = np.flatnonzero(~np.isin(folds, ENC_FOLDS))
    pick = np.sort(np.random.default_rng(7).choice(r, size=min(SUBSET_S1, len(r)), replace=False)).astype(np.int32)
    np.save(d / "subset_s1.npy", pick)
    cand = cand.filter(pl.col("s1_idx").is_in(pick))
    gt = pl.read_parquet(d / "gt.parquet").with_columns(pl.lit(1, pl.Int8).alias("y"))
    feat = pl.concat([label(f, gt) for f in feature_chunks("train", cand)])
    feat.write_parquet(d / "features.parquet", compression="zstd")
    print(f"[feat] subset: {feat.shape} written ({len(pick):,} S1)", flush=True)


def score(split: str):
    """Stream all candidates: features -> p1 (stage-1 fold models) -> compact parquet."""
    from .ranker import predict_stage, predict_stage_oof
    d = split_dir(split)
    cand = load_candidates(split)
    parts = []
    for f in feature_chunks(split, cand):
        # train: subset rows get their out-of-fold p1, all others the fold-average
        p1 = predict_stage_oof(f, "stage1", d) if split == "train" else predict_stage(f, "stage1")
        keep = [c for c in f.columns if c.startswith("dctx_")] + KEEP
        parts.append(f.select(["s1_idx", "tgt", "tgt_idx"] + keep).with_columns(pl.Series("p1", p1)))
    out = pl.concat(parts)
    out.write_parquet(d / "p1.parquet", compression="zstd")
    print(f"[feat] {split}: p1 for {len(out):,} pairs written", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["subset", "score"])
    ap.add_argument("--split", default="train")
    a = ap.parse_args()
    subset() if a.cmd == "subset" else score(a.split)
