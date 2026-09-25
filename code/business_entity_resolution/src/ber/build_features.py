"""Stage 3 driver: compute pair features for every candidate of a split.

Usage: python -m ber.build_features --split train|test
Writes work/<split>/features.parquet (one row per candidate pair; `y` on train).
"""
import argparse
import time

import numpy as np
import polars as pl

from .config import split_dir
from .features import TfidfBank, group_context, name_freq, pair_features

CHUNK = 6_000_000  # candidate rows per feature batch


def build(split: str):
    d = split_dir(split)
    cand = pl.read_parquet(d / "candidates.parquet")
    s1 = pl.read_parquet(d / "source1.parquet")
    parts = []
    for k in (2, 3):
        tg = pl.read_parquet(d / f"source{k}.parquet")
        f1, f2 = name_freq(s1, tg)
        t = time.time()
        bank = TfidfBank(s1, tg)
        print(f"[feat] {split} S{k}: tfidf bank in {time.time() - t:.0f}s", flush=True)
        ck = cand.filter(pl.col("tgt") == k)
        for s in range(0, len(ck), CHUNK):
            t = time.time()
            c = ck.slice(s, CHUNK)
            f = pair_features(c, s1, tg, bank)
            f = f.with_columns(pl.Series("nfreq1", f1[c["s1_idx"].to_numpy()]),
                               pl.Series("nfreq2", f2[c["tgt_idx"].to_numpy()]))
            parts.append(f)
            print(f"[feat] {split} S{k} rows {s:,}-{s + len(c):,} in {time.time() - t:.0f}s", flush=True)
        del tg, bank
    feat = pl.concat(parts)
    feat = group_context(feat, "dense_s", "dctx")
    feat = group_context(feat, "sparse_s", "sctx")
    if split == "train":
        gt = pl.read_parquet(d / "gt.parquet").with_columns(pl.lit(1, pl.Int8).alias("y"))
        feat = feat.join(gt, on=["s1_idx", "tgt", "tgt_idx"], how="left").with_columns(pl.col("y").fill_null(0))
    feat = feat.with_columns(pl.col(pl.Float64).cast(pl.Float32))
    feat.write_parquet(d / "features.parquet", compression="zstd")
    print(f"[feat] {split}: {feat.shape} written", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    build(ap.parse_args().split)
