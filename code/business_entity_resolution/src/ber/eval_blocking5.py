"""Recall vs candidate-set size for the v5 hybrid blocking rule (train, all links).

Usage: python -m ber.eval_blocking5
"""
import itertools

import polars as pl

from .candidates_v5 import apply_rule
from .config import split_dir


def main():
    d = split_dir("train")
    cand = pl.read_parquet(d / "candidates5_full.parquet",
                           columns=["s1_idx", "tgt", "tgt_idx", "dense_r2", "sparse_r2", "rdense_r"])
    gt = pl.read_parquet(d / "gt.parquet")
    n1 = pl.read_parquet(d / "source1.parquet", columns=["idx"]).height
    k = ["s1_idx", "tgt", "tgt_idx"]
    print(f"union recall {gt.join(cand.select(k), on=k, how='semi').height / len(gt):.5f}  "
          f"size/S1 {len(cand) / n1:.1f}")
    for cd, cs in itertools.product((10, 15, 20, 25), (0, 3, 5, 8, 10)):
        c = apply_rule(cand, cd, cs)
        r = gt.join(c.select(k), on=k, how="semi").height / len(gt)
        print(f"dense<{cd:2d} | sparse<{cs:2d}: recall {r:.5f}  cand/S1 {len(c) / n1:6.2f}", flush=True)


if __name__ == "__main__":
    main()
