"""Blocking diagnostics on the training split: recall vs candidate cap.

Usage: python -m ber.eval_blocking
"""
import polars as pl

from .candidates import apply_cap
from .config import split_dir


def main():
    d = split_dir("train")
    cand = pl.read_parquet(d / "candidates_full.parquet",
                           columns=["s1_idx", "tgt", "tgt_idx", "dense_r", "dense_r2", "rdense_r"])
    gt = pl.read_parquet(d / "gt.parquet")
    s1 = pl.read_parquet(d / "source1.parquet", columns=["idx", "country"])
    n_s1 = len(s1)
    g = gt.join(cand, on=["s1_idx", "tgt", "tgt_idx"], how="left")
    g = g.join(s1.rename({"idx": "s1_idx"}), on="s1_idx")
    print(f"true pairs {len(g):,}; any-path recall {g['dense_r2'].is_not_null().mean():.5f}")
    for col in ("dense_r", "rdense_r"):
        print(f"  {col} alone: {g[col].is_not_null().mean():.5f}")
    for cap in (3, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40):
        c = apply_cap(cand, cap)
        hit = gt.join(c.select("s1_idx", "tgt", "tgt_idx"), on=["s1_idx", "tgt", "tgt_idx"], how="semi")
        # S1 entities whose full truth is covered (upper bound on per-entity recall=1)
        print(f"cap {cap:3d}: recall {len(hit) / len(gt):.5f}  cand/S1 {len(c) / n_s1:6.2f}")


if __name__ == "__main__":
    main()
