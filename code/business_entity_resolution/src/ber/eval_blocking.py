"""Blocking diagnostics on the training split: recall vs candidate cap.

Usage: python -m ber.eval_blocking
"""
import polars as pl

from .config import split_dir


def main():
    d = split_dir("train")
    cand = pl.read_parquet(d / "candidates.parquet")
    gt = pl.read_parquet(d / "gt.parquet")
    s1 = pl.read_parquet(d / "source1.parquet", columns=["idx", "country"])
    n_s1 = len(s1)
    g = gt.join(cand.select("s1_idx", "tgt", "tgt_idx", "rrf_r", "dense_r", "sparse_r", "rdense_r"),
                on=["s1_idx", "tgt", "tgt_idx"], how="left")
    g = g.join(s1.rename({"idx": "s1_idx"}), on="s1_idx")
    print(f"true pairs {len(g):,}; any-path recall {g['rrf_r'].is_not_null().mean():.5f}")
    for col in ("dense_r", "sparse_r", "rdense_r"):
        print(f"  {col} alone: {g[col].is_not_null().mean():.5f}")
    for cap in (3, 5, 8, 10, 12, 15, 20, 25, 30, 40, 60):
        c = cand.filter(pl.col("rrf_r") <= cap)
        rec = (g["rrf_r"] <= cap).fill_null(False).mean()
        by = g.group_by("tgt", "country").agg(((pl.col("rrf_r") <= cap).fill_null(False).mean()).alias("r"))
        by = " ".join(f"S{t}/{c}={r:.4f}" for t, c, r in by.sort("tgt", "country").rows())
        print(f"cap {cap:3d}: recall {rec:.5f}  cand/S1 {len(c) / n_s1:6.2f}   {by}")


if __name__ == "__main__":
    main()
