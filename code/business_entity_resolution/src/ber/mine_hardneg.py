"""Mine hard negatives for the second bi-encoder round (biencoder_v2).

For every encoder-fold (E) S1 entity, keep the 3 highest-cosine non-matching candidates per
target source from the first-round blocking output (work/train/candidates.parquet).

  python -m ber.mine_hardneg      # writes work/train/hardneg_e.parquet
"""
import numpy as np
import polars as pl

from .config import split_dir
from .splits import ENC_FOLDS, s1_folds


def main(per_source: int = 3):
    d = split_dir("train")
    n = pl.read_parquet(d / "source1.parquet", columns=["idx"]).height
    e = np.flatnonzero(np.isin(s1_folds(n), ENC_FOLDS)).astype(np.int32)
    c = (pl.scan_parquet(d / "candidates.parquet").select("s1_idx", "tgt", "tgt_idx", "dense_s")
           .filter(pl.col("s1_idx").is_in(e)).collect())
    gt = pl.read_parquet(d / "gt.parquet").with_columns(y=pl.lit(1, pl.Int8))
    c = c.join(gt, on=["s1_idx", "tgt", "tgt_idx"], how="left").with_columns(pl.col("y").fill_null(0))
    neg = (c.filter(pl.col("y") == 0).sort("dense_s", descending=True)
             .group_by("s1_idx", "tgt", maintain_order=True).head(per_source))
    neg.select("s1_idx", "tgt", "tgt_idx").write_parquet(d / "hardneg_e.parquet")
    print(f"[hardneg] {len(neg):,} hard negatives for {neg['s1_idx'].n_unique():,} E-fold S1")


if __name__ == "__main__":
    main()
