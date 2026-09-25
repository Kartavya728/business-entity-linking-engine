"""Stage 5: turn pair probabilities into per-S1 match lists that maximise macro F0.5.

Rules (tuned on out-of-fold predictions):
  1. exclusivity: a target record may be assigned to at most one S1 (its best claimant),
     mirroring the training data where every S2/S3 record links to <= 1 S1;
  2. per S1, pick the prefix of candidates (sorted by p) that maximises the expected
     F0.5 under independent Bernoulli(p) labels, compared against the expected score of
     predicting nothing (= P(no true match) ~ prod(1 - p));
  3. a floor threshold on p for any selected pair.
"""
import numpy as np
import polars as pl

from .metric import f05


def exclusive(df: pl.DataFrame, p: str = "p2") -> pl.DataFrame:
    """Keep only each target's highest-probability claimant."""
    r = pl.col(p).rank("ordinal", descending=True).over(["tgt", "tgt_idx"])
    return df.filter(r == 1)


def select_expected_f(df: pl.DataFrame, p: str = "p2", floor: float = 0.3, miss: float = 0.0,
                      empty_bias: float = 1.0) -> pl.DataFrame:
    """Per S1, choose top-k by p maximising E[F0.5] ~ 1.25*sum_k p / (k + 0.25*(sum p + miss)).

    empty option scores prod(1-p) * empty_bias. Rows below `floor` are never selected.
    """
    d = df.filter(pl.col(p) >= floor * 0.5).sort(["s1_idx", p], descending=[False, True])
    d = d.with_columns(
        pl.col(p).cum_sum().over("s1_idx").alias("_cs"),
        pl.int_range(1, pl.len() + 1).over("s1_idx").alias("_k"),
        pl.col(p).sum().over("s1_idx").alias("_tot"),
        (1 - pl.col(p)).log().sum().over("s1_idx").exp().alias("_p0"),
    )
    d = d.with_columns((1.25 * pl.col("_cs") / (pl.col("_k") + 0.25 * (pl.col("_tot") + miss))).alias("_ef"))
    best = d.group_by("s1_idx").agg(pl.col("_ef").max().alias("_bf"),
                                    pl.col("_k").get(pl.col("_ef").arg_max()).alias("_bk"),
                                    pl.col("_p0").first().alias("_p0"))
    d = d.join(best, on="s1_idx")
    d = d.filter((pl.col("_k") <= pl.col("_bk")) & (pl.col("_bf") > pl.col("_p0") * empty_bias)
                 & (pl.col(p) >= floor))
    return d.drop([c for c in d.columns if c.startswith("_")])


def select_threshold(df: pl.DataFrame, p: str = "p2", t: float = 0.5) -> pl.DataFrame:
    return df.filter(pl.col(p) >= t)


def score(sel: pl.DataFrame, gt: pl.DataFrame, universe: np.ndarray) -> float:
    """Macro F0.5 over `universe` S1 indices; sel/gt have s1_idx, tgt, tgt_idx."""
    u = pl.DataFrame({"s1_idx": universe.astype(np.int32)})
    g = gt.join(u, on="s1_idx", how="semi")
    tp = sel.join(g, on=["s1_idx", "tgt", "tgt_idx"], how="semi").group_by("s1_idx").len("tp")
    npred = sel.join(u, on="s1_idx", how="semi").group_by("s1_idx").len("np")
    ntrue = g.group_by("s1_idx").len("nt")
    t = (u.join(npred, on="s1_idx", how="left").join(ntrue, on="s1_idx", how="left")
          .join(tp, on="s1_idx", how="left").fill_null(0))
    return float(f05(t["np"].to_numpy(), t["nt"].to_numpy(), t["tp"].to_numpy()).mean())
