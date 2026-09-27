"""Build a submission from saved stage-2 probability files, with per-country choices.

Each country takes pair probabilities from one file (or a weighted blend of two), optionally
re-weighted for label shift, and is decided with its own rule. Countries are separable:
blocking, exclusivity and the per-S1 decision never cross a country boundary.

  python -m ber.combine --out submissions/X \
      --default p2_v8.parquet --default-cfg decision_v8.json \
      --country France --src p2_v6.parquet [--src2 p2_v8.parquet --w2 0.5] \
      [--odds 0.5 | --em] [--floor 0.6 --empty-bias 1.0] [--empty]

--odds k : multiply the country's match odds by k before deciding (label-shift correction).
--em     : estimate k with the EM prior re-estimation of Saerens et al. (2002), using the
           training-country prior of the same candidate universe (train OOF, p2 > 0.01).
--empty  : predict no match for the country (diagnostic probe).
Paths without '/' are resolved in work/test (probabilities) and work/models (configs).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from .config import split_dir
from .decide import exclusive, select_expected_f, select_threshold
from .infer import id_lists
from .io import write_id_lists
from .ranker import KEYS, MODEL_DIR

P_MIN = 0.01  # probability files keep pairs with p2 > 0.01


def _p2(path: str) -> pl.DataFrame:
    """Probabilities keyed by row indices (work/test/p2_*.parquet) or by entity ids (reference/*.parquet,
    portable across machines because indices depend on the prepared parquet row order)."""
    p = Path(path) if "/" in path else split_dir("test") / path
    df = pl.read_parquet(p)
    if "s1_entity_id" in df.columns:
        d = split_dir("test")
        s1 = pl.read_parquet(d / "source1.parquet", columns=["idx", "entity_id"]).rename({"idx": "s1_idx", "entity_id": "s1_entity_id"})
        tg = pl.concat([pl.read_parquet(d / f"source{k}.parquet", columns=["idx", "entity_id"])
                          .rename({"idx": "tgt_idx", "entity_id": "tgt_entity_id"}).with_columns(pl.lit(k, pl.Int8).alias("tgt"))
                        for k in (2, 3)])
        n = len(df)
        df = df.join(s1, on="s1_entity_id").join(tg, on="tgt_entity_id")
        if len(df) < n:
            print(f"[combine] warning: {n - len(df):,} reference pairs have ids missing from the test data")
    return df.select(*KEYS, pl.col("p2").cast(pl.Float64))


def _cfg(path: str) -> dict:
    p = Path(path) if "/" in path else MODEL_DIR / path
    return json.load(open(p))


def decide(p2: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    d = exclusive(p2) if cfg.get("excl", True) else p2
    if cfg.get("rule", "ef") == "thr":
        return select_threshold(d, t=cfg["t"])
    return select_expected_f(d, floor=cfg["floor"], empty_bias=cfg["empty_bias"])


def shift_odds(p: np.ndarray, k: float) -> np.ndarray:
    return k * p / (k * p + (1 - p))


def em_odds(p: np.ndarray, prior_train: float, iters: int = 100) -> float:
    """Saerens et al. (2002): re-estimate the positive prior of a new domain from the model's
    posteriors, assuming p(x | y) is shared; returns the odds multiplier k."""
    q = prior_train
    for _ in range(iters):
        k = (q / (1 - q)) / (prior_train / (1 - prior_train))
        q_new = float(shift_odds(p, k).mean())
        if abs(q_new - q) < 1e-7:
            break
        q = q_new
    return (q / (1 - q)) / (prior_train / (1 - prior_train))


def train_prior() -> float:
    o = pl.read_parquet(split_dir("train") / "oof.parquet", columns=["p2", "y"]).filter(pl.col("p2") > P_MIN)
    return float(o["y"].mean())


def main(a):
    d = split_dir("test")
    s1 = pl.read_parquet(d / "source1.parquet", columns=["entity_id", "country"])
    ctry = s1["country"].to_numpy()
    tids = {k: pl.read_parquet(d / f"source{k}.parquet", columns=["entity_id"])["entity_id"].to_numpy() for k in (2, 3)}
    in_c = lambda df, c: pl.Series(ctry[df["s1_idx"].to_numpy()] == c)
    base = _p2(a.default)
    sel = decide(base.filter(~in_c(base, a.country)), _cfg(a.default_cfg))
    if not a.empty:
        p = _p2(a.src)
        p = p.filter(in_c(p, a.country))
        if a.src2:
            q = _p2(a.src2); q = q.filter(in_c(q, a.country))
            p = (p.join(q, on=KEYS, how="full", coalesce=True, suffix="_2")
                  .with_columns(((1 - a.w2) * pl.col("p2").fill_null(P_MIN / 2) + a.w2 * pl.col("p2_2").fill_null(P_MIN / 2)).alias("p2"))
                  .select(*KEYS, "p2"))
        k = a.odds
        if a.em:
            pt = train_prior()
            k = em_odds(p["p2"].to_numpy().astype(np.float64), pt)
            print(f"[combine] EM label shift: train prior {pt:.4f} -> odds multiplier {k:.3f}")
        if k != 1.0:
            p = p.with_columns(pl.Series("p2", shift_odds(p["p2"].to_numpy().astype(np.float64), k)))
        cfg = dict(_cfg(a.src_cfg)) if a.src_cfg else dict(_cfg(a.default_cfg))
        if a.floor is not None:
            cfg["floor"] = a.floor
        if a.empty_bias is not None:
            cfg["empty_bias"] = a.empty_bias
        alt = decide(p, cfg)
        print(f"[combine] {a.country}: {alt.height:,} matched pairs over {alt['s1_idx'].n_unique():,} S1 "
              f"(odds x{k:.3f}, cfg {cfg})")
        sel = pl.concat([sel.select(*KEYS, pl.col("p2").cast(pl.Float64)), alt.select(*KEYS, pl.col("p2").cast(pl.Float64))])
    else:
        print(f"[combine] {a.country}: predicting no matches (probe)")
    o = Path(a.out); o.mkdir(parents=True, exist_ok=True)
    write_id_lists(o / "matching_results.tsv", s1["entity_id"].to_numpy(), id_lists(sel, len(s1), tids, "p2"),
                   ("source1_entity_id", "matched_entity_ids"))
    json.dump(vars(a), open(o / "combine_args.json", "w"), indent=1)
    print(f"[combine] wrote {o / 'matching_results.tsv'} ({sel.height:,} pairs)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--default", default="p2.parquet", help="probabilities for the other countries")
    ap.add_argument("--default-cfg", default="decision.json")
    ap.add_argument("--country", default="France")
    ap.add_argument("--src", default="p2.parquet")
    ap.add_argument("--src-cfg", default="", help="decision config for the country (default: --default-cfg)")
    ap.add_argument("--src2", default="")
    ap.add_argument("--w2", type=float, default=0.5)
    ap.add_argument("--odds", type=float, default=1.0)
    ap.add_argument("--em", action="store_true")
    ap.add_argument("--floor", type=float, default=None)
    ap.add_argument("--empty-bias", type=float, default=None)
    ap.add_argument("--empty", action="store_true")
    main(ap.parse_args())
