"""Test-time inference: features -> stage 1 -> cross-encoder -> stage 2 -> decision -> TSVs.

Usage: python -m ber.infer [--skip-ce]
Assumes prepare, candidates and `build_features score --split test` (p1) have been run.
Writes output/matching_results.tsv and output/candidate_pairs.tsv.
"""
import argparse
import json

import numpy as np
import polars as pl

from .config import OUT_DIR, split_dir
from .decide import exclusive, select_expected_f, select_threshold
from .io import write_id_lists
from .ranker import MODEL_DIR, predict_stage
from .train_ranker import stage2_frame


def id_lists(df: pl.DataFrame, s1_n: int, tgt_ids: dict, order_col: str):
    """s1 row -> list of target entity_ids (sorted by order_col desc)."""
    out = {}
    df = df.sort(["s1_idx", order_col], descending=[False, True])
    for s, t, j in zip(df["s1_idx"].to_list(), df["tgt"].to_list(), df["tgt_idx"].to_list()):
        out.setdefault(s, []).append(tgt_ids[t][j])
    return out


def decide(df: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    d = exclusive(df) if cfg.get("excl") else df
    if cfg["rule"] == "thr":
        return select_threshold(d, t=cfg["t"])
    return select_expected_f(d, floor=cfg["floor"], empty_bias=cfg["empty_bias"])


def main(skip_ce: bool):
    d = split_dir("test")
    if not skip_ce:
        from .train_ce import score
        score("test")
    feat = stage2_frame("test")
    feat = feat.with_columns(pl.Series("p2", predict_stage(feat, "stage2")))
    feat.select("s1_idx", "tgt", "tgt_idx", "p1", "p2").filter(pl.col("p2") > 0.01).write_parquet(d / "p2.parquet")
    cfg = json.load(open(MODEL_DIR / "decision.json"))
    sel = decide(feat.select("s1_idx", "tgt", "tgt_idx", "p2"), cfg)

    s1_ids = pl.read_parquet(d / "source1.parquet", columns=["entity_id"])["entity_id"].to_numpy()
    tgt_ids = {k: pl.read_parquet(d / f"source{k}.parquet", columns=["entity_id"])["entity_id"].to_numpy()
               for k in (2, 3)}
    cand = pl.read_parquet(d / "candidates.parquet", columns=["s1_idx", "tgt", "tgt_idx", "dense_s"])
    write_id_lists(OUT_DIR / "candidate_pairs.tsv", s1_ids, id_lists(cand, len(s1_ids), tgt_ids, "dense_s"),
                   ("source1_entity_id", "candidate_entity_ids"))
    write_id_lists(OUT_DIR / "matching_results.tsv", s1_ids, id_lists(sel, len(s1_ids), tgt_ids, "p2"),
                   ("source1_entity_id", "matched_entity_ids"))
    n_match = sel["s1_idx"].n_unique()
    print(f"[infer] {len(sel):,} matches over {n_match:,}/{len(s1_ids):,} S1 "
          f"({1 - n_match / len(s1_ids):.4f} predicted singletons); cand/S1 {len(cand) / len(s1_ids):.2f}")
    s1c = pl.read_parquet(d / "source1.parquet", columns=["idx", "country"])
    st = (s1c.join(sel.group_by("s1_idx").len("n").rename({"s1_idx": "idx"}), on="idx", how="left")
             .fill_null(0).group_by("country").agg(pl.col("n").mean().alias("avg_matches"),
                                                   (pl.col("n") == 0).mean().alias("empty_rate")))
    print(st)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-ce", action="store_true")
    main(ap.parse_args().skip_ce)
