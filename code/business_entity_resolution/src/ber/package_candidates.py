"""Final candidate set = stage-1 filtered blocking output (p1 >= config.CAND_MIN_P1, the pairs stage 2
scores) plus every predicted match.

The raw hybrid blocking keeps ~46 candidates per S1 (99.86% recall on train); the stage-1
ranker's filter at p1 >= 0.01 keeps ~4.7 per S1 on test at 99.60% train pair recall (6.4 per S1
at 0.001). Stage 2 scores exactly these pairs; matches taken from another run (per-country
hybrids) are added back so matching_results stays a subset of the candidates.

  python -m ber.package_candidates --matching submissions/<v>/matching_results.tsv --out output/candidate_pairs.tsv
"""
import argparse

import polars as pl

from .config import CAND_MIN_P1, split_dir
from .io import write_id_lists

P1_MIN = CAND_MIN_P1


def main(matching: str, out: str):
    d = split_dir("test")
    s1_ids = pl.read_parquet(d / "source1.parquet", columns=["entity_id"])["entity_id"].to_numpy()
    tids = {k: pl.read_parquet(d / f"source{k}.parquet", columns=["entity_id"])["entity_id"] for k in (2, 3)}
    p = pl.read_parquet(d / "p1.parquet", columns=["s1_idx", "tgt", "tgt_idx", "p1"]).filter(pl.col("p1") >= P1_MIN)
    ids = pl.concat([pl.DataFrame({"eid": tids[k], "tgt": pl.Series([k] * len(tids[k]), dtype=pl.Int8),
                                   "tgt_idx": pl.Series(range(len(tids[k])), dtype=p["tgt_idx"].dtype)}) for k in (2, 3)])
    p = p.join(ids, on=["tgt", "tgt_idx"]).select("s1_idx", "eid", "p1")
    m = pl.read_csv(matching, separator="\t", schema_overrides={"matched_entity_ids": pl.Utf8})
    s1 = pl.DataFrame({"source1_entity_id": s1_ids, "s1_idx": pl.Series(range(len(s1_ids)), dtype=p["s1_idx"].dtype)})
    m = (m.join(s1, on="source1_entity_id").with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
          .explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
          .select("s1_idx", pl.col("matched_entity_ids").alias("eid"), pl.lit(2.0).alias("p1")))
    m = m.with_columns(pl.col("s1_idx").cast(p["s1_idx"].dtype), pl.col("p1").cast(pl.Float64))
    c = pl.concat([m, p.with_columns(pl.col("p1").cast(pl.Float64))]).group_by("s1_idx", "eid").agg(pl.col("p1").max()).sort(["s1_idx", "p1"], descending=[False, True])
    lists = {}
    for s, e in zip(c["s1_idx"].to_list(), c["eid"].to_list()):
        lists.setdefault(s, []).append(e)
    write_id_lists(out, s1_ids, lists, ("source1_entity_id", "candidate_entity_ids"))
    print(f"[package] {len(c):,} candidate pairs ({len(c) / len(s1_ids):.2f} per S1), "
          f"{len(m):,} matched pairs all included -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--matching", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    main(a.matching, a.out)
