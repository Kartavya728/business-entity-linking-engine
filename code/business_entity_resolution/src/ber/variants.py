"""Re-apply the decision rule to saved test probabilities with per-country overrides.

Usage:
  python -m ber.variants --out <dir> [--country France --floor 0.6 --empty-bias 3.0]
Reads work/test/p2.parquet (written by ber.infer) and work/models/decision.json; countries
not overridden use the tuned rule. Writes <dir>/matching_results.tsv.
"""
import argparse
import json
from pathlib import Path

import polars as pl

from .config import split_dir
from .decide import exclusive, select_expected_f
from .infer import decide, id_lists
from .io import write_id_lists
from .ranker import MODEL_DIR


def boost_same_address(p2: pl.DataFrame, d, country: str, level: float, min_ns: float) -> pl.DataFrame:
    """Raise p2 of `country` pairs whose normalised addresses are identical (non-empty) and whose
    filler-free names are similar (token-set ratio >= min_ns): p2 -> level + (1 - level) * p2.

    Label-free rationale: in the training countries distractor records never keep the exact
    address (same-address rate 0.05% for non-matches vs 22% for matches), so a same-address
    near-name pair is almost always a true copy. The mapping is monotone, so exclusive
    assignment still prefers the stronger claimant."""
    from rapidfuzz import fuzz
    s1 = pl.read_parquet(d / "source1.parquet", columns=["idx", "addr_n", "name_f", "country"])
    s1 = s1.rename({"idx": "s1_idx", "addr_n": "a1", "name_f": "n1"})
    parts = []
    for k in (2, 3):
        t = pl.read_parquet(d / f"source{k}.parquet", columns=["idx", "addr_n", "name_f"])
        t = t.rename({"idx": "tgt_idx", "addr_n": "a2", "name_f": "n2"})
        x = (p2.filter(pl.col("tgt") == k).join(s1, on="s1_idx").join(t, on="tgt_idx")
               .filter((pl.col("country") == country) & (pl.col("a1") == pl.col("a2")) & (pl.col("a1") != "")))
        parts.append(x.select("s1_idx", "tgt", "tgt_idx", "n1", "n2"))
    x = pl.concat(parts)
    x = x.filter(pl.Series([fuzz.token_set_ratio(a, b) >= min_ns for a, b in zip(x["n1"], x["n2"])]))
    x = x.select("s1_idx", "tgt", "tgt_idx", pl.lit(True).alias("boost"))
    out = (p2.join(x, on=["s1_idx", "tgt", "tgt_idx"], how="left")
             .with_columns(pl.when(pl.col("boost")).then(level + (1 - level) * pl.col("p2"))
                             .otherwise(pl.col("p2")).alias("p2")).drop("boost"))
    print(f"[variants] {country}: boosted {len(x):,} same-address pairs "
          f"({(p2.join(x, on=['s1_idx', 'tgt', 'tgt_idx'])['p2'] < 0.5).sum():,} were below 0.5)")
    return out


def main(out: str, country: str, floor: float, empty_bias: float, boost: float = 0.0, boost_ns: float = 80,
         p2_file: str = "p2.parquet"):
    d = split_dir("test")
    s1 = pl.read_parquet(d / "source1.parquet", columns=["entity_id", "country"])
    c = s1["country"].to_numpy(); ids = s1["entity_id"].to_numpy()
    tids = {k: pl.read_parquet(d / f"source{k}.parquet", columns=["entity_id"])["entity_id"].to_numpy() for k in (2, 3)}
    p2 = pl.read_parquet(d / p2_file).select("s1_idx", "tgt", "tgt_idx", "p2")
    if boost > 0:
        p2 = boost_same_address(p2, d, country or "France", boost, boost_ns)
    cfg = json.load(open(MODEL_DIR / "decision.json"))
    sel = decide(p2, cfg)
    if country and not boost:
        m = lambda df: pl.Series(c[df["s1_idx"].to_numpy()] == country)
        alt = select_expected_f(exclusive(p2.filter(m(p2))), floor=floor, empty_bias=empty_bias)
        n_old = sel.filter(m(sel)).height
        sel = pl.concat([sel.filter(~m(sel)), alt.select(sel.columns)])
        print(f"[variants] {country}: {n_old:,} -> {alt.height:,} matched pairs "
              f"(floor {floor}, empty_bias {empty_bias})")
    o = Path(out); o.mkdir(parents=True, exist_ok=True)
    write_id_lists(o / "matching_results.tsv", ids, id_lists(sel, len(ids), tids, "p2"),
                   ("source1_entity_id", "matched_entity_ids"))
    print(f"[variants] wrote {o / 'matching_results.tsv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--country", default="")
    ap.add_argument("--floor", type=float, default=0.3)
    ap.add_argument("--empty-bias", type=float, default=1.6)
    ap.add_argument("--boost", type=float, default=0.0, help="same-address boost level (0 = off)")
    ap.add_argument("--boost-ns", type=float, default=80)
    ap.add_argument("--p2-file", default="p2.parquet")
    a = ap.parse_args()
    main(a.out, a.country, a.floor, a.empty_bias, a.boost, a.boost_ns, a.p2_file)
