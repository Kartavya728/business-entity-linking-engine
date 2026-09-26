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


def main(out: str, country: str, floor: float, empty_bias: float):
    d = split_dir("test")
    s1 = pl.read_parquet(d / "source1.parquet", columns=["entity_id", "country"])
    c = s1["country"].to_numpy(); ids = s1["entity_id"].to_numpy()
    tids = {k: pl.read_parquet(d / f"source{k}.parquet", columns=["entity_id"])["entity_id"].to_numpy() for k in (2, 3)}
    p2 = pl.read_parquet(d / "p2.parquet").select("s1_idx", "tgt", "tgt_idx", "p2")
    cfg = json.load(open(MODEL_DIR / "decision.json"))
    sel = decide(p2, cfg)
    if country:
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
    a = ap.parse_args()
    main(a.out, a.country, a.floor, a.empty_bias)
