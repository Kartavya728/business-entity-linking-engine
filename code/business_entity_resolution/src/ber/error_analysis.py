"""Where does macro F0.5 go? Break down OOF errors of the tuned decision rule.

Usage: python -m ber.error_analysis [--examples 10]
"""
import argparse
import json

import numpy as np
import polars as pl

from .config import split_dir
from .infer import decide
from .metric import f05
from .ranker import MODEL_DIR


def main(n_ex: int):
    d = split_dir("train")
    oof = pl.read_parquet(d / "oof.parquet")
    cfg = json.load(open(MODEL_DIR / "decision.json"))
    universe = np.load(d / "subset_s1.npy")
    gt = pl.read_parquet(d / "gt.parquet").filter(pl.col("s1_idx").is_in(universe))
    sel = decide(oof.select("s1_idx", "tgt", "tgt_idx", "p2"), cfg)
    k = ["s1_idx", "tgt", "tgt_idx"]
    u = pl.DataFrame({"s1_idx": universe.astype(np.int32)})
    tp = sel.join(gt, on=k, how="semi")
    fp = sel.join(gt, on=k, how="anti")
    fn = gt.join(sel, on=k, how="anti")
    fn_block = fn.join(oof.select(k), on=k, how="anti")   # never a candidate
    fn_model = fn.join(oof.select(k), on=k, how="semi")   # candidate but rejected
    t = (u.join(sel.group_by("s1_idx").len("np"), on="s1_idx", how="left")
          .join(gt.group_by("s1_idx").len("nt"), on="s1_idx", how="left")
          .join(tp.group_by("s1_idx").len("tp"), on="s1_idx", how="left").fill_null(0))
    f = f05(t["np"].to_numpy(), t["nt"].to_numpy(), t["tp"].to_numpy())
    t = t.with_columns(pl.Series("f", f))
    print(f"macro F0.5 {f.mean():.5f} over {len(t):,} S1; cfg {cfg}")
    print(f"pairs: tp {len(tp):,} fp {len(fp):,} fn {len(fn):,} (blocking {len(fn_block):,}, model {len(fn_model):,})")
    loss = 1 - t["f"]
    sing = t["nt"] == 0
    print(f"loss share: singletons with FP {loss.filter(sing).sum() / loss.sum():.3f} "
          f"(n={int((sing & (t['np'] > 0)).sum()):,}); "
          f"entities predicted empty but have links {loss.filter(~sing & (t['np'] == 0)).sum() / loss.sum():.3f}; "
          f"partial {loss.filter(~sing & (t['np'] > 0)).sum() / loss.sum():.3f}")
    print(f"avg loss per S1 = {loss.mean():.5f}")
    s1 = pl.read_parquet(d / "source1.parquet", columns=["business_name", "business_address", "country"])
    tg = {j: pl.read_parquet(d / f"source{j}.parquet", columns=["business_name", "business_address"]) for j in (2, 3)}
    pp = oof.select(k + ["p2"])
    for title, df in (("FALSE POSITIVES", fp), ("MODEL FALSE NEGATIVES", fn_model)):
        print(f"\n=== {title} ===")
        ex = df.join(pp, on=k, how="left").sample(min(n_ex, len(df)), seed=0)
        for r in ex.iter_rows(named=True):
            a = s1.row(r["s1_idx"]); b = tg[r["tgt"]].row(r["tgt_idx"])
            print(f"[{a[2]}] p2={r['p2']:.3f}  {a[0]!r} | {a[1]!r}\n        S{r['tgt']}: {b[0]!r} | {b[1]!r}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", type=int, default=10)
    main(ap.parse_args().examples)
