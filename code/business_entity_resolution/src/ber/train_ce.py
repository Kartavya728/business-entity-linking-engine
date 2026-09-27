"""Train cross-encoders on encoder-fold (E) candidates, and score pairs with them.

Usage:
  python -m ber.train_ce train --model M [--n-s1 N] [--epochs E] [--seed S]   # fit on E-fold candidates
  python -m ber.train_ce score --split train|test --model M [--shard i --nshard n]
  python -m ber.train_ce merge --split train|test --model M --nshard n         # join shard files
Models are listed in ce_registry.py. Scoring covers pairs with p1 >= CE_MIN_P1 and is
incremental (pairs already in the score file are skipped). With --nshard, each process scores
every n-th pair into <file>.shard<i>; `merge` appends the shards to the score file, so several
GPUs can score one model in parallel.
"""
import argparse

import numpy as np
import polars as pl

from .ce_registry import CE_MODELS
from .config import split_dir
from .splits import ENC_FOLDS, s1_folds

CE_MIN_P1 = 0.003
HARD_RANK = 12  # negatives: candidates within this fused dense rank
KEYS = ["s1_idx", "tgt", "tgt_idx"]


def _texts(split, col="text"):
    d = split_dir(split)
    if col == "ftext":  # field-structured text: raw view || canonical view
        return {k: [f"{a} || {b}" for a, b in pl.read_parquet(d / f"source{k}.parquet", columns=["text", "ctext"]).iter_rows()]
                for k in (1, 2, 3)}
    return {k: pl.read_parquet(d / f"source{k}.parquet", columns=[col])[col].to_list() for k in (1, 2, 3)}


def train(model: str, n_s1: int = 0, epochs: int = 0, seed: int = -1):
    from .crossencoder import train_crossencoder
    m = CE_MODELS[model]
    n_s1 = n_s1 or m["n_s1"]; epochs = epochs or m["epochs"]; seed = m["seed"] if seed < 0 else seed
    d = split_dir("train")
    cand = pl.read_parquet(d / "candidates.parquet", columns=[*KEYS, "dense_r2", "sparse_r2"])
    gt = pl.read_parquet(d / "gt.parquet").with_columns(pl.lit(1, pl.Int8).alias("y"))
    folds = s1_folds(pl.read_parquet(d / "source1.parquet", columns=["idx"]).height)
    e_s1 = np.flatnonzero(np.isin(folds, ENC_FOLDS))
    e_s1 = np.random.default_rng(seed).choice(e_s1, size=min(n_s1, len(e_s1)), replace=False)
    c = cand.filter(pl.col("s1_idx").is_in(e_s1))
    c = c.join(gt, on=KEYS, how="left").with_columns(pl.col("y").fill_null(0))
    if m["band"]:  # LLM matchers: every E-fold candidate inside the hard-pair band
        lo, hi = m["band"]
        p1 = pl.read_parquet(d / "p1.parquet", columns=[*KEYS, "p1"])
        c = c.join(p1, on=KEYS).filter((pl.col("p1") >= lo) & (pl.col("p1") < hi)).drop("p1")
    else:
        c = c.filter((pl.col("y") == 1) | (pl.col("dense_r2") < HARD_RANK) | (pl.col("sparse_r2") < m["sparse_neg"]))
    print(f"[ce] {model}: {len(c):,} training pairs from {len(e_s1):,} E-fold S1, pos rate {c['y'].mean():.3f}, "
          f"{epochs} epoch(s)", flush=True)
    T = _texts("train", m["text"])
    a = [T[1][i] for i in c["s1_idx"].to_list()]
    b = [T[k][j] for k, j in zip(c["tgt"].to_list(), c["tgt_idx"].to_list())]
    train_crossencoder(a, b, c["y"].to_numpy(), str(m["out"]), epochs=epochs, batch_size=m["bs"], lr=m["lr"],
                       base=m["base"], swap=m["swap"], max_len=m["max_len"], seed=42 + seed, prec=m["prec"],
                       lora=m["lora"], prompt=m["prompt"], yesno=m["yesno"])


CKPT_PAIRS = 1_000_000  # scoring writes its output every this many pairs, so a killed job resumes


def score(split: str, model: str, shard: int = 0, nshard: int = 1, limit: int = 0):
    from .crossencoder import CrossEncoder
    m = CE_MODELS[model]
    d = split_dir(split)
    p = pl.read_parquet(d / "p1.parquet", columns=[*KEYS, "p1"])
    lo, hi = m["band"] or (CE_MIN_P1, 2.0)
    p = p.filter((pl.col("p1") >= lo) & (pl.col("p1") < hi)).select(KEYS).sort(KEYS)
    if split == "train" and m["band"]:
        # LLM matchers score only what stage 2 needs on train: the pairs of its training S1
        # (subset_s1.npy) and every rival claim on the same targets, so each training row gets the
        # same S1- and target-side context as on test (where every band pair is scored)
        sub = np.load(d / "subset_s1.npy")
        rivals = p.filter(pl.col("s1_idx").is_in(sub)).select("tgt", "tgt_idx").unique()
        p = p.join(rivals, on=["tgt", "tgt_idx"], how="semi").sort(KEYS)
    col, fname = m["col"], m["file"]
    out = d / (f"{fname}.smoke" if limit else f"{fname}.shard{shard}" if nshard > 1 else fname)
    old = pl.read_parquet(d / fname) if (d / fname).exists() else None
    if old is not None:  # incremental: only score pairs not scored before
        p = p.join(old.select(KEYS), on=KEYS, how="anti").sort(KEYS)
    if nshard > 1:  # shard assignment first, so a resumed shard keeps the same pairs
        p = p.gather_every(nshard, offset=shard)
    part = pl.read_parquet(out) if nshard > 1 and not limit and out.exists() else None
    if part is not None:  # resumed shard: skip what it already wrote
        p = p.join(part.select(KEYS), on=KEYS, how="anti").sort(KEYS)
    if limit:
        p = p.head(limit)
    print(f"[ce] scoring {len(p):,} new {split} pairs with {model} (shard {shard}/{nshard})", flush=True)
    base = None if limit else part if nshard > 1 else old  # rows kept in front of the new scores
    if not len(p):
        if base is None:
            p.with_columns(pl.lit(None, pl.Float32).alias(col)).write_parquet(out)
        return
    T = _texts(split, m["text"])
    ce = CrossEncoder.load(str(m["out"]), max_len=m["max_len"], prec=m["prec"])
    new = []
    for s in range(0, len(p), CKPT_PAIRS):
        q = p.slice(s, CKPT_PAIRS)
        a = [T[1][i] for i in q["s1_idx"].to_list()]
        b = [T[k][j] for k, j in zip(q["tgt"].to_list(), q["tgt_idx"].to_list())]
        new.append(q.with_columns(pl.Series(col, ce.predict(a, b, batch_size=m["score_bs"]))))
        tmp = out.with_name(out.name + ".tmp")  # write + rename: a kill never leaves a torn file
        pl.concat(([base] if base is not None else []) + new).write_parquet(tmp)
        tmp.replace(out)
        print(f"[ce] {model} {split}: {min(s + CKPT_PAIRS, len(p)):,}/{len(p):,} pairs written to {out.name}", flush=True)


def merge(split: str, model: str, nshard: int):
    m = CE_MODELS[model]
    d = split_dir(split)
    parts = [pl.read_parquet(d / f"{m['file']}.shard{i}") for i in range(nshard)]
    if (d / m["file"]).exists():
        parts.insert(0, pl.read_parquet(d / m["file"]))
    out = pl.concat(parts).unique(KEYS, keep="first")
    out.write_parquet(d / m["file"])
    for i in range(nshard):
        (d / f"{m['file']}.shard{i}").unlink()
    print(f"[ce] merged {nshard} shards -> {d / m['file']} ({len(out):,} pairs)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "score", "merge"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--model", default="small", choices=list(CE_MODELS))
    ap.add_argument("--n-s1", type=int, default=0, help="E-fold S1 entities to train on (0 = registry default)")
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: score only this many pairs")
    a = ap.parse_args()
    if a.cmd == "train":
        train(a.model, a.n_s1, a.epochs, a.seed)
    elif a.cmd == "score":
        score(a.split, a.model, a.shard, a.nshard, a.limit)
    else:
        merge(a.split, a.model, a.nshard)
