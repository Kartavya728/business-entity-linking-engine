"""Stage 5b: label-free domain adaptation of the stage-2 matcher to an unseen country (France).

Why: French names are built from a small vocabulary ('Club', 'Amicale', 'Comite', 'Ecole'),
and French true copies often swap one such word for another at the *same* address. Training
countries never show this pattern, so stage 2 rejects about half of these pairs.

Label-free evidence: in the training countries, distractor records essentially never keep the
exact address (same-address rate 0.05% for non-matches vs 22% for matches). So for the
unseen country we build pseudo-labels:
  positive : identical non-empty address + similar filler-free name (token-set >= NS_POS),
             target claimed by no other same-address S1, or p2 > HI
  negative : p2 < LO and not a same-address near-name pair
and add them to the labelled training rows. The model is cross-fitted by S1 (fold k trains
on pseudo-labels of the other folds), so every adapted-country pair is scored by a model that
never saw its own pseudo-label. New features describe *which* words differ (their
within-country document frequency), letting the trees separate vocabulary swaps (common
words) from typos and distractor words.

  python -m ber.adapt fit --country France   # fold models + train OOF + adapted-country p2
  python -m ber.adapt predict                 # fold-average p2 for every test row
The train OOF (work/train/oof_adapt.parquet) is tuned like stage 2, so the adapted model is
used for the training countries only if it beats the stage-2 OOF score there.
"""
import argparse
import gc
import json
from multiprocessing import Pool

import numpy as np
import polars as pl
import xgboost as xgb
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from .config import split_dir
from .fillers import _doc_freq
from .ranker import KEYS, MODEL_DIR, PARAMS, feat_cols, subset_cv, to_x
from .train_ranker import stage2_frame

N_CV = 4
NS_POS, HI, LO = 80, 0.995, 0.005
MIN_P1 = 0.003
DT_COLS = ["dt_miss_dfmax", "dt_miss_dfmin", "dt_extra_dfmax", "dt_extra_dfmin", "dt_swap_jw",
           "dt_same_addr", "dt_ns"]

_G = {}


def _init(n1, a1, c1, n2, a2, df):
    _G.update(n1=n1, a1=a1, c1=c1, n2=n2, a2=a2, df=df)


def _rows(args):
    s_idx, t_src, t_idx = args
    n1, a1, c1, n2, a2, df = (_G[k] for k in ("n1", "a1", "c1", "n2", "a2", "df"))
    out = np.full((len(s_idx), len(DT_COLS)), np.nan, dtype=np.float32)
    for r, (s, g, j) in enumerate(zip(s_idx, t_src, t_idx)):
        A = n1[s]; B = n2[g][j]; D = df[c1[s]]
        sa, sb = set(A.split()), set(B.split())
        x, y = sa - sb, sb - sa
        fx = [D.get(t, 0.0) for t in x] or [-1.0]
        fy = [D.get(t, 0.0) for t in y] or [-1.0]
        jw = JaroWinkler.similarity(next(iter(x)), next(iter(y))) if len(x) == 1 and len(y) == 1 else -1.0
        same = float(bool(a2[g][j]) and a1[s] == a2[g][j])
        out[r] = (max(fx), min(fx), max(fy), min(fy), jw, same, fuzz.token_set_ratio(A, B))
    return out


def diff_token_feats(df: pl.DataFrame, split: str, procs: int = 32) -> pl.DataFrame:
    """Word-difference features for rows with p1 >= MIN_P1 (NaN elsewhere)."""
    d = split_dir(split)
    s1 = pl.read_parquet(d / "source1.parquet", columns=["name_f", "addr_n", "country"])
    tg = {k: pl.read_parquet(d / f"source{k}.parquet", columns=["name_f", "addr_n"]) for k in (2, 3)}
    dfreq = {}
    for c in s1["country"].unique().to_list():
        f = _doc_freq(s1.filter(pl.col("country") == c).select(pl.col("name_f").alias("name_n")))
        dfreq[c] = dict(zip(f["t"].to_list(), f["f"].to_list()))
    sel = np.flatnonzero(df["p1"].to_numpy() >= MIN_P1)
    s = df["s1_idx"].to_numpy()[sel]; g = df["tgt"].to_numpy()[sel]; j = df["tgt_idx"].to_numpy()[sel]
    chunks = [(s[i:i + 50_000], g[i:i + 50_000], j[i:i + 50_000]) for i in range(0, len(sel), 50_000)]
    n2 = {k: tg[k]["name_f"].to_numpy() for k in tg}; a2 = {k: tg[k]["addr_n"].to_numpy() for k in tg}
    # workers inherit the lookup arrays through fork (copy-on-write) instead of each receiving a
    # pickled copy, which exhausted RAM with many workers next to a large stage-2 frame
    import multiprocessing as mp
    _init(s1["name_f"].to_list(), s1["addr_n"].to_list(), s1["country"].to_list(),
          {k: v.tolist() for k, v in n2.items()}, {k: v.tolist() for k, v in a2.items()}, dfreq)
    with mp.get_context("fork").Pool(procs) as pool:
        res = pool.map(_rows, chunks)
    _G.clear()
    M = np.full((len(df), len(DT_COLS)), np.nan, dtype=np.float32)
    if res:
        M[sel] = np.concatenate(res)
    print(f"[adapt] {split}: word-difference features for {len(sel):,}/{len(df):,} rows", flush=True)
    return df.with_columns([pl.Series(c, M[:, i]) for i, c in enumerate(DT_COLS)])


def pseudo_labels(te: pl.DataFrame, p2: np.ndarray) -> np.ndarray:
    """1 / 0 / -1 (unlabelled) for the adapted-country rows."""
    ns = te["dt_ns"].to_numpy(); same = te["dt_same_addr"].to_numpy() == 1
    near = same & (ns >= NS_POS)
    # a target claimed by several same-address near-name S1 records is ambiguous
    k = te.select("tgt", "tgt_idx").with_columns(pl.Series("near", near))
    nclaim = k.with_columns(pl.col("near").cast(pl.Int32).sum().over("tgt", "tgt_idx").alias("n"))["n"].to_numpy()
    pos = (near & (nclaim == 1)) | (p2 > HI)
    neg = (p2 < LO) & ~near
    y = np.full(len(te), -1, dtype=np.int8)
    y[neg] = 0; y[pos] = 1
    return y


def main(country: str):
    dtr, dte = split_dir("train"), split_dir("test")
    # labelled rows (as in train_ranker.stage2)
    tr = stage2_frame("train")
    tr = tr.with_columns(pl.Series("cv", subset_cv(tr["s1_idx"].to_numpy(), dtr, N_CV))).filter(pl.col("cv") >= 0)
    gt = pl.read_parquet(dtr / "gt.parquet").with_columns(pl.lit(1, pl.Int8).alias("y"))
    tr = tr.join(gt, on=KEYS, how="left").with_columns(pl.col("y").fill_null(0))
    tr = diff_token_feats(tr, "train")
    # adapted-country rows
    te = stage2_frame("test")
    ctry = pl.read_parquet(dte / "source1.parquet", columns=["country"])["country"].to_numpy()
    te = te.filter(pl.Series(ctry[te["s1_idx"].to_numpy()] == country))
    te = diff_token_feats(te, "test")
    base = pl.read_parquet(dte / "p2.parquet").select(*KEYS, "p2")
    p2 = te.select(KEYS).join(base, on=KEYS, how="left")["p2"].fill_null(0.0).to_numpy()
    y = pseudo_labels(te, p2)
    print(f"[adapt] {country}: {len(te):,} rows; pseudo-positive {(y == 1).sum():,}, "
          f"negative {(y == 0).sum():,}, unlabelled {(y == -1).sum():,}", flush=True)
    rng = np.random.default_rng(0)
    fold_of_s1 = rng.integers(0, N_CV, int(te["s1_idx"].max()) + 1)
    te_cv = fold_of_s1[te["s1_idx"].to_numpy()]

    cols = [c for c in feat_cols(tr) if c in te.columns] + ["p1"]
    Xtr = to_x(tr, cols); ytr = tr["y"].to_numpy().astype(np.float32); cvtr = tr["cv"].to_numpy()
    Xte = to_x(te, cols)
    tr_keys = tr.select(KEYS, "y", "cv", "p1"); te_keys = te.select(KEYS)
    del tr, te
    gc.collect()
    lab = y >= 0
    out = np.zeros(len(te_keys), dtype=np.float32)
    oof = np.zeros(len(tr_keys), dtype=np.float32)
    for k in range(N_CV):
        a = cvtr != k; b = lab & (te_cv != k)
        X = np.concatenate([Xtr[a], Xte[b]]); Y = np.concatenate([ytr[a], y[b].astype(np.float32)])
        va = (cvtr == k)
        dtrain = xgb.QuantileDMatrix(X, Y, max_bin=PARAMS["max_bin"])
        dva = xgb.DMatrix(Xtr[va], ytr[va])
        m = xgb.train(PARAMS, dtrain, 1500, evals=[(dva, "va")], early_stopping_rounds=50, verbose_eval=250)
        m = m[: m.best_iteration + 1]
        m.save_model(MODEL_DIR / f"adapt_{country}_cv{k}.json")
        out[te_cv == k] = m.predict(xgb.DMatrix(Xte[te_cv == k]))
        oof[va] = m.predict(xgb.DMatrix(Xtr[va]))
        print(f"[adapt] fold {k}: best_it {m.num_boosted_rounds()} on {len(Y):,} rows", flush=True)
        del X, Y, dtrain, dva, m
        gc.collect()
    json.dump({"cols": cols, "n_cv": N_CV}, open(MODEL_DIR / f"adapt_{country}_meta.json", "w"))
    tr_keys.with_columns(pl.Series("p2", oof)).write_parquet(dtr / "oof_adapt.parquet")
    from .train_ranker import tune
    gt2 = pl.read_parquet(dtr / "gt.parquet")
    print("[adapt] OOF decision grid on training countries (compare with stage-2 best):", flush=True)
    tune(tr_keys.select(KEYS).with_columns(pl.Series("p2", oof)), gt2, np.load(dtr / "subset_s1.npy"))
    res = te_keys.with_columns(pl.Series("p2_adapt", out), pl.Series("p2_base", p2), pl.Series("y_pseudo", y))
    res.write_parquet(dte / f"p2_adapt_{country}.parquet")
    print(f"[adapt] {country}: mean p2 {p2.mean():.4f} -> {out.mean():.4f}; "
          f"accepted>0.5 {(p2 > 0.5).sum():,} -> {(out > 0.5).sum():,}", flush=True)


def predict():
    """Fold-average adapted-model p2 for all test rows (chunked)."""
    d = split_dir("test")
    meta = json.load(open(MODEL_DIR / "adapt_France_meta.json"))
    te = diff_token_feats(stage2_frame("test"), "test")
    models = []
    for k in range(meta["n_cv"]):
        m = xgb.Booster(); m.load_model(MODEL_DIR / f"adapt_France_cv{k}.json"); m.set_param({"device": "cuda"})
        models.append(m)
    out = np.zeros(len(te), dtype=np.float32)
    for s in range(0, len(te), 5_000_000):
        X = xgb.DMatrix(to_x(te.slice(s, 5_000_000), meta["cols"]))
        out[s:s + 5_000_000] = sum(m.predict(X) for m in models) / len(models)
    te.select(KEYS).with_columns(pl.Series("p2", out)).filter(pl.col("p2") > 0.01).write_parquet(d / "p2_adapt_all.parquet")
    print(f"[adapt] wrote fold-average p2 for {len(te):,} test rows", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "predict"])
    ap.add_argument("--country", default="France")
    a = ap.parse_args()
    main(a.country) if a.cmd == "fit" else predict()
