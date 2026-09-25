"""Stage 4: two-stage GBDT matcher (XGBoost, Apache-2.0, GPU).

Stage 1: pair features + retrieval scores/ranks + retrieval-group context -> p1.
Stage 2: stage-1 features + competition context computed from p1:
         * within the S1's candidate list (rank, gap to best, sum, count > .5)
         * within the target's claimant list (each S2/S3 record belongs to <= 1 S1)
         * cross-source agreement (best p1 among the S1's candidates in the other source)
         -> p2 (final match probability).
Training uses S1-grouped folds over ranker folds R (3..9); E-fold rows (encoder
training entities) are never trained on, only scored, so they act as realistic
competitors in the context features.
"""
import json

import numpy as np
import polars as pl
import xgboost as xgb

from .config import WORK_DIR
from .features import group_context

KEYS = ["s1_idx", "tgt", "tgt_idx"]
NON_FEATS = set(KEYS) | {"y", "cv", "fold", "p1", "p2"}
MODEL_DIR = WORK_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

PARAMS = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", device="cuda",
              max_depth=9, eta=0.08, subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
              reg_lambda=2.0, max_bin=256)
N_ROUNDS = 1500


def feat_cols(df: pl.DataFrame):
    return [c for c in df.columns if c not in NON_FEATS]


def to_x(df: pl.DataFrame, cols):
    return df.select(pl.col(c).cast(pl.Float32) for c in cols).to_numpy()


def stage2_context(df: pl.DataFrame, p: str = "p1") -> pl.DataFrame:
    """Competition features derived from a pair probability column."""
    df = group_context(df, p, "pc")
    r_t = pl.col(p).rank("ordinal", descending=True).over(["tgt", "tgt_idx"])
    r_s = pl.col(p).rank("ordinal", descending=True).over(["s1_idx", "tgt"])
    df = df.with_columns(
        pl.when(r_t == 2).then(pl.col(p)).otherwise(None).max().over(["tgt", "tgt_idx"]).fill_null(0).alias("_t2"),
        pl.when(r_s == 2).then(pl.col(p)).otherwise(None).max().over(["s1_idx", "tgt"]).fill_null(0).alias("_s2"),
        pl.col(p).sum().over("s1_idx").alias("pc_s1sum_all"),
        (pl.col(p) > 0.5).sum().over("s1_idx").cast(pl.Float32).alias("pc_s1cnt50_all"),
        (pl.col(p) > 0.5).sum().over(["s1_idx", "tgt"]).cast(pl.Float32).alias("pc_s1cnt50"),
    )
    df = df.with_columns(
        # margin to the best *other* claimant of this target / other candidate of this S1
        (pl.col(p) - pl.when(r_t == 1).then(pl.col("_t2")).otherwise(pl.col("pc_tmax"))).alias("pc_tmargin"),
        (pl.col(p) - pl.when(r_s == 1).then(pl.col("_s2")).otherwise(pl.col("pc_s1max"))).alias("pc_s1margin"),
    ).drop("_t2", "_s2")
    other = (df.group_by(["s1_idx", "tgt"]).agg(pl.col(p).max().alias("_m"))
               .with_columns((5 - pl.col("tgt")).cast(pl.Int8).alias("tgt")).rename({"_m": "pc_other_max"}))
    df = df.join(other, on=["s1_idx", "tgt"], how="left").with_columns(pl.col("pc_other_max").fill_null(-1))
    return df


def fit_stage(df: pl.DataFrame, name: str, cols, n_cv: int):
    """Train n_cv fold models on R rows (cv >= 0); return OOF-style predictions for all rows.

    R rows get the prediction of the model that did not see their fold; E rows (cv == -1)
    get the average of all fold models.
    """
    cv = df["cv"].to_numpy(); y = df["y"].to_numpy().astype(np.float32)
    X = to_x(df, cols)
    pred = np.zeros(len(df), dtype=np.float32)
    e_rows = cv < 0
    for k in range(n_cv):
        tr = (cv >= 0) & (cv != k); va = cv == k
        dtr = xgb.QuantileDMatrix(X[tr], y[tr], max_bin=PARAMS["max_bin"])
        dva = xgb.DMatrix(X[va], y[va])
        m = xgb.train(PARAMS, dtr, N_ROUNDS, evals=[(dva, "va")], early_stopping_rounds=50, verbose_eval=250)
        pred[va] = m.predict(dva, iteration_range=(0, m.best_iteration + 1))
        if e_rows.any():
            pred[e_rows] += m.predict(xgb.DMatrix(X[e_rows]), iteration_range=(0, m.best_iteration + 1)) / n_cv
        m.save_model(MODEL_DIR / f"{name}_cv{k}.json")
        print(f"[ranker] {name} fold {k}: best_it {m.best_iteration} "
              f"logloss {m.best_score:.5f}", flush=True)
        del dtr, dva
    json.dump({"cols": cols, "n_cv": n_cv}, open(MODEL_DIR / f"{name}_meta.json", "w"))
    return pred


def predict_stage(df: pl.DataFrame, name: str) -> np.ndarray:
    meta = json.load(open(MODEL_DIR / f"{name}_meta.json"))
    X = xgb.DMatrix(to_x(df, meta["cols"]))
    out = np.zeros(len(df), dtype=np.float32)
    for k in range(meta["n_cv"]):
        m = xgb.Booster(); m.load_model(MODEL_DIR / f"{name}_cv{k}.json")
        m.set_param({"device": "cuda"})
        out += m.predict(X) / meta["n_cv"]
    return out
