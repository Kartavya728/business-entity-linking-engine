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
import gc
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
    """Competition features derived from a pair probability column (numpy, sort-based)."""
    from .features import _keys, grp_stats
    df = group_context(df, p, "pc")
    gs, gt = _keys(df)
    x = df[p].to_numpy().astype(np.float32)
    a = grp_stats(gs, x); b = grp_stats(gt, x)
    s1 = df["s1_idx"].to_numpy().astype(np.int64); tg = df["tgt"].to_numpy().astype(np.int64)
    hi = (x > 0.5).astype(np.float32)
    n1 = int(s1.max()) + 1
    s1sum = np.bincount(s1, weights=x, minlength=n1)[s1].astype(np.float32)
    s1cnt = np.bincount(s1, weights=hi, minlength=n1)[s1].astype(np.float32)
    gcnt = np.bincount(gs, weights=hi)[gs].astype(np.float32)
    # best p1 of the same S1 in the other target source (-1 when none)
    M = np.full((n1, 4), -1.0, dtype=np.float32)
    M[s1, tg] = a["max"]  # every row of a (S1, source) group carries the same group max
    other = M[s1, 5 - tg]
    return df.with_columns(
        pl.Series("pc_s1sum_all", s1sum), pl.Series("pc_s1cnt50_all", s1cnt), pl.Series("pc_s1cnt50", gcnt),
        # margin to the best *other* claimant of this target / other candidate of this S1
        pl.Series("pc_tmargin", x - np.where(b["rank"] == 1, b["second"], b["max"])),
        pl.Series("pc_s1margin", x - np.where(a["rank"] == 1, a["second"], a["max"])),
        pl.Series("pc_other_max", other),
    )


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
        path = MODEL_DIR / f"{name}_cv{k}.json"
        if path.exists():  # resume: reuse a fold model trained earlier with the same columns
            m = xgb.Booster(); m.load_model(path); m.set_param({"device": "cuda"})
            it = (0, m.num_boosted_rounds())
            print(f"[ranker] {name} fold {k}: reusing saved model ({it[1]} rounds)", flush=True)
        else:
            dtr = xgb.QuantileDMatrix(X[tr], y[tr], max_bin=PARAMS["max_bin"])
            dva = xgb.DMatrix(X[va], y[va])
            m = xgb.train(PARAMS, dtr, N_ROUNDS, evals=[(dva, "va")], early_stopping_rounds=50, verbose_eval=250)
            it, best = (0, m.best_iteration + 1), m.best_score
            m = m[: it[1]]  # keep only the best iterations
            m.save_model(path)
            print(f"[ranker] {name} fold {k}: best_it {it[1] - 1} logloss {best:.5f}", flush=True)
            del dtr, dva
        pred[va] = m.predict(xgb.DMatrix(X[va]))
        if e_rows.any():
            pred[e_rows] += m.predict(xgb.DMatrix(X[e_rows])) / n_cv
        del m
        gc.collect()
    json.dump({"cols": cols, "n_cv": n_cv}, open(MODEL_DIR / f"{name}_meta.json", "w"))
    return pred


def predict_stage(df: pl.DataFrame, name: str, chunk: int = 5_000_000) -> np.ndarray:
    """Fold-average prediction, in row chunks to bound GPU memory."""
    meta = json.load(open(MODEL_DIR / f"{name}_meta.json"))
    models = []
    for k in range(meta["n_cv"]):
        m = xgb.Booster(); m.load_model(MODEL_DIR / f"{name}_cv{k}.json")
        m.set_param({"device": "cuda"})
        models.append(m)
    out = np.zeros(len(df), dtype=np.float32)
    for s in range(0, len(df), chunk):
        X = xgb.DMatrix(to_x(df.slice(s, chunk), meta["cols"]))
        out[s:s + chunk] = sum(m.predict(X) for m in models) / len(models)
        del X
    return out


def subset_cv(s1_idx: np.ndarray, d, n_cv: int) -> np.ndarray:
    """cv fold of each row's S1 when it is in the training subset (work/train/subset_s1.npy), else -1."""
    from .splits import s1_folds
    n_s1 = pl.read_parquet(d / "source1.parquet", columns=["idx"]).height
    folds = s1_folds(n_s1)
    insub = np.zeros(n_s1, dtype=bool)
    insub[np.load(d / "subset_s1.npy")] = True
    cv_all = np.where(insub, (folds - 3) % n_cv, -1).astype(np.int8)
    return cv_all[s1_idx]


def predict_stage_oof(df: pl.DataFrame, name: str, d) -> np.ndarray:
    """Stage prediction where subset S1 rows use the fold model that did not train on them."""
    meta = json.load(open(MODEL_DIR / f"{name}_meta.json"))
    n_cv = meta["n_cv"]
    cv = subset_cv(df["s1_idx"].to_numpy(), d, n_cv)
    X = xgb.DMatrix(to_x(df, meta["cols"]))
    out = np.zeros(len(df), dtype=np.float32)
    avg = cv < 0
    for k in range(n_cv):
        m = xgb.Booster(); m.load_model(MODEL_DIR / f"{name}_cv{k}.json")
        m.set_param({"device": "cuda"})
        p = m.predict(X)
        out[cv == k] = p[cv == k]
        out[avg] += p[avg] / n_cv
    return out


LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=255, min_data_in_leaf=100,
                  feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0,
                  max_bin=255, num_threads=40, verbose=-1)


def fit_stage_lgb(df: pl.DataFrame, name: str, cols, n_cv: int, rounds: int = 3000):
    """LightGBM (MIT) counterpart of fit_stage, used as a second model family in the ensemble."""
    import lightgbm as lgb
    cv = df["cv"].to_numpy(); y = df["y"].to_numpy().astype(np.float32)
    X = to_x(df, cols)
    pred = np.zeros(len(df), dtype=np.float32)
    for k in range(n_cv):
        tr = (cv >= 0) & (cv != k); va = cv == k
        path = MODEL_DIR / f"{name}_cv{k}.txt"
        if path.exists():
            m = lgb.Booster(model_file=str(path))
        else:
            dtr = lgb.Dataset(X[tr], y[tr], free_raw_data=True)
            dva = lgb.Dataset(X[va], y[va], reference=dtr)
            m = lgb.train(LGB_PARAMS, dtr, rounds, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)])
            m.save_model(str(path), num_iteration=m.best_iteration)
            print(f"[ranker] {name} fold {k}: best_it {m.best_iteration} "
                  f"logloss {m.best_score['valid_0']['binary_logloss']:.5f}", flush=True)
        pred[va] = m.predict(X[va])
        del m
        gc.collect()
    json.dump({"cols": cols, "n_cv": n_cv, "kind": "lgb"}, open(MODEL_DIR / f"{name}_meta.json", "w"))
    return pred


def predict_stage_lgb(df: pl.DataFrame, name: str, chunk: int = 5_000_000) -> np.ndarray:
    import lightgbm as lgb
    meta = json.load(open(MODEL_DIR / f"{name}_meta.json"))
    models = [lgb.Booster(model_file=str(MODEL_DIR / f"{name}_cv{k}.txt")) for k in range(meta["n_cv"])]
    out = np.zeros(len(df), dtype=np.float32)
    for s in range(0, len(df), chunk):
        X = to_x(df.slice(s, chunk), meta["cols"])
        out[s:s + chunk] = sum(m.predict(X, num_threads=40) for m in models) / len(models)
    return out
