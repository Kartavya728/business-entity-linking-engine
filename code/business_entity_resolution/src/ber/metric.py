"""Exact competition metric: per-S1 F0.5, macro-averaged, singletons included."""
import numpy as np


def f05(n_pred: np.ndarray, n_true: np.ndarray, n_tp: np.ndarray) -> np.ndarray:
    """Vectorised per-entity F0.5.

    Empty prediction on an empty truth scores 1.0; any other case with tp == 0 scores 0.
    """
    n_pred = np.asarray(n_pred, dtype=np.float64)
    n_true = np.asarray(n_true, dtype=np.float64)
    n_tp = np.asarray(n_tp, dtype=np.float64)
    # F_beta = (1+b^2) tp / ((1+b^2) tp + b^2 fn + fp) = 1.25 tp / (1.25 tp + 0.25 fn + fp)
    denom = 1.25 * n_tp + 0.25 * (n_true - n_tp) + (n_pred - n_tp)
    out = np.where(denom > 0, 1.25 * n_tp / np.maximum(denom, 1e-12), 0.0)
    out = np.where((n_pred == 0) & (n_true == 0), 1.0, out)
    return out


def macro_f05(pred: dict, truth: dict, s1_universe) -> float:
    """pred/truth: s1 key -> set of target keys. s1_universe: all S1 keys evaluated."""
    n_pred, n_true, n_tp = [], [], []
    for s in s1_universe:
        p = pred.get(s, ()) or ()
        t = truth.get(s, ()) or ()
        p, t = set(p), set(t)
        n_pred.append(len(p)); n_true.append(len(t)); n_tp.append(len(p & t))
    return float(f05(n_pred, n_true, n_tp).mean())
