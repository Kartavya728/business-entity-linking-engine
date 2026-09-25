"""Deterministic S1-level folds for the training split.

fold = 0..9 per S1 entity (random, seeded). Usage convention:
  folds 0-2  (30%) -> "E": train neural encoders (bi-encoder, cross-encoder)
  folds 3-9  (70%) -> "R": train/validate the GBDT ranker with fold-wise OOF
Keeping E and R disjoint means neural scores used as ranker features are never
computed on the encoders' own training entities.
"""
import numpy as np

from .config import SEED

N_FOLDS = 10
ENC_FOLDS = (0, 1, 2)


def s1_folds(n_s1: int) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    return (rng.permutation(n_s1) % N_FOLDS).astype(np.int8)


def enc_mask(n_s1: int) -> np.ndarray:
    return np.isin(s1_folds(n_s1), ENC_FOLDS)
