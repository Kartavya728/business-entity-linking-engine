"""Stage 2 (v5): hybrid GPU blocking = dense bi-encoder + reverse dense + GPU IDF-sparse.

Why: the bi-encoder was trained on English/Indian text and ranks unseen-language records
by generic words (French 'Amis', 'Comite', 'SARL'), pushing true French matches down to
ranks 10-40. An IDF-weighted typed-token index fitted on each country's own records
down-weights generic words automatically, so the union restores recall for any language.

Per target source and within each country:
  dense  : S1 -> target top-K_DENSE by cosine          (dense_r / dense_s)
  rdense : target -> S1 top-K_REV (reverse)            (rdense_r)
  sparse : S1 -> target top-K_SPARSE by IDF cosine     (sparse_r / sparse_s)
Every union pair gets both exact scores and union ranks (dense_r2, sparse_r2). The pair
is kept when dense_r2 < CD or sparse_r2 < CS or rdense_r == 0 (see apply_rule).

  python -m ber.candidates_v5 --split train|test          # build candidates_full
  python -m ber.candidates_v5 --split S --prune CD CS     # write candidates.parquet
"""
import argparse
import time

import numpy as np
import polars as pl
import torch

from .biencoder import BiEncoder
from .candidates import _codes, _rowdot
from .config import WORK_DIR, split_dir
from .gpu_sparse import pair_dot, sparse_topk_gpu
from .retrieval import knn_grouped, token_doc_v2, topk_to_frame

K_DENSE, K_REV, K_SPARSE = 40, 5, 25


def apply_rule(u: pl.DataFrame, cd: int, cs: int) -> pl.DataFrame:
    return u.filter((pl.col("dense_r2") < cd) | (pl.col("sparse_r2") < cs) | (pl.col("rdense_r") == 0))


def build(split: str, enc_path: str):
    d = split_dir(split)
    s1 = pl.read_parquet(d / "source1.parquet", columns=["idx", "text", "country", "name_f", "name_fk", "addr_n", "nums"])
    enc = BiEncoder(enc_path)
    t = time.time()
    E1 = enc.encode(s1["text"].to_list(), log_every=0)
    q_docs = token_doc_v2(s1)
    print(f"[cand5] S1 encoded in {time.time() - t:.0f}s", flush=True)
    for k in (2, 3):
        t = time.time()
        tg = pl.read_parquet(d / f"source{k}.parquet", columns=["idx", "text", "country", "name_f", "name_fk", "addr_n", "nums"])
        Et = enc.encode(tg["text"].to_list(), log_every=0)
        g1, gt = _codes(s1["country"], tg["country"])
        di, ds = knn_grouped(E1, Et, g1, gt, K_DENSE)
        ri, rs = knn_grouped(Et, E1, gt, g1, K_REV)
        si, ss, mats = sparse_topk_gpu(q_docs, token_doc_v2(tg), g1, gt, K_SPARSE)
        fd = topk_to_frame(di, ds, "dense"); fr = topk_to_frame(ri, rs, "rdense", q_is_s1=False)
        fs = topk_to_frame(si, ss, "sparse")
        u = pl.concat([f.select("s1_idx", "tgt_idx") for f in (fd, fr, fs)]).unique()
        u = (u.join(fd.drop("dense_s"), on=["s1_idx", "tgt_idx"], how="left")
              .join(fr, on=["s1_idx", "tgt_idx"], how="left")
              .join(fs.drop("sparse_s"), on=["s1_idx", "tgt_idx"], how="left"))
        a = u["s1_idx"].to_numpy(); b = u["tgt_idx"].to_numpy()
        u = u.with_columns(pl.Series("dense_s", _rowdot(E1, Et, a, b)),
                           pl.Series("sparse_s", pair_dot(mats, g1, a, b)), pl.lit(k, pl.Int8).alias("tgt"))
        u = u.with_columns(
            pl.col("dense_s").rank("ordinal", descending=True).over("s1_idx").cast(pl.Int16).sub(1).alias("dense_r2"),
            pl.col("sparse_s").rank("ordinal", descending=True).over("s1_idx").cast(pl.Int16).sub(1).alias("sparse_r2"),
        )
        u.write_parquet(d / f"cand5_s{k}.parquet")
        print(f"[cand5] S{k}: {len(u):,} union pairs in {time.time() - t:.0f}s", flush=True)
        del Et, mats
        torch.cuda.empty_cache()
    full = pl.concat([pl.read_parquet(d / f"cand5_s{k}.parquet") for k in (2, 3)], how="diagonal_relaxed")
    full.write_parquet(d / "candidates5_full.parquet")
    for k in (2, 3):
        (d / f"cand5_s{k}.parquet").unlink()
    print(f"[cand5] {split}: {len(full):,} union pairs ({len(full) / len(s1):.1f} per S1)", flush=True)


def prune(split: str, cd: int, cs: int):
    d = split_dir(split)
    c = apply_rule(pl.read_parquet(d / "candidates5_full.parquet"), cd, cs)
    c.write_parquet(d / "candidates.parquet")
    n1 = pl.read_parquet(d / "source1.parquet", columns=["idx"]).height
    print(f"[cand5] {split}: rule dense<{cd} | sparse<{cs} | rev0 -> {len(c):,} pairs ({len(c) / n1:.2f} per S1)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--enc", default=str(WORK_DIR / "biencoder_v2"))
    ap.add_argument("--prune", nargs=2, type=int, metavar=("CD", "CS"))
    a = ap.parse_args()
    prune(a.split, *a.prune) if a.prune else build(a.split, a.enc)
