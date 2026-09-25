"""Stage 2: candidate generation (blocking), GPU-only.

For each target source (S2, S3) separately and within each country (partition key only):
  * dense  : S1 -> target top-K by the fine-tuned bi-encoder cosine (exact GPU kNN)
  * rdense : target -> S1 top-R (reverse kNN); adds pairs where this S1 is among the
             target's closest S1 records (helps S1 entities with many matches)
Every union pair gets its exact cosine; the pair is kept when it is within the S1's
top-`cap` by cosine or it is the target's single best S1 (reverse rank 0). The kept set
is exactly what the matching model scores (= candidate_pairs.tsv).

A CPU sparse token path was evaluated and dropped: it added only +0.03% recall over
dense retrieval (99.906% -> 99.935%) at ~25x the cost.
"""
import argparse
import time

import numpy as np
import polars as pl
import torch

from .biencoder import BiEncoder
from .config import WORK_DIR, split_dir
from .retrieval import knn_grouped, topk_to_frame

DEFAULTS = dict(k_dense=40, k_rev=5, cap=40)


def _codes(a: pl.Series, b: pl.Series):
    """Integer-encode two country columns with a shared vocabulary (open set)."""
    vocab = {c: i for i, c in enumerate(sorted(set(a.to_list()) | set(b.to_list())))}
    return (np.array([vocab[c] for c in a.to_list()], dtype=np.int32),
            np.array([vocab[c] for c in b.to_list()], dtype=np.int32))


def _rowdot(E1, E2, a, b, chunk=2_000_000):
    """Cosine for aligned pairs (a[i], b[i]) of fp16 embedding tables."""
    out = np.empty(len(a), dtype=np.float32)
    for s in range(0, len(a), chunk):
        x = E1[torch.from_numpy(a[s:s + chunk])].cuda().float()
        y = E2[torch.from_numpy(b[s:s + chunk])].cuda().float()
        out[s:s + chunk] = (x * y).sum(1).cpu().numpy()
    return out


def apply_cap(u: pl.DataFrame, cap: int) -> pl.DataFrame:
    """Keep top-`cap` by cosine per S1 (and source), plus reverse-best pairs."""
    return u.filter((pl.col("dense_r2") < cap) | (pl.col("rdense_r") == 0))


def build_candidates(split: str, enc_path: str, p=DEFAULTS):
    d = split_dir(split)
    s1 = pl.read_parquet(d / "source1.parquet", columns=["idx", "text", "country"])
    enc = BiEncoder(enc_path)
    t = time.time()
    E1 = enc.encode(s1["text"].to_list())
    print(f"[cand] encoded S1 {len(s1):,} in {time.time() - t:.0f}s", flush=True)
    for k in (2, 3):
        out = d / f"cand_s{k}.parquet"
        tg = pl.read_parquet(d / f"source{k}.parquet", columns=["idx", "text", "country"])
        t = time.time()
        Et = enc.encode(tg["text"].to_list())
        g1, gt = _codes(s1["country"], tg["country"])
        di, ds = knn_grouped(E1, Et, g1, gt, p["k_dense"])
        ri, rs = knn_grouped(Et, E1, gt, g1, p["k_rev"])
        fd = topk_to_frame(di, ds, "dense")
        fr = topk_to_frame(ri, rs, "rdense", q_is_s1=False)
        u = pl.concat([fd.select("s1_idx", "tgt_idx"), fr.select("s1_idx", "tgt_idx")]).unique()
        u = (u.join(fd.drop("dense_s"), on=["s1_idx", "tgt_idx"], how="left")
              .join(fr, on=["s1_idx", "tgt_idx"], how="left"))
        a = u["s1_idx"].to_numpy(); b = u["tgt_idx"].to_numpy()
        u = u.with_columns(pl.Series("dense_s", _rowdot(E1, Et, a, b)), pl.lit(k, pl.Int8).alias("tgt"))
        # final rank by exact cosine over the union (dense_r2), plus raw ranks as features
        u = u.with_columns(pl.col("dense_s").rank("ordinal", descending=True).over("s1_idx")
                           .cast(pl.Int16).sub(1).alias("dense_r2"))
        u = apply_cap(u, p["cap"])
        u.write_parquet(out)
        print(f"[cand] S{k}: {len(u):,} pairs in {time.time() - t:.0f}s", flush=True)
        del Et
        torch.cuda.empty_cache()
    cand = pl.concat([pl.read_parquet(d / f"cand_s{k}.parquet") for k in (2, 3)], how="diagonal_relaxed")
    cand.write_parquet(d / "candidates_full.parquet")
    print(f"[cand] {split}: {len(cand):,} candidate pairs ({len(cand) / len(s1):.1f} per S1)", flush=True)
    return cand


def prune(split: str, cap: int):
    """Write candidates.parquet = candidates_full capped at `cap` per S1 per source."""
    d = split_dir(split)
    c = apply_cap(pl.read_parquet(d / "candidates_full.parquet"), cap)
    c.write_parquet(d / "candidates.parquet")
    n1 = pl.read_parquet(d / "source1.parquet", columns=["idx"]).height
    print(f"[cand] {split}: cap {cap}: {len(c):,} pairs ({len(c) / n1:.2f} per S1)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--enc", default=str(WORK_DIR / "biencoder_v2"))
    ap.add_argument("--cap", type=int, default=DEFAULTS["cap"])
    ap.add_argument("--prune", action="store_true", help="only re-cap candidates_full")
    a = ap.parse_args()
    if a.prune:
        prune(a.split, a.cap)
    else:
        build_candidates(a.split, a.enc, {**DEFAULTS, "cap": a.cap})
