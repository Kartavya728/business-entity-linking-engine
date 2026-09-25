"""Stage 2: candidate generation (blocking).

For each target source (S2, S3) separately and within each country:
  * dense  : S1 -> target top-K by fine-tuned bi-encoder cosine        (typos, reorderings)
  * rdense : target -> S1 top-R (reverse); adds pairs where the target's best S1 is this S1
  * sparse : S1 -> target top-K by IDF-weighted typed-token cosine     (rare exact tokens)
The union is re-scored so that every pair has all retrieval scores, then fused with a
reciprocal-rank score and capped at `cap` targets per S1 per source. The capped set is
exactly what the matching model scores (= candidate_pairs.tsv).
"""
import argparse
import time

import numpy as np
import polars as pl
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from .biencoder import BiEncoder
from .config import WORK_DIR, split_dir
from .retrieval import knn_grouped, sparse_topk, token_doc, topk_to_frame

DEFAULTS = dict(k_dense=40, k_rev=5, k_sparse=40, cap=25, max_df=20000)


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


def _sparse_rowdot(q_docs, d_docs, qg, dg, a, b, max_df):
    """Typed-token IDF cosine for aligned pairs, fitted per country like sparse_topk."""
    out = np.zeros(len(a), dtype=np.float32)
    q_docs = np.asarray(q_docs, dtype=object); d_docs = np.asarray(d_docs, dtype=object)
    pg = qg[a]
    for g in np.unique(pg):
        m = np.flatnonzero(pg == g)
        di = np.flatnonzero(dg == g)
        vec = TfidfVectorizer(analyzer=str.split, sublinear_tf=True, dtype=np.float32,
                              max_df=max_df if len(di) > max_df else 1.0)
        vec.fit(d_docs[di])
        ua, ia = np.unique(a[m], return_inverse=True)
        ub, ib = np.unique(b[m], return_inverse=True)
        Q = vec.transform(q_docs[ua]); D = vec.transform(d_docs[ub])
        for s in range(0, len(m), 2_000_000):
            sl = slice(s, s + 2_000_000)
            out[m[sl]] = np.asarray(Q[ia[sl]].multiply(D[ib[sl]]).sum(1)).ravel()
    return out


def build_candidates(split: str, enc_path: str, p=DEFAULTS):
    d = split_dir(split)
    s1 = pl.read_parquet(d / "source1.parquet")
    enc = BiEncoder(enc_path)
    t = time.time()
    E1 = enc.encode(s1["text"].to_list())
    print(f"[cand] encoded S1 {len(s1):,} in {time.time() - t:.0f}s", flush=True)
    q_docs = token_doc(s1)
    parts = []
    for k, src in ((2, "source2"), (3, "source3")):
        tg = pl.read_parquet(d / f"{src}.parquet")
        t = time.time()
        Et = enc.encode(tg["text"].to_list())
        g1, gt = _codes(s1["country"], tg["country"])
        di, ds = knn_grouped(E1, Et, g1, gt, p["k_dense"])
        ri, rs = knn_grouped(Et, E1, gt, g1, p["k_rev"])
        d_docs = token_doc(tg)
        si, ss = sparse_topk(q_docs, d_docs, g1, gt, p["k_sparse"], max_df=p["max_df"])
        print(f"[cand] S{k}: retrieval done in {time.time() - t:.0f}s", flush=True)
        fd = topk_to_frame(di, ds, "dense")
        fr = topk_to_frame(ri, rs, "rdense", q_is_s1=False)
        fs = topk_to_frame(si, ss, "sparse")
        u = (pl.concat([f.select("s1_idx", "tgt_idx") for f in (fd, fr, fs)]).unique())
        u = (u.join(fd.drop("dense_s"), on=["s1_idx", "tgt_idx"], how="left")
              .join(fr.drop("rdense_s"), on=["s1_idx", "tgt_idx"], how="left")
              .join(fs.drop("sparse_s"), on=["s1_idx", "tgt_idx"], how="left"))
        a = u["s1_idx"].to_numpy(); b = u["tgt_idx"].to_numpy()
        u = u.with_columns(
            pl.Series("dense_s", _rowdot(E1, Et, a, b)),
            pl.Series("sparse_s", _sparse_rowdot(q_docs, d_docs, g1, gt, a, b, p["max_df"])),
        )
        # Reciprocal-rank fusion; missing ranks count as "just outside the list".
        u = u.with_columns(
            (1 / (60 + pl.col("dense_r").fill_null(p["k_dense"]).cast(pl.Float32))
             + 1 / (60 + pl.col("sparse_r").fill_null(p["k_sparse"]).cast(pl.Float32))
             + 1 / (60 + pl.col("rdense_r").fill_null(p["k_rev"] * 4).cast(pl.Float32))
             ).alias("rrf"),
            pl.lit(k, pl.Int8).alias("tgt"),
        )
        u = u.with_columns(pl.col("rrf").rank("ordinal", descending=True).over("s1_idx")
                           .cast(pl.Int16).alias("rrf_r"))
        u = u.filter(pl.col("rrf_r") <= p["cap"])
        parts.append(u)
        del Et
        torch.cuda.empty_cache()
    cand = pl.concat(parts, how="diagonal_relaxed")
    cand.write_parquet(d / "candidates.parquet")
    print(f"[cand] {split}: {len(cand):,} candidate pairs "
          f"({len(cand) / len(s1):.1f} per S1)", flush=True)
    return cand


def blocking_recall(cand: pl.DataFrame, gt: pl.DataFrame, s1_subset=None):
    """Fraction of true pairs present in candidates (optionally over a subset of S1)."""
    g = gt if s1_subset is None else gt.filter(pl.col("s1_idx").is_in(s1_subset))
    hit = g.join(cand.select("s1_idx", "tgt", "tgt_idx"), on=["s1_idx", "tgt", "tgt_idx"], how="semi")
    return len(hit) / max(1, len(g))


def prune(split: str, cap: int):
    """Keep the top-`cap` fused candidates per S1 per source (full set kept as candidates_full)."""
    d = split_dir(split)
    full = d / "candidates_full.parquet"
    if not full.exists():
        (d / "candidates.parquet").rename(full)
    c = pl.read_parquet(full).filter(pl.col("rrf_r") <= cap)
    c.write_parquet(d / "candidates.parquet")
    n1 = pl.read_parquet(d / "source1.parquet", columns=["idx"]).height
    print(f"[cand] {split}: pruned to cap {cap}: {len(c):,} pairs ({len(c) / n1:.2f} per S1)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--enc", default=str(WORK_DIR / "biencoder"))
    ap.add_argument("--cap", type=int, default=DEFAULTS["cap"])
    ap.add_argument("--prune", action="store_true", help="only re-cap existing candidates")
    a = ap.parse_args()
    if a.prune:
        prune(a.split, a.cap)
    else:
        build_candidates(a.split, a.enc, {**DEFAULTS, "cap": a.cap})
